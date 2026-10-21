"""Cross-attention editor (editor_arch="xattn"): model, visual-token cache, training and serving, CPU only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt import visual_cache
from vla_precision.gr00t_rlt.core import CONTINUOUS, load_config, phase_active, representation_sha
from vla_precision.gr00t_rlt.model import (EditorHeads, Learner, XAttnEditorHeads, make_heads, spatial_features,
                                           visual_tokens)
from vla_precision.gr00t_rlt.replay import Replay
from test_direct_editor import editor_record


def xattn(c, **learner):
    return dict(c, learner=dict(c["learner"], editor_arch="xattn", **learner))


def inputs(c, n=3, phase=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(z=torch.randn(n, c["token"]["embed_dim"], generator=g), state=torch.randn(n, c["state_dim"], generator=g),
                reference=torch.randn(n, 30, 32, generator=g) * .5, phase=torch.full((n,), phase),
                vis=torch.randn(n, 512, 2048, generator=g).half())


def test_xattn_editor_starts_as_the_reference_is_smaller_and_keeps_phases_masks_and_bounds(tmp_path):
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    heads = make_heads(xattn(c))
    assert isinstance(heads, XAttnEditorHeads) and isinstance(make_heads(c), EditorHeads) and make_heads(c).arch == "mlp"
    assert sum(p.numel() for p in heads.actor.parameters()) < sum(p.numel() for p in make_heads(c).actor.parameters())
    x = inputs(c)
    act = lambda h, **kw: h.act(x["z"], x["state"], x["reference"], phase=x["phase"], vis=x["vis"], **kw)
    assert torch.equal(act(heads), x["reference"][..., :18])                      # zero-initialized output
    with pytest.raises(ValueError, match="visual prefix tokens"):
        heads.act(x["z"], x["state"], x["reference"], phase=x["phase"])
    torch.nn.init.constant_(heads.actor[-1].bias, 10.)
    edit = act(heads) - x["reference"][..., :18]
    active = torch.as_tensor(phase_active(c, "p3"))
    assert torch.allclose(edit[..., active], torch.full_like(edit[..., active], heads.scale * np.tanh(10.)))
    assert not edit[..., ~active].any()                                             # held channels untouched
    heads.set_editor_phases(["p1"])
    assert not (act(heads) - x["reference"][..., :18]).any()                       # P3 not an editor phase here
    heads.set_editor_phases(None)
    heads.set_editor_mask("moving-arm")
    x1 = inputs(c, phase=0)
    edit = heads.act(x1["z"], x1["state"], x1["reference"], phase=x1["phase"], vis=x1["vis"]) - x1["reference"][..., :18]
    assert edit[..., 6:12].abs().sum() > 0 and not edit[..., :6].any() and not edit[..., 12:].any()   # P1: right arm only


def test_cross_attention_reads_individual_tokens_not_only_their_quadrant_means(tmp_path):
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    torch.manual_seed(0)
    heads = make_heads(xattn(c))
    torch.nn.init.normal_(heads.actor[-1].weight, std=.1)
    x = inputs(c, n=1)
    moved = x["vis"].clone()
    moved[0, [0, 1]] = moved[0, [1, 0]]                    # swap two tokens of one quadrant: same 2x2 means
    pool = lambda v: spatial_features(torch.cat([v.float(), torch.zeros(1, 256, 2048)], 1),
                                      torch.arange(768)[None] < 512)
    assert torch.allclose(pool(x["vis"]), pool(moved), atol=1e-6)
    a = heads.act(x["z"], x["state"], x["reference"], phase=x["phase"], vis=x["vis"])
    b = heads.act(x["z"], x["state"], x["reference"], phase=x["phase"], vis=moved)
    assert (a - b).abs().max() > 1e-6                     # the MLP editor could not tell these apart


def test_visual_tokens_layout_and_cache_round_trip(tmp_path):
    prefix = torch.randn(1, 968, 2048)
    mask = torch.zeros(1, 968, dtype=torch.bool)
    mask[:, :512] = True
    assert torch.equal(visual_tokens(prefix, mask), prefix[:, :512])
    with pytest.raises(ValueError, match="two real views"):
        visual_tokens(prefix, torch.ones(1, 968, dtype=torch.bool))
    run = tmp_path / "run"
    (run / "observations").mkdir(parents=True)
    tokens = np.random.default_rng(0).standard_normal((968, 2048)).astype(np.float16)
    np.savez(run / "observations/a.npz", prefix=tokens, mask=mask[0].numpy(), contract_sha256=np.asarray("C"))
    cache = tmp_path / "cache"
    assert visual_cache.build(run, ["observations/a.npz"] * 2, cache, "C") == (0, 1)
    assert visual_cache.build(run, ["observations/a.npz"], cache, "C") == (1, 0)
    assert np.array_equal(np.load(visual_cache.cache_file(cache, "observations/a.npz")), tokens[:512])
    with pytest.raises(ValueError, match="recipe mismatch"):
        visual_cache.build(run, ["observations/a.npz"], tmp_path / "other", "D")
    vis, nxt = visual_cache.Loader(cache, np.asarray(["observations/a.npz"] * 3),
                                   np.asarray(["observations/a.npz"] * 3)).submit([0, 2]).result()
    assert vis.shape == nxt.shape == (2, 512, 2048) and vis.dtype == np.float16


def test_learner_trains_the_xattn_actor_from_the_visual_tokens(tmp_path):
    torch.set_num_threads(1)
    c = xattn(load_config(CONTINUOUS, "p1", workspace=tmp_path))
    c["learner"].update(actor_update_interval=1, critic_warmup_steps=0, actor_lr=1e-3)
    learner = Learner(c)
    n = 4
    x = inputs(c, n=n)
    human = torch.zeros(n, 30, 18, dtype=torch.bool)
    human[:, 1:, 6:12] = True
    valid = torch.as_tensor(phase_active(c, "p3")).expand(n, 30, 18).clone()
    b = dict(token=x["z"], next_token=x["z"], state=x["state"], next_state=x["state"], reference=x["reference"],
             next_reference=x["reference"], action=x["reference"][..., :18] + .1, valid=valid, human=human,
             bc_eligible=torch.ones(n, 30, 18, dtype=torch.bool), reward=torch.ones(n), discount=torch.full((n,), .9),
             phase=x["phase"], next_phase=x["phase"], vis=x["vis"], next_vis=x["vis"],
             scheduler_context=torch.zeros(n, c["scheduler"]["context_dim"]),
             next_scheduler_context=torch.zeros(n, c["scheduler"]["context_dim"]))
    proj = learner.model.actor.vis[1].weight.clone()
    for _ in range(3):
        m = learner.update(b)
    assert m["actor_updated"] and m["human_bc_loss"] > 0
    assert not torch.equal(proj, learner.model.actor.vis[1].weight)               # gradients reach the token projection
    with pytest.raises(ValueError, match="visual prefix tokens"):
        learner.update({k: v for k, v in b.items() if k not in ("vis", "next_vis")})


def _trajectory_with_observations(c, tmp_path):
    run = tmp_path / "run"
    (run / "observations").mkdir(parents=True)
    prefix = np.ones((968, 2048), np.float16)          # pools to editor_record's all-ones token
    mask = np.zeros(968, bool)
    mask[:512] = True
    db = Replay(run)
    eid = db.begin(dict(phase="p1", contract_sha256=c["contract_sha256"], token_sha256=representation_sha(c),
                        behavior_sha256="test"))
    for p in range(5):
        for k in (p, p + 1):
            np.savez(run / f"observations/o{k}.npz", prefix=prefix, mask=mask, contract_sha256=np.asarray(c["contract_sha256"]))
        row = editor_record(c)
        start, end = (p + 1) * 1_000_000_000, (p + 2) * 1_000_000_000
        row.update(phase=np.int64(p), next_phase=np.int64(min(p + 1, 4)), phase_reward_ns=np.int64(end - 1 if p < 4 else 0),
                   start_ns=np.int64(start), end_ns=np.int64(end),
                   observation_file=np.asarray(f"observations/o{p}.npz"), next_observation_file=np.asarray(f"observations/o{p+1}.npz"))
        row["valid"][:] = phase_active(c, f"p{p+1}")
        row["bc_eligible"][:] = row["valid"]
        db.append(eid, row)
        db.phase_success(eid, f"p{p+1}", end - 1)
    db.finish(eid, "success", 6_000_000_001)
    snapshot = run / "snapshots/p1-1.npz"
    snapshot.parent.mkdir()
    db.export(c, representation_sha(c), snapshot)
    db.close()
    return run, snapshot


def test_fit_heads_trains_resumes_and_serves_an_xattn_editor(tmp_path, monkeypatch):
    from vla_precision.gr00t_rlt import serving
    from vla_precision.gr00t_rlt.core import sha256
    torch.set_num_threads(1)
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    c["action_active"] = [True] * 18
    c["learner"]["batch_size"] = 5
    run, snapshot = _trajectory_with_observations(c, tmp_path)
    spec = importlib.util.spec_from_file_location("xattn_cli", Path(__file__).resolve().parents[1] / "scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "check_code", lambda c: None)
    monkeypatch.setattr(cli, "validate_assets", lambda c: None)
    args = SimpleNamespace(device="cpu", replay=snapshot, resume=None, updates=2, editor_arch="xattn")
    cli.fit_heads(c, args)
    out = Path(c["output"]) / "learner"
    first = json.loads((out / "latest.json").read_text())["checkpoint"]
    saved = torch.load(first, weights_only=True)
    assert saved["learner_options"]["editor_arch"] == "xattn" and saved["learner_options"]["editor_xattn"] == cli.XATTN_DEFAULTS
    assert len(list((Path(c["output"]) / "visual-tokens").glob("*.npy"))) == 6
    args.resume, args.editor_arch = Path(first), None
    cli.fit_heads(c, args)                                              # resumed: stays xattn
    second = Path(json.loads((out / "latest.json").read_text())["checkpoint"])
    assert torch.load(second, weights_only=True)["learner_options"]["editor_arch"] == "xattn"
    args.resume, args.editor_arch = second, "mlp"
    with pytest.raises(ValueError, match="xattn editor"):
        cli.fit_heads(c, args)
    # Serving builds the cross-attention network for this checkpoint and an MLP for the base.
    policy = serving.LivePolicy.__new__(serving.LivePolicy)
    policy.c, policy.device, policy.run, policy.token_sha = c, "cpu", run, representation_sha(c)
    (run / serving.SELECT_FILE).write_text(json.dumps(dict(select="checkpoint", checkpoint=str(second), sha256=sha256(second))))
    policy._select_actor()
    assert policy.actor_kind == "editor_rl" and policy.heads.arch == "xattn"
    (run / serving.SELECT_FILE).write_text(json.dumps(dict(select="base")))
    policy._select_actor()
    assert policy.actor_kind == "editor_zero" and policy.heads.arch == "mlp"


def test_fit_heads_refuses_tokens_from_the_wrong_observation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    c["action_active"] = [True] * 18
    c["learner"]["batch_size"] = 5
    run, snapshot = _trajectory_with_observations(c, tmp_path)
    mask = np.zeros(968, bool)
    mask[:512] = True
    np.savez(run / "observations/o2.npz", prefix=np.zeros((968, 2048), np.float16), mask=mask,
             contract_sha256=np.asarray(c["contract_sha256"]))
    spec = importlib.util.spec_from_file_location("xattn_cli2", Path(__file__).resolve().parents[1] / "scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "check_code", lambda c: None)
    monkeypatch.setattr(cli, "validate_assets", lambda c: None)
    with pytest.raises(ValueError, match="do not pool to the replay"):
        cli.fit_heads(c, SimpleNamespace(device="cpu", replay=snapshot, resume=None, updates=2, editor_arch="xattn"))
