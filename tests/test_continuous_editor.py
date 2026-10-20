"""Shared actor/critic and full-task reward tests, CPU only in bwrap."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import (CONTINUOUS, EDITOR, load_config, phase_active,
    representation_sha, reward_and_discount, validate_replay)
from vla_precision.gr00t_rlt.model import Learner, make_heads
from vla_precision.gr00t_rlt.replay import Replay
from test_direct_editor import editor_record, bare_policy


def config(tmp_path):
    c = load_config(CONTINUOUS, workspace=tmp_path)
    c["action_active"] = [True]*18
    c["learner"]["batch_size"] = 5
    return c


def test_shared_contract_and_output_but_different_masks(tmp_path):
    recipes = [load_config(CONTINUOUS, f"p{i}", tmp_path) for i in range(1,6)]
    assert len({c["contract_sha256"] for c in recipes}) == 1
    assert len({c["output"] for c in recipes}) == 1
    assert recipes[0]["output"].endswith("/shared")
    assert recipes[0]["contract_sha256"] != load_config(EDITOR, workspace=tmp_path)["contract_sha256"]
    assert phase_active(recipes[0], "p1") != phase_active(recipes[0], "p2")


def test_reward_at_event_and_final_discount():
    r = config(Path("/tmp"))["reward"]
    # At unit discount/zero time penalty, the five event rewards are exactly
    # 1,1,1,1,11. Intermediate phases are NOT Bellman terminals.
    r = {**r, "gamma":1., "step_penalty":0.}
    assert reward_and_discount(1, 1_000_000_001, "continuing", r, phase_reward_ns=500_000_001) == (1.,1.)
    assert reward_and_discount(1, 1_000_000_001, "success", r) == (11.,0.)
    assert reward_and_discount(1, 1_000_000_001, "failure", r) == (0.,0.)
    with pytest.raises(ValueError, match="timestamp"):
        reward_and_discount(1, 1_000_000_001, "continuing", r, phase_reward_ns=2_000_000_000)


def test_phase_conditioned_actor_critics_and_batched_hand_masks(tmp_path):
    torch.set_num_threads(1)
    c = config(tmp_path)
    model = make_heads(c)
    z, state, ref = torch.zeros(5,16384), torch.zeros(5,59), torch.zeros(5,30,32)
    phase = torch.arange(5)
    assert model.actor[0].in_features == 16384+59+30*32+5
    assert model.critics[0].net[0].in_features == 16384+59+30*(32+36)+5+64
    assert torch.equal(model.act(z,state,ref,phase=phase), ref[...,:18])
    with torch.no_grad():
        model.actor[-1].bias.fill_(1.)
    active = model.active_for(phase).expand(5,30,18)
    edited = model.act(z,state,ref,phase=phase)
    assert (edited[~active] == 0).all() and (edited[active] > 0).all()
    for i in range(5):
        assert model.active_for(phase)[i,0].tolist() == phase_active(c, f"p{i+1}")
    smooth = model.smooth_action(edited, ref, torch.ones_like(edited), phase=phase)
    assert (smooth[~active] == 0).all() and smooth.max() <= .2
    with pytest.raises(ValueError, match="phase labels"):
        model.act(z,state,ref)


def add_trajectory(db, c, outcome="success", start_phase=0):
    meta = dict(phase=f"p{start_phase+1}", contract_sha256=c["contract_sha256"], token_sha256=representation_sha(c), behavior_sha256="test")
    eid = db.begin(meta)
    for p in range(start_phase, 5):
        row = editor_record(c)
        start, end = (p+1)*1_000_000_000, (p+2)*1_000_000_000
        row.update(phase=np.int64(p), next_phase=np.int64(min(p+1,4)),
                   phase_reward_ns=np.int64(end-1 if p<4 else 0), start_ns=np.int64(start), end_ns=np.int64(end))
        row["valid"][:] = phase_active(c, f"p{p+1}")
        row["bc_eligible"][:] = row["valid"]
        db.append(eid,row)
        if p < 4 or outcome == "success":
            db.phase_success(eid, f"p{p+1}", end-1)
    db.finish(eid,outcome,6_000_000_001)
    return eid


@pytest.mark.parametrize("start", range(5))
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_suffix_replay_preserves_start_and_rewards(tmp_path, start, outcome):
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db, c, outcome=outcome, start_phase=start)
    path = tmp_path/"suffix.npz"
    db.export(c, representation_sha(c), path)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["episode_start_phases"] == {eid:f"p{start+1}"}
    assert meta["fulltask_success_episodes"] == ([eid] if start == 0 and outcome == "success" else [])
    with np.load(path) as f:
        data = {k:f[k].copy() for k in f.files}
    assert data["phase"].tolist() == list(range(start, 5))
    validate_replay(data, meta, c, token_sha256=representation_sha(c))
    db.close()


def test_export_can_keep_only_the_episodes_one_actor_collected(tmp_path):
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    mine, other = add_trajectory(db, c), add_trajectory(db, c)
    with db.db:
        db.db.execute("UPDATE episodes SET behavior='step600' WHERE id=?", (mine,))
    db.export(c, representation_sha(c), tmp_path/"all.npz")
    assert json.loads((tmp_path/"all.json").read_text())["episodes"] == [mine, other]
    db.export(c, representation_sha(c), tmp_path/"mine.npz", behaviors=["step600"])
    meta = json.loads((tmp_path/"mine.json").read_text())
    assert meta["episodes"] == [mine] and meta["behavior_filter"] == ["step600"]
    with pytest.raises(ValueError, match="No complete"):
        db.export(c, representation_sha(c), tmp_path/"none.npz", behaviors=["nobody"])
    db.close()


def test_suffix_replay_rejects_wrong_start_or_missing_milestone(tmp_path):
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db, c, start_phase=2)
    with db.db:
        db.db.execute("UPDATE episodes SET phase='p1' WHERE id=?", (eid,))
    with pytest.raises(ValueError, match="start phase"):
        db.export(c, representation_sha(c), tmp_path/"bad.npz")
    with db.db:
        db.db.execute("UPDATE episodes SET phase='p3' WHERE id=?", (eid,))
        db.db.execute("DELETE FROM phase_events WHERE episode=? AND phase='p3'", (eid,))
    with pytest.raises(ValueError, match="phase reward evidence"):
        db.export(c, representation_sha(c), tmp_path/"bad.npz")
    db.close()


def test_shared_replay_train_resume_and_discard(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db,c)
    snapshot = tmp_path/"shared.npz"
    db.export(c,representation_sha(c),snapshot)
    with np.load(snapshot) as f:
        data = {k:f[k].copy() for k in f.files}
    meta = json.loads(snapshot.with_suffix(".json").read_text())
    reward, discount = validate_replay(data,meta,c,token_sha256=representation_sha(c))
    assert (discount[:4]>0).all() and discount[-1]==0
    spec = importlib.util.spec_from_file_location("shared_train_cli",Path(__file__).resolve().parents[1]/"scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,"check_code",lambda c:None)
    monkeypatch.setattr(cli,"validate_assets",lambda c:None)
    args = SimpleNamespace(device="cpu",replay=snapshot,resume=None,updates=2)
    cli.fit_heads(c,args)
    out = Path(c["output"])/"learner"
    metrics = [json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines()]
    assert sum(metrics[-1]["phase_samples"].values()) == 5
    assert all(k in metrics[-1] for k in ("critic_loss","bc_loss","ref_loss","rl_loss"))
    assert all(k in metrics[-1] for k in ("critic_grad_norm","critic_grad_clipped","actor_grad_norm","actor_grad_clipped",
                                          "grad_norm_bc","grad_norm_ref","grad_norm_rl","grad_cos_bc_ref","grad_cos_bc_rl"))
    first = json.loads((out/"latest.json").read_text())
    args.resume = Path(first["checkpoint"])
    cli.fit_heads(c,args)
    saved = torch.load(json.loads((out/"latest.json").read_text())["checkpoint"],weights_only=True)
    assert saved["learner"]["step"] == 4 and saved["learner"]["critic_updates"] == 4
    assert "anchor" not in saved["learner"] and "reference_anchor" not in saved["learner_options"]
    # Anchored round: the reference loss targets the step-2 editor; a resumed run keeps the anchor.
    args.resume, args.reference_anchor = Path(json.loads((out/"latest.json").read_text())["checkpoint"]), Path(first["checkpoint"])
    cli.fit_heads(c,args)
    saved = torch.load(json.loads((out/"latest.json").read_text())["checkpoint"],weights_only=True)
    assert saved["learner_options"]["reference_anchor"]["step"] == 2 and "anchor" in saved["learner"]
    assert "anchor_gap_mean_abs" in json.loads((out/"metrics.jsonl").read_text().splitlines()[-1])
    args.resume, args.reference_anchor = Path(json.loads((out/"latest.json").read_text())["checkpoint"]), None
    cli.fit_heads(c,args)
    again = torch.load(json.loads((out/"latest.json").read_text())["checkpoint"],weights_only=True)
    assert again["learner"]["step"] == 8 and again["learner_options"]["reference_anchor"] == saved["learner_options"]["reference_anchor"]
    db.discard(eid)
    with pytest.raises(ValueError,match="No complete"):
        db.export(c,representation_sha(c),tmp_path/"discard.npz")
    db.close()


def test_serving_keeps_shared_weights_across_phase_and_pause(tmp_path, monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"VLAPolicyBridge"))
    from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS
    c = config(tmp_path)
    p = bare_policy(c)
    p._load_actor()
    p.episode, p.capture_root = None, None
    p.stats = dict(q01=[-1]*32,q99=[1]*32)
    ref = np.zeros((30,32),np.float32)
    mask = np.zeros(800,bool)
    mask[:512] = True
    feature = dict(prefix=np.ones((800,2048),np.float16),mask=mask,state=np.zeros(59,np.float32),reference=ref)
    reset, loaded = [], []
    p.features = SimpleNamespace(extract=lambda s:feature,reset=lambda:reset.append(1),
        physical_actions=lambda a,r:np.concatenate([a,r[:,18:]],axis=-1))
    monkeypatch.setattr(p,"_load_actor",lambda *rollout:loaded.append(1))
    obs = synthetic_observation()
    with pytest.raises(ValueError, match="rollout identity"):
        p.get_action(obs, {"episode":"missing-stable-trajectory-id"})
    for i in range(5):
        obs["language"]["annotation.human.task_description"] = [[INSTRUCTIONS[i]]]
        result = p.get_action(obs, {"episode":f"request-{i}","rollout":"same-physical-trial"})
        assert result["info"]["rlt"]["phase"] == f"p{i+1}"
        assert result["info"]["rlt"]["active"] == phase_active(c,f"p{i+1}")
    assert loaded == reset == [1]
    p.get_action(obs,{"episode":"resume-request","rollout":"same-physical-trial"})
    assert loaded == [1]
    p.get_action(obs,{"episode":"new-request","rollout":"new-trial"})
    assert loaded == [1,1]


def test_replay_requires_all_phase_events_and_new_contract(tmp_path):
    c=config(tmp_path)
    db=Replay(tmp_path/"replay")
    eid=add_trajectory(db,c)
    with db.db:
        db.db.execute("DELETE FROM phase_events WHERE episode=? AND phase='p5'",(eid,))
    with pytest.raises(ValueError,match="P5 milestone"):
        db.export(c,representation_sha(c),tmp_path/"bad.npz")
    old=load_config(EDITOR,workspace=tmp_path)
    with pytest.raises(ValueError,match="No complete"):
        db.export(old,representation_sha(old),tmp_path/"old-contract.npz")
    db.close()


def test_half_speed_recipe_is_its_own_contract(tmp_path):
    half = Path(str(CONTINUOUS).replace("continuous.yaml", "continuous_30hz.yaml"))
    c30, c60 = load_config(half, workspace=tmp_path), load_config(CONTINUOUS, workspace=tmp_path)
    assert c30["scheduler"]["action_hz"] == 30 and c60["scheduler"]["action_hz"] == 60
    assert c30["contract_sha256"] != c60["contract_sha256"]
    assert c30["output"] != c60["output"]
    assert c30["reward"] == c60["reward"]   # elapsed-time discount clock is rate-independent


def test_unsupported_playback_rate_is_rejected(tmp_path):
    import yaml
    raw = yaml.safe_load(Path(CONTINUOUS).read_text())
    raw["scheduler"]["action_hz"] = 45
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="action_hz"):
        load_config(path, workspace=tmp_path)


def test_human_bc_drops_held_takeover_rows_and_keeps_moving_ones():
    import numpy as np
    from vla_precision.gr00t_rlt.core import drop_static_human, STATIC_HUMAN
    half = np.ones(18, np.float32) * .01          # 1 normalized unit = 10 mm / 0.01 rad
    half[12:] = 500.
    n, hz = 2, 30.
    action = np.zeros((n, 30, 18), np.float32)
    human = np.zeros((n, 30, 18), bool)
    # window 0: right arm held for rows 0-9, then moves 30 mm/s (0.1 normalized per row) for rows 10-19
    human[0, :20, 6:12] = True
    action[0, 10:20, 6] = .1 * np.arange(1, 11)
    # left arm creeps 1 mm/s (below 2 mm/s) on rows 0-19: still
    human[0, :20, 0:6] = True
    action[0, :20, 0] = (.001 / hz / .01) * np.arange(20)
    # window 1: a held hand channel, a toggling one, and an isolated labelled row
    human[1, :10, 13:15] = True
    action[1, 5:10, 14] = np.linspace(.2, 1., 5)
    human[1, 20, 6:12] = True
    data = dict(action=action, human=human.copy(), bc_eligible=human.copy(), valid=np.ones_like(human))
    counts = drop_static_human(data, half, hz, **STATIC_HUMAN)
    kept = data["human"]
    assert (data["bc_eligible"] == kept).all()            # dropped rows get no reference loss either
    assert not kept[0, :9, 6:12].any() and kept[0, 10:20, 6:12].all()
    assert kept[0, 9, 6:12].all()                          # the row a move starts from is moving
    assert not kept[0, :, 0:6].any()
    assert not kept[1, :, 13].any() and kept[1, 4:10, 14].all() and not kept[1, :4, 14].any()
    assert not kept[1, 20].any()
    assert counts["human_bc_elements"] == int(human.sum()) and counts["kept"] == int(kept.sum())


def test_actor_term_gradients_sum_to_the_backward_gradient():
    from vla_precision.gr00t_rlt.model import term_gradients, gradient_stats
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    x = torch.randn(16, 4)
    def objective():
        y = net(x)
        return dict(bc=(y - 1).square().mean(), ref=y.square().mean(), rl=-0.0 * y.sum())
    grads = term_gradients(net.parameters(), objective())
    net.zero_grad()
    sum(objective().values()).backward()
    for i, p in enumerate(net.parameters()):
        assert torch.allclose(sum(g[i] for g in grads.values()), p.grad, atol=1e-6)
    s = gradient_stats(grads)
    assert s["grad_norm_rl"] == 0 and s["grad_cos_bc_rl"] == 0
    assert s["grad_norm_bc"] > 0 and -1 <= s["grad_cos_bc_ref"] <= 1


def test_reference_anchor_replaces_the_frozen_reference_as_the_ref_loss_target(tmp_path):
    c = config(tmp_path)
    c["learner"].update(q_weight=0., actor_update_interval=1, actor_lr=0.)
    phase = torch.full((4,), 2)
    b = dict(token=torch.zeros(4, c["token"]["embed_dim"]), state=torch.zeros(4, c["state_dim"]),
             reference=torch.zeros(4, 30, 32), action=torch.zeros(4, 30, 18),
             valid=torch.ones(4, 30, 18, dtype=torch.bool), human=torch.zeros(4, 30, 18, dtype=torch.bool),
             bc_eligible=torch.ones(4, 30, 18, dtype=torch.bool), reward=torch.zeros(4), discount=torch.zeros(4),
             phase=phase, next_phase=phase)
    b.update(next_token=b["token"], next_state=b["state"], next_reference=b["reference"])
    if "context_dim" in c.get("scheduler", {}):
        b.update(scheduler_context=torch.zeros(4, c["scheduler"]["context_dim"]), next_scheduler_context=torch.zeros(4, c["scheduler"]["context_dim"]))
    plain = Learner(c)
    assert plain.update(b)["ref_loss"] == 0                     # zero editor == pi0.5 reference
    shifted = Learner(c)
    other = make_heads(c)
    torch.nn.init.constant_(other.actor[-1].bias, 1.)
    shifted.set_anchor(other.state_dict())
    m = shifted.update(b)
    act = torch.as_tensor(phase_active(c, "p3"))
    gap = float(other.scale * np.tanh(1.))
    assert m["ref_loss"] == pytest.approx(gap**2 * act.float().mean().item(), rel=1e-5)   # anchor edits only active channels
    assert m["anchor_gap_mean_abs"] == pytest.approx(gap * act.float().mean().item(), rel=1e-5)
    restored = Learner(c)
    restored.load_state_dict(shifted.state_dict())
    assert torch.equal(restored.anchor.actor[-1].bias, other.actor[-1].bias)


def test_human_batch_quota_draws_half_from_human_windows_and_zero_is_the_old_sampler():
    spec = importlib.util.spec_from_file_location("quota_cli", Path(__file__).resolve().parents[1]/"scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    phase = np.repeat(np.arange(5), 200)
    phase_rows = [np.flatnonzero(phase == p) for p in range(5)]
    has_human = np.zeros(1000, bool)
    has_human[[3, 250, 251, 610]] = True        # P1, P2, P2, P4 only
    human_rows = [r[has_human[r]] for r in phase_rows]
    human_rows = [r for r in human_rows if len(r)]
    ids = cli.sample_batch(np.random.default_rng(0), 1000, 128, phase_rows, human_rows, 64)
    assert len(ids) == 128 and has_human[ids[64:]].all()
    assert {int(phase[i]) for i in ids[64:]} == {0, 1, 3}   # stratified over phases that have human windows
    # quota 0: identical draws to the pre-quota sampler
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    rng_b.integers(1000, size=128)
    old = np.asarray([rng_b.choice(phase_rows[i]) for i in rng_b.integers(5, size=128)])
    assert (cli.sample_batch(rng_a, 1000, 128, phase_rows, human_rows, 0) == old).all()


def test_linear_editor_is_unbounded_but_clamped_to_the_action_range_and_tanh_is_unchanged(tmp_path):
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    heads = make_heads(c)
    torch.nn.init.constant_(heads.actor[-1].bias, 10.)          # pre-activation 10 everywhere
    z = torch.zeros(2, c["token"]["embed_dim"]); s = torch.zeros(2, c["state_dim"]); ref = torch.zeros(2, 30, 32)
    phase = torch.tensor([2, 2])
    act = phase_active(c, "p3")
    tanh = heads.act(z, s, ref, phase=phase)
    assert torch.allclose(tanh[0, 0, torch.as_tensor(act)], torch.full((int(sum(act)),), heads.scale * np.tanh(10.), dtype=torch.float32))
    heads.activation = "linear"
    lin = heads.act(z, s, ref, phase=phase)
    assert torch.allclose(lin[0, 0, torch.as_tensor(act)], torch.full((int(sum(act)),), 2.0))   # 0.2 * 10, unbounded
    torch.nn.init.constant_(heads.actor[-1].bias, 100.)
    assert heads.act(z, s, ref, phase=phase).max() == 2.2                                       # action-range clamp
    noisy = heads.smooth_action(torch.full((2, 30, 18), 1.), ref, torch.zeros(2, 30, 18), phase=phase)
    assert noisy[0, 0, torch.as_tensor(act)].eq(1.).all()                                       # no +/-scale clamp when linear
    c2 = dict(c, learner=dict(c["learner"], editor_activation="linear"))
    assert make_heads(c2).activation == "linear"
    with pytest.raises(ValueError, match="editor_activation"):
        make_heads(dict(c, learner=dict(c["learner"], editor_activation="relu")))
