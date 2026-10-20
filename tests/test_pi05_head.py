"""PI05 action-head BPTT: cached-prefix sampler parity and learner, CPU only."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

STARVLA = Path(__file__).resolve().parents[2] / "starVLA"
sys.path.insert(0, str(STARVLA))
pytest.importorskip("starVLA.model.modules.action_model.OpenPI_ActionHead")
from starVLA.model.modules.action_model.OpenPI_ActionHead import (GemmaDims, OpenPI05ActionHead, make_att_2d_masks,
                                                                   make_att_4d_masks)
try:
    from starVLA.model.framework.VLM4A.PI0 import PI0
    denoise_step = PI0.denoise_step
except Exception:            # the framework package needs `rich` (absent in isaac6): verbatim copy of PI0.denoise_step
    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, _adarms_cond = self.action_head.embed_action_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention_mask = make_att_4d_masks(torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2))
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return self.action_head(attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                                suffix_embs=suffix_embs, adarms_cond=_adarms_cond)
from transformers.cache_utils import DynamicCache

from vla_precision.gr00t_rlt.core import CONTINUOUS, load_config, phase_active, representation_sha
from vla_precision.gr00t_rlt.replay import Replay
from vla_precision.gr00t_rlt.pi05_head import (HeadLearner, collate_prefix, compact_prefix, denoise, hold_mask,
                                               load_prefix, save_prefix)

DIMS = GemmaDims(width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)


def tiny_head(horizon=30, action_dim=32):
    torch.manual_seed(0)
    head = OpenPI05ActionHead(DIMS, action_dim=action_dim, action_horizon=horizon, max_state_dim=6)
    for norm in [m for m in head.modules() if hasattr(m, "dense") and m.dense is not None]:
        torch.nn.init.normal_(norm.dense.weight, std=.05)            # adaRMS is zero-initialized
    return head.eval()


def production_sample(head, cache, pad, noise, steps):
    """PI0.sample_actions' Euler loop over PI0.denoise_step, verbatim."""
    fake = SimpleNamespace(action_head=head, vlm=SimpleNamespace(language_model=SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=head.action_in_proj))])))
    x_t, dt, time = noise.clone(), torch.tensor(-1.0 / steps), torch.tensor(1.0)
    while time >= -dt / 2:
        x_t = x_t + dt * denoise_step(fake, None, pad, cache, x_t, time.expand(len(noise)))
        time += dt
    return x_t


def random_prefix(length, masked):
    pad = torch.ones(1, length, dtype=torch.bool)
    pad[0, masked] = False
    cache = DynamicCache()
    for layer in range(DIMS.depth):
        cache.update(torch.randn(1, 1, length, DIMS.head_dim), torch.randn(1, 1, length, DIMS.head_dim), layer)
    return cache, pad


def test_cached_compact_prefix_sampler_matches_production_per_observation():
    head = tiny_head()
    torch.manual_seed(1)
    prefixes = [random_prefix(12, [3, 4, 5]), random_prefix(9, [8])]       # masked image tokens / padding
    noise = torch.randn(1, 30, 32)
    with torch.no_grad():
        expected = [production_sample(head, cache, pad, noise, 10) for cache, pad in prefixes]
        kv, valid = collate_prefix([compact_prefix(cache, pad) for cache, pad in prefixes])
        assert kv.shape[4] == 12 and valid.sum(1).tolist() == [9, 8] and not valid[0, 3:6].any()
        got = denoise(head, kv, valid, noise.expand(2, -1, -1), 10)
    for i in range(2):
        assert torch.allclose(got[i], expected[i][0], atol=1e-5)


def test_prefix_cache_file_roundtrip_is_bit_exact(tmp_path):
    kv = torch.randn(2, 3, 1, 7, 16).to(torch.bfloat16)
    mask = torch.tensor([True] * 3 + [False] * 2 + [True] * 4)
    save_prefix(tmp_path / "x.kv.npz", (kv, mask))
    got, got_mask = load_prefix(tmp_path / "x.kv.npz")
    assert torch.equal(got, kv) and torch.equal(got_mask, mask) and not list(tmp_path.glob("*.tmp*"))
    full, valid = collate_prefix([(kv, mask)])
    assert torch.equal(full[:, :, 0, :, mask], kv) and not full[:, :, 0, :, ~mask].any() and torch.equal(valid[0], mask)


def test_bptt_reaches_every_denoising_layer_and_checkpointing_gives_the_same_gradient():
    head = tiny_head().train()
    cache, pad = random_prefix(10, [2])
    kv, valid = collate_prefix([compact_prefix(cache, pad)])
    noise = torch.randn(1, 30, 32)
    grads = []
    for checkpoint in (False, True):
        head.zero_grad()
        denoise(head, kv, valid, noise, 10, checkpoint=checkpoint).square().mean().backward()
        grads.append({n: p.grad.clone() for n, p in head.named_parameters() if p.grad is not None})
    assert {"action_in_proj.weight", "action_out_proj.weight", "time_mlp_in.weight",
            "gemma_expert.model.layers.0.self_attn.q_proj.weight"} <= set(grads[0])
    assert all(grads[0][n].abs().sum() > 0 for n in ("action_in_proj.weight", "gemma_expert.model.layers.0.mlp.up_proj.weight"))
    assert all(torch.allclose(grads[0][n], grads[1][n], atol=1e-6) for n in grads[0])


def test_hold_mask_anchors_held_channels_and_joint_seeds():
    valid = torch.zeros(1, 30, 18, dtype=torch.bool)
    valid[..., 6:12] = True
    m = hold_mask(valid, 32)
    assert m.shape == (1, 30, 32) and not m[..., 6:12].any() and m[..., :6].all() and m[..., 18:].all()


def batch(c, n=4, phase=2):
    kvs = [compact_prefix(*random_prefix(8 + i, [1])) for i in range(n)]
    kv, valid = collate_prefix(kvs)
    active = torch.as_tensor(phase_active(c, f"p{phase+1}"))
    human = torch.zeros(n, 30, 18, dtype=torch.bool)
    human[0, 1:, 6:12] = True
    b = dict(token=torch.randn(n, c["token"]["embed_dim"]), state=torch.randn(n, c["state_dim"]),
             reference=torch.zeros(n, 30, 32), action=torch.full((n, 30, 18), .1),
             valid=active.expand(n, 30, 18).clone(), human=human, bc_eligible=torch.ones(n, 30, 18, dtype=torch.bool),
             reward=torch.ones(n), discount=torch.full((n,), .9), td_eligible=torch.ones(n, dtype=torch.bool),
             phase=torch.full((n,), phase), next_phase=torch.full((n,), phase),
             scheduler_context=torch.zeros(n, c["scheduler"]["context_dim"]),
             next_scheduler_context=torch.zeros(n, c["scheduler"]["context_dim"]),
             kv=kv, kv_valid=valid, next_kv=kv, next_kv_valid=valid)
    b.update(next_token=b["token"], next_state=b["state"], next_reference=b["reference"])
    return b


def test_head_learner_trains_the_head_through_the_sampler_and_round_trips(tmp_path):
    torch.set_num_threads(1)
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    c["learner"].update(head_lr=1e-3, critic_warmup_steps=0)
    learner = HeadLearner(c, tiny_head(), torch.randn(1, 30, 32), steps=3)
    assert learner.head.gemma_expert.lm_head is None
    before = learner.head.action_out_proj.weight.clone()
    b = batch(c)
    m1 = learner.update(b)
    assert not m1["actor_updated"] and np.isfinite(m1["critic_loss"])
    m2 = learner.update(b, term_stats=True)
    assert m2["actor_updated"] and m2["q_weight"] == c["learner"]["q_weight"]
    assert m2["human_bc_elements"] == 29 * 6 and m2["reference_elements"] == 4 * 30 * int(sum(phase_active(c, "p3"))) - 29 * 6
    assert {"hold_loss", "grad_norm_bc", "grad_norm_ref", "grad_norm_rl", "head_edit_mean_abs"} <= set(m2)
    assert not torch.equal(before, learner.head.action_out_proj.weight)
    restored = HeadLearner(c, tiny_head(), torch.randn(1, 30, 32), steps=3)
    restored.load_state_dict(learner.state_dict())
    assert restored.step == 2 and torch.equal(restored.head.action_out_proj.weight, learner.head.action_out_proj.weight)
    assert torch.equal(restored.target_head.action_out_proj.weight, learner.target_head.action_out_proj.weight)


def test_cli_encodes_nothing_but_trains_and_evaluates_a_head_end_to_end(tmp_path, capsys):
    import importlib.util, json
    from test_direct_editor import editor_record
    torch.set_num_threads(1)
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    c["action_active"] = [True] * 18
    c["action_half_range"] = [1.] * 18
    db = Replay(tmp_path / "replay")
    eid = db.begin(dict(phase="p1", contract_sha256=c["contract_sha256"], token_sha256=representation_sha(c), behavior_sha256="test"))
    for p in range(5):
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
    snapshot = tmp_path / "snap.npz"
    db.export(c, representation_sha(c), snapshot)
    db.close()
    cache = tmp_path / "cache"
    cache.mkdir()
    head = tiny_head()
    torch.save(dict(state=head.state_dict(), dims=vars(DIMS), action_dim=32, action_horizon=30, max_state_dim=6, steps=3,
                    noise=torch.randn(1, 30, 32), reference_seed=42, checkpoint="tiny", contract_sha256=c["contract_sha256"]),
               cache / "base_head.pt")
    for p in range(6):
        save_prefix(cache / f"o{p}.kv.npz", compact_prefix(*random_prefix(6 + p, [0])))
    spec = importlib.util.spec_from_file_location("pi05_head_rl", Path(__file__).resolve().parents[1] / "scripts/pi05_head_rl.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    a = SimpleNamespace(replay=snapshot, cache=cache, output=tmp_path / "run", resume=None, critic_init=None, updates=4,
                        batch_size=4, head_lr=1e-3, critic_lr=None, q_weight=None, reference_weight=None, keep_static_human=True,
                        human_batch_fraction=0., checkpoint_steps=False, jacobian_steps=2, micro_batch=3, term_stats_every=1, device="cpu",
                        run=tmp_path / "replay")
    cli.train(a, c)
    metrics = [json.loads(x) for x in (a.output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 4 and metrics[-1]["actor_updated"] and "grad_norm_rl" in metrics[-1]
    saved = torch.load(a.output / "head-step-00000004.pt", weights_only=True)
    assert saved["learner"]["step"] == 4 and saved["learner_options"]["head_lr"] == 1e-3
    serve = torch.load(a.output / "head-step-00000004.serve.pt", weights_only=True)
    assert all(v.dtype == torch.bfloat16 for v in serve["action_head"].values())
    with pytest.raises(ValueError, match="already has a head run"):
        cli.train(a, c)
    capsys.readouterr()
    cli.evaluate(SimpleNamespace(replay=snapshot, cache=cache, head=a.output / "head-step-00000004.pt", device="cpu", windows=5), c)
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["windows"] == {"p3_policy": 1, "p3_human": 0, "p1_policy": 1} and report["p3_right_edit_mm"] > 0


def test_jacobian_truncation_keeps_every_step_in_the_weight_gradient_but_identity_state_jacobian():
    """Same forward; every step builds weight-gradient graph; only the last K steps keep
    their state Jacobian (the noise gradient differs from exact BPTT unless K = all)."""
    head = tiny_head().train()
    kv, valid = collate_prefix([compact_prefix(*random_prefix(10, [2]))])
    noise = torch.randn(1, 30, 32)
    exact = denoise(head, kv, valid, noise, 10)
    cut = denoise(head, kv, valid, noise, 10, jacobian_steps=2)
    assert torch.allclose(exact, cut)
    calls = []
    hook = head.action_in_proj.register_forward_hook(lambda m, i, o: calls.append(o.requires_grad))
    denoise(head, kv, valid, noise, 10, jacobian_steps=3)
    hook.remove()
    assert calls == [True] * 10                                   # every step feeds the weight gradient
    noise.requires_grad_(True)
    g_exact = torch.autograd.grad(denoise(head, kv, valid, noise, 10).sum(), noise)[0]
    g_cut = torch.autograd.grad(denoise(head, kv, valid, noise, 10, jacobian_steps=3).sum(), noise)[0]
    g_none = torch.autograd.grad(denoise(head, kv, valid, noise, 10, jacobian_steps=10).sum(), noise)[0]
    assert torch.allclose(g_none, g_exact) and not torch.allclose(g_cut, g_exact)
    ones = torch.ones_like(noise)
    assert not torch.allclose(g_cut, ones)                         # last 3 steps keep their Jacobian
    head.zero_grad()
    cut.square().mean().backward()
    assert head.time_mlp_in.weight.grad.abs().sum() > 0


def test_micro_batched_actor_gradient_equals_the_full_batch_gradient(tmp_path):
    torch.set_num_threads(1)
    c = load_config(CONTINUOUS, "p1", workspace=tmp_path)
    c["learner"].update(head_lr=1e-3, critic_warmup_steps=0, actor_update_interval=1, max_grad_norm=1e9)
    b = batch(c, n=5)
    noise = torch.randn(1, 30, 32)
    grads = []
    for micro in (None, 2):
        torch.manual_seed(3)
        learner = HeadLearner(c, tiny_head(), noise, steps=3, micro_batch=micro)
        learner.ao.step = lambda: None                       # keep the weights; inspect .grad
        m = learner.update(b)
        grads.append(({n: p.grad.clone() for n, p in learner.head.named_parameters() if p.grad is not None}, m))
    (g1, m1), (g2, m2) = grads
    assert set(g1) == set(g2) and all(torch.allclose(g1[n], g2[n], atol=1e-6, rtol=1e-4) for n in g1)
    for k in ("human_bc_loss", "ref_loss", "hold_loss", "q_pi"):
        assert m1[k] == pytest.approx(m2[k], rel=1e-5, abs=1e-8)
