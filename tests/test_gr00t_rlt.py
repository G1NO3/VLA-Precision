"""GR00T reward parity, bimanual action masks, replay boundaries and AC updates."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import LEGACY, load_config as _load_config, reward_and_discount, validate_replay
from vla_precision.gr00t_rlt.model import Heads, Learner, supervised_loss, token_model


def load_config(**kw):
    """Keep the original five-checkpoint regression fixtures explicit."""
    return _load_config(LEGACY, **kw)


@pytest.mark.parametrize("phase", ["p1", "p2", "p3", "p4", "p5"])
def test_phase_recipe(phase):
    c = load_config(phase=phase)
    assert c["reward"]["failure_reward"] == 0
    assert c["reward"]["gamma"] == pytest.approx(.999 ** .5)
    assert c["baseline"]["left_thumb"] == ("absolute" if phase == "p1" else "chunk_relative")
    assert c["actor_action_dim"] == 18 and not c["live_enabled"]


@pytest.mark.parametrize("outcome", ["continuing", "success", "failure"])
@pytest.mark.parametrize("ticks", [1, 6, 30, 90, 600])
def test_jm_reward_formula(outcome, ticks):
    c = load_config()
    reward = c["reward"]
    actual, discount = reward_and_discount(1_000_000_000, 1_000_000_000 + round(ticks / 60 * 1e9), outcome, reward)
    values = np.full(ticks, -.005)
    if outcome != "continuing":
        values[-1] = 10 if outcome == "success" else 0
    assert actual == pytest.approx(sum(v * reward["gamma"] ** i for i, v in enumerate(values)))
    assert discount == pytest.approx(reward["gamma"] ** ticks if outcome == "continuing" else 0)


def test_real_jm_configuration_and_function_if_available():
    base = Path("/home/jwang3617/jm_pipette")
    if not base.exists():
        pytest.skip("Optional read-only comparison with user's jm_pipette workspace")
    cfg = json.loads((base / "runs/qwenoft_ac/config.json").read_text())
    c = load_config()
    assert c["reward"]["failure_reward"] == cfg["failure_reward"]
    assert c["reward"]["rate_hz"] == cfg["rate_hz"]
    path = base / "VLAPolicyBridge/vla_policy_bridge/hil/ac_rewards.py"
    spec = importlib.util.spec_from_file_location("jm_rewards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for outcome in ("continuing", "success", "failure"):
        got = reward_and_discount(0, 1_500_000_000, outcome, c["reward"])
        expected = mod.reward_and_discount(90, outcome != "continuing", outcome == "success", .999**.5,
                                          -.005, failure_reward=cfg["failure_reward"], success_reward=10)
        np.testing.assert_allclose(got, expected)


def test_label_wait_counts_and_discard_not_a_failure():
    r = load_config()["reward"]
    assert reward_and_discount(0, 100_000_000, "success", r, terminal_ns=1_000_000_000) == reward_and_discount(0, 1_000_000_000, "success", r)
    with pytest.raises(ValueError):
        reward_and_discount(0, 1_000_000_000, "discarded", r)
    with pytest.raises(ValueError):
        reward_and_discount(2, 1, "continuing", r)


def replay_fixture():
    c = load_config()
    data = {k: np.zeros((2, 59), np.float32) for k in ("state", "next_state")}
    data.update({k: np.zeros((2, 256), np.float32) for k in ("token", "next_token")})
    data.update({k: np.zeros((2, 30, 32), np.float32) for k in ("reference", "next_reference")})
    data.update(action=np.zeros((2, 30, 18), np.float32), valid=np.ones((2, 30, 18), bool),
                human=np.zeros((2, 30, 18), bool), bc_eligible=np.ones((2, 30, 18), bool),
                start_ns=np.array([0, 500_000_000]), end_ns=np.array([500_000_000, 1_000_000_000]),
                terminal_ns=np.array([0, 1_500_000_000]), outcome=np.array(["continuing", "success"]),
                episode_outcome=np.array(["success", "success"]))
    meta = dict(schema="gr00t_rlt_replay.v1", phase="p1", contract_sha256=c["contract_sha256"],
                execution_contract=c["execution_contract"], token_sha256="token", completed_episodes_only=True)
    return c, data, meta


def test_replay_elapsed_terminal_and_hashes():
    c, data, meta = replay_fixture()
    rewards, discounts = validate_replay(data, meta, c, token_sha256="token")
    assert discounts[0] == pytest.approx(c["reward"]["gamma"] ** 30)
    assert discounts[1] == 0 and rewards[1] > 9
    for key, value in (("token_sha256", "other"), ("phase", "p2"), ("execution_contract", "bc_async"), ("synthetic", True)):
        with pytest.raises(ValueError):
            validate_replay(data, {**meta, key: value}, c, token_sha256="token")


def test_replay_rejects_legacy_xyz_and_invalid_masks():
    c, data, meta = replay_fixture()
    with pytest.raises(ValueError):
        validate_replay({**data, "action": np.zeros((2, 30, 3))}, meta, c, token_sha256="token")
    data["valid"][0, 3, 1] = False
    with pytest.raises(ValueError):
        validate_replay(data, meta, c, token_sha256="token")


def test_per_channel_human_and_wait_bc():
    pred = torch.zeros(1, 30, 18, requires_grad=True)
    human = torch.zeros_like(pred, dtype=torch.bool)
    human[..., :6] = True
    valid = torch.ones_like(human)
    eligible = valid.clone()
    eligible[:, 4, :6] = False
    loss = supervised_loss(pred, torch.ones_like(pred), torch.full((1, 30, 32), 2.),
                           human, eligible, valid, load_config()["learner"])
    loss.backward()
    assert pred.grad[:, 4, :6].abs().sum() == 0
    assert pred.grad[:, 4, 6:].abs().sum() > 0
    assert pred.grad[:, 5, :6].abs().sum() > 0


def test_critic_masks_zero_padding_and_sees_joint_seeds():
    torch.manual_seed(3)
    c = load_config()
    model = Heads(c)
    z, s, ref = torch.zeros(1, 256), torch.zeros(1, 59), torch.zeros(1, 30, 32)
    a = torch.zeros(1, 30, 18)
    valid = torch.ones_like(a, dtype=torch.bool)
    valid[:, -2:] = False
    q = model.q(z, s, ref, a, valid)
    altered = a.clone()
    altered[:, -2:] = 100
    torch.testing.assert_close(q, model.q(z, s, ref, altered, valid))
    ref[..., 18:] = 1
    assert not torch.allclose(q, model.q(z, s, ref, a, valid))


def test_real_update_and_optimizer_roundtrip(tmp_path):
    torch.set_num_threads(2)
    c, data, meta = replay_fixture()
    rewards, discounts = validate_replay(data, meta, c, token_sha256="token")
    keys = ("token", "next_token", "state", "next_state", "reference", "next_reference", "action", "valid", "human", "bc_eligible")
    b = {k: torch.from_numpy(data[k]) for k in keys}
    b.update(reward=torch.from_numpy(rewards), discount=torch.from_numpy(discounts))
    learner = Learner(c)
    original = copy.deepcopy(learner.model.state_dict())
    learner.update(b)
    metric = learner.update(b)
    assert np.isfinite(list(metric.values())).all() and "actor_loss" in metric
    assert any(not torch.equal(original[k], v) for k, v in learner.model.state_dict().items())
    path = tmp_path / "heads.pt"
    torch.save(learner.state_dict(), path)
    recovered = Learner(c)
    recovered.load_state_dict(torch.load(path, weights_only=True))
    assert recovered.step == 2
    assert len(recovered.ao.state) > 0
    for k, v in recovered.model.state_dict().items():
        torch.testing.assert_close(v, learner.model.state_dict()[k])


def test_upstream_token_gradient_and_variable_prefix():
    c = load_config()
    c["token"].update(input_dim=16, embed_dim=8, prefix_seq_len=12, num_heads=2, num_layers=1)
    model = token_model(c)
    loss, _ = model.loss(torch.randn(2, 7, 16), torch.ones(2, 7, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.rl_token_embed.grad.abs().sum() > 0


def test_constant_channels_preserve_reference_and_reject_q_labels():
    c, data, meta = replay_fixture()
    c["action_active"] = [True] * 18
    c["action_active"][13] = False
    model = Heads(c)
    ref = torch.randn(1, 30, 32)
    action = model.act(torch.zeros(1, 256), torch.zeros(1, 59), ref, dropout=1.)
    torch.testing.assert_close(action[..., 13], ref[..., 13])
    with pytest.raises(ValueError, match="Zero-span"):
        validate_replay(data, meta, c, token_sha256="token")


def test_offline_cli_token_bc_rl_and_resume(tmp_path, monkeypatch):
    """Exercise real training commands on tiny fixtures, never real collection."""
    from types import SimpleNamespace
    from vla_precision.gr00t_rlt.core import sha256
    path = Path(__file__).parents[1] / "scripts/gr00t_rlt.py"
    spec = importlib.util.spec_from_file_location("gr00t_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "validate_assets", lambda c: {})
    monkeypatch.setattr(cli, "check_code", lambda c: None)
    c, data, meta = replay_fixture()
    c["output"] = str(tmp_path)
    c["token"].update(input_dim=16, embed_dim=8, prefix_seq_len=12, num_heads=2, num_layers=1, batch_size=2)
    c["learner"]["batch_size"] = 2
    root = tmp_path / "features"
    root.mkdir()
    records = []
    rng = np.random.default_rng(42)
    for i, split in enumerate(("train", "train", "validation")):
        name = f"{i}.npz"
        np.savez(root / name, prefix=rng.normal(size=(7, 16)).astype(np.float32), mask=np.ones(7, bool),
                 state=np.zeros(59, np.float32), reference=np.zeros((30, 32), np.float32),
                 demo_action=np.zeros((30, 32), np.float32))
        records.append(dict(file=name, split=split, episode=i, sha256=sha256(root / name)))
    cli.write_json(root / "manifest.json", dict(records=records, assets={}, contract_sha256=c["contract_sha256"]))
    args = SimpleNamespace(device="cpu", updates=2, replay=tmp_path / "replay.npz", resume=None)
    cli.fit_token(c, args)
    cli.fit_bc(c, args)
    meta["token_sha256"] = sha256(tmp_path / "token/best.pt")
    data["token"] = data["token"][:, :8]
    data["next_token"] = data["next_token"][:, :8]
    np.savez(args.replay, **data)
    cli.write_json(args.replay.with_suffix(".json"), meta)
    cli.fit_heads(c, args)
    args.resume = tmp_path / "learner/step-00000002.pt"
    cli.fit_heads(c, args)
    final = torch.load(tmp_path / "learner/step-00000004.pt", weights_only=True)
    assert final["learner"]["step"] == 4
    assert not final["live_enabled"]
