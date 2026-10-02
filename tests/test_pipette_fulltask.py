"""Data geometry and JAX input contracts; no robot or full model allocation."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla_precision.data.pipette_fulltask import (
    FORMAT, TASKS, FullTaskDataset, action_chunk, moving_anchors, validate_recording_split,
)


def commands(n=40):
    cmd = np.zeros((n, 32), np.float32)
    cmd[:, 0] = np.arange(n) * .001
    cmd[:, 6] = .2
    cmd[:, 12:18] = 800 - np.arange(n)[:, None]
    cmd[:, 18:] = .3
    return cmd


@pytest.mark.parametrize("relative", [False, True])
def test_label_semantics(relative):
    cmd = commands()
    a = action_chunk(cmd, 5, left_thumb_relative=relative)
    np.testing.assert_allclose(a[0, :12], 0, atol=1e-7)
    np.testing.assert_allclose(a[1, 0], .001, atol=1e-7)
    np.testing.assert_allclose(a[:, 13:], cmd[5:35, 13:])
    np.testing.assert_allclose(a[:, 12], cmd[5:35, 12] - (cmd[5, 12] if relative else 0))


def test_rotation_is_body_relative_not_subtraction():
    cmd = commands()
    r0 = Rotation.from_euler("x", 1.1)
    delta = Rotation.from_euler("z", .6)
    cmd[0, 3:6] = r0.as_rotvec()
    cmd[1, 3:6] = (r0 * delta).as_rotvec()
    a = action_chunk(cmd, 0)
    np.testing.assert_allclose(a[1, 3:6], delta.as_rotvec(), atol=1e-7)
    assert not np.allclose(a[1, 3:6], cmd[1, 3:6] - cmd[0, 3:6])


def test_tail_repeats_absolute_then_relativizes():
    cmd = commands(4)
    a = action_chunk(cmd, 2)
    np.testing.assert_array_equal(a[1:], np.broadcast_to(a[1], a[1:].shape))
    np.testing.assert_allclose(a[1, 0], .001, atol=1e-7)


def test_pause_filter_matches_six_dim_end_to_start():
    cmd = np.zeros((30, 32), np.float32)
    cmd[-1, [0, 6]] = .0008
    assert 0 in moving_anchors(cmd)  # combined norm 1.13 mm, both below 1 mm
    cmd[-1] = 0
    cmd[10, 0] = .01
    assert 0 not in moving_anchors(cmd)  # return-to-start also filtered


def test_recording_leakage_rejected():
    rows = [dict(episode_index=10, source_session="s", source_episode="e_pick"),
            dict(episode_index=1, source_session="s", source_episode="e_return")]
    with pytest.raises(ValueError, match="leakage"):
        validate_recording_split(rows, {10: 1, 1: 5})
    rows[1]["episode_index"] = 11
    validate_recording_split(rows, {10: 1, 11: 5})


def test_prepared_dataset(tmp_path):
    root = tmp_path / "p2"
    root.mkdir()
    records = []
    for ep, split in ((0, "train"), (1, "validation")):
        dest = tmp_path / "shared" / f"episode_{ep:06d}"
        dest.mkdir(parents=True)
        np.savez(dest / "numeric.npz", state=np.zeros((40, 59), np.float32),
                 commands=commands(), train_anchors=moving_anchors(commands()))
        records.append(dict(episode=ep, split=split, frames=40))
    (root / "manifest.json").write_text(json.dumps(dict(format=FORMAT, phase=2,
            source=str(tmp_path), shared_root="../shared", episodes=records)))
    train, val = FullTaskDataset(root), FullTaskDataset(root, split="validation")
    assert set(train.episodes).isdisjoint(val.episodes)
    assert val.numeric_item(0)["actions"][0, 12] == 0
    assert val.validation_indices() == val.validation_indices()
    assert len(val.validation_indices()) == 480


def test_two_cameras_59_state_32_actions():
    from vla_precision.integrations.openpi.policies.pipette_fulltask import FullTaskInputs, FullTaskOutputs
    sample = dict(state=np.arange(59, dtype=np.float32), prompt=TASKS[1], actions=commands(30),
                  base_0_rgb=np.full((270, 270, 3), 123, np.uint8),
                  left_wrist_0_rgb=np.full((270, 270, 3), 234, np.uint8))
    out = FullTaskInputs()(sample)
    assert out["state"].shape == (59,)
    assert out["image_mask"]["right_wrist_0_rgb"] == False
    assert out["image"]["right_wrist_0_rgb"].sum() == 0
    assert out["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert FullTaskOutputs()(out)["actions"].shape == (30, 32)
    np.testing.assert_array_equal(sample["state"], out["state"])


def test_state_dropout_only_train_and_whole_vector():
    from vla_precision.integrations.openpi.policies.pipette_fulltask import FullTaskStateDropout
    sample = {"state": np.ones(59, np.float32)}
    drop = FullTaskStateDropout(probability=1)
    np.testing.assert_array_equal(drop(sample)["state"], 1)
    np.testing.assert_array_equal(drop.with_training(True)(sample)["state"], 0)
    np.testing.assert_array_equal(sample["state"], 1)


def test_native_jax_adapter_retains_state_and_weights_dimensions():
    from vla_precision.integrations.openpi.configs import get_config
    from openpi import transforms
    cfg = get_config("pi05_full_finetune_pipette_fulltask")
    assert cfg.model.pi05 and cfg.model.discrete_state_input
    assert (cfg.model.action_dim, cfg.model.action_horizon, cfg.model.max_token_len) == (32, 30, 320)
    out = transforms.PadStatesAndActions(32)({"state": np.zeros(59), "actions": np.zeros((30, 32))})
    assert out["state"].shape == (59,)


def test_physical_metrics_hold_baseline():
    from vla_precision.integrations.openpi.fulltask_eval import physical_metrics
    target = np.zeros((2, 30, 32))
    pred = target.copy()
    pred[..., :3] = .001
    out = physical_metrics(pred, target, target)
    assert out["left_xyz_rmse_mm"] == pytest.approx(1)
    assert out["left_xyz_rmse_mm_hold"] == 0
    assert out["left_rotation_rms_deg"] == 0


def test_best_checkpoint_native_orbax_roundtrip(tmp_path):
    from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat
    install_lerobot_import_compat()
    import jax.numpy as jnp
    from flax import nnx
    import orbax.checkpoint as ocp
    from openpi.training import checkpoints
    from openpi.models.model import restore_params
    class Toy(nnx.Module):
        def __init__(self):
            self.test = nnx.Param(jnp.arange(4, dtype=jnp.float32))
    params = nnx.state(Toy())
    def assets(path):
        (path / "test.txt").write_text("saved")
    manager = ocp.CheckpointManager(tmp_path / "best",
        item_handlers={"params": ocp.PyTreeCheckpointHandler(), "assets": checkpoints.CallbackHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True))
    try:
        manager.save(1, {"params": {"params": params}, "assets": assets})
        manager.wait_until_finished()
        restored = restore_params(str(tmp_path / "best/1/params"))
        np.testing.assert_array_equal(restored["test"], np.arange(4))
        assert (tmp_path / "best/1/assets/test.txt").read_text() == "saved"
    finally:
        manager.close()


def test_evaluator_fixed_noise_and_best_selection(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.shared import normalize
    from vla_precision.integrations.openpi.configs import get_config
    from vla_precision.integrations.openpi.fulltask_eval import FullTaskEvaluator
    root = tmp_path / "data/p2"
    root.mkdir(parents=True)
    dest = tmp_path / "data/shared/episode_000001"
    dest.mkdir(parents=True)
    np.savez(dest / "numeric.npz", state=np.zeros((40, 59), np.float32),
             commands=commands(), train_anchors=moving_anchors(commands()))
    (root / "manifest.json").write_text(json.dumps(dict(format=FORMAT, phase=2,
        source=str(tmp_path), shared_root="../shared",
        episodes=[dict(episode=1, split="validation", frames=40)])))
    monkeypatch.setattr(FullTaskDataset, "image", lambda *args: np.zeros((270, 270, 3), np.uint8))
    base = get_config("pi05_full_finetune_pipette_fulltask")
    cfg = dataclasses.replace(base, exp_name="tiny", checkpoint_base_dir=str(tmp_path / "checkpoints"),
                              assets_base_dir=str(tmp_path / "assets"),
                              data=dataclasses.replace(base.data, repo_id="tiny"))
    cfg.checkpoint_dir.mkdir(parents=True)
    stats = {key: normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim),
                                     q01=-np.ones(dim), q99=np.ones(dim))
             for key, dim in (("state", 59), ("actions", 32))}
    normalize.save(cfg.assets_dirs / "tiny", stats)
    options = SimpleNamespace(eval_samples=4, eval_batch_size=2, eval_seed=42, num_inference_steps=10)
    root_cfg = SimpleNamespace(openpi=options, data=SimpleNamespace(lerobot_root=str(root)))
    class Toy(nnx.Module):
        def __init__(self):
            self.bias = nnx.Param(jnp.zeros((30, 32)))
        def sample_actions(self, rng, observation, num_steps):
            return self.bias.value + jax.random.normal(rng, (len(observation.state), 30, 32)) * .01
    graph, params = nnx.split(Toy())
    state = SimpleNamespace(params=params, model_def=graph, ema_params=None)
    evaluator = FullTaskEvaluator(cfg, root_cfg)
    try:
        first = evaluator.evaluate(state, 1)
        second = evaluator.evaluate(state, 2)
        assert first == second
        assert json.loads((Path(cfg.checkpoint_dir) / "best_selection.json").read_text())["step"] == 1
        assert len((Path(cfg.checkpoint_dir) / "holdout_metrics.jsonl").read_text().splitlines()) == 2
    finally:
        evaluator.close()
