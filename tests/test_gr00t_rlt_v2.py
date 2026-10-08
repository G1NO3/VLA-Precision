"""Unified five-task GR00T contracts; no network, model loading or robot I/O."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_precision.data.pipette_fulltask import FORMAT, FullTaskDataset
from vla_precision.gr00t_rlt.core import DEFAULT, V2_PHASE_HELD, load_config, sha256, validate_assets


def mock_assets(tmp_path, phase="p1"):
    c = load_config(phase=phase, workspace=tmp_path)
    root = Path(c["run_root"])
    root.mkdir(parents=True)
    stats = {key: {"q01": [-1.] * width, "q99": [1.] * width}
             for key, width in (("state", 59), ("action", 32))}
    # A constant channel must stay inactive in addition to phase-held channels.
    stats["action"]["q99"][0] = -1.
    (root / "dataset_statistics.json").write_text(json.dumps({"new_embodiment": stats}))
    cfg = dict(framework=dict(name="CosmosGR00TN1d7", qwenvl=dict(select_layer=16),
               action_model=dict(state_dim=59, action_dim=32, action_horizon=30)),
               datasets=dict(vla_data=dict(action_mode_apply_keys=["action.left_hand_rel"])))
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    weight = Path(c["checkpoint"])
    weight.parent.mkdir()
    weight.write_bytes(b"test fixture, never a deployable checkpoint")
    receipt = dict(repo=c["baseline"]["repo"], revision=c["baseline"]["revision"], files={})
    for p in (root / "config.yaml", root / "dataset_statistics.json", weight):
        receipt["files"][str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    (root / "download_receipt.json").write_text(json.dumps(receipt))
    c = load_config(phase=phase, workspace=tmp_path)
    c["baseline"]["weights_sha256"] = sha256(weight)
    return c


@pytest.mark.parametrize("phase", list(V2_PHASE_HELD))
def test_v2_shared_checkpoint_and_phase_mask(tmp_path, phase):
    c = mock_assets(tmp_path, phase)
    assert c["baseline"]["repo"] == "jren313/starvla-pipette-5task-v2"
    assert c["baseline"]["step"] == 9000
    assert c["baseline"]["left_thumb"] == "chunk_relative"
    assert not c["live_enabled"]
    expected = [True] * 18
    for i in [0] + V2_PHASE_HELD[phase]:
        expected[i] = False
    assert c["action_active"] == expected
    assert validate_assets(c, verify_weights=True)["action_active"] == expected


@pytest.mark.parametrize("phase", list(V2_PHASE_HELD))
def test_masks_match_deployed_bridge(monkeypatch, phase):
    bridge = Path(__file__).resolve().parents[2] / "VLAPolicyBridge"
    if not bridge.is_dir():
        pytest.skip("Optional cross-repository contract check")
    monkeypatch.syspath_prepend(str(bridge))
    from vla_policy_bridge.phase_poses import gr00t_phase_hand_holds
    from vla_policy_bridge.pi05_fulltask import INSTRUCTIONS
    held = gr00t_phase_hand_holds(INSTRUCTIONS[int(phase[1:]) - 1])
    indices = ([12] if 4 in held.get("left", ()) else [])
    indices += [13 + i for i in held.get("right", ())]
    assert indices == V2_PHASE_HELD[phase]


@pytest.mark.parametrize("change,match", [
    ("receipt", "missing required artifacts"),
    ("weights", "pinned v2 checkpoint"),
    ("thumb", "left-thumb action semantics"),
])
def test_assets_reject_mismatched_metadata(tmp_path, change, match):
    c = mock_assets(tmp_path)
    root = Path(c["run_root"])
    if change == "receipt":
        path = root / "download_receipt.json"
        receipt = json.loads(path.read_text())
        del receipt["files"]["config.yaml"]
        path.write_text(json.dumps(receipt))
    elif change == "weights":
        c["baseline"]["weights_sha256"] = "wrong"
    else:
        path = root / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["datasets"]["vla_data"]["action_mode_apply_keys"] = []
        path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        validate_assets(c)


def test_dataset_p1_explicit_relative_keeps_legacy_default(tmp_path):
    root = tmp_path / "p1"
    root.mkdir()
    dest = tmp_path / "shared/episode_000000"
    dest.mkdir(parents=True)
    commands = np.zeros((40, 32), np.float32)
    commands[:, 12] = 800 - np.arange(40)
    np.savez(dest / "numeric.npz", commands=commands, state=np.zeros((40, 59), np.float32),
             train_anchors=np.array([5]))
    (root / "manifest.json").write_text(json.dumps(dict(format=FORMAT, phase=1,
        shared_root="../shared", source=str(tmp_path),
        episodes=[dict(episode=0, split="train", frames=40)])))
    old = FullTaskDataset(root).numeric_item(0)["actions"]
    new = FullTaskDataset(root, left_thumb_relative=True).numeric_item(0)["actions"]
    np.testing.assert_allclose(old[:, 12], commands[5:35, 12])
    np.testing.assert_allclose(new[:, 12], -np.arange(30))
    np.testing.assert_array_equal(old[:, 13:], new[:, 13:])


def test_reject_changed_hold_and_live_enabled(tmp_path):
    cfg = yaml.safe_load(DEFAULT.read_text())
    path = tmp_path / "recipe.yaml"
    cfg["phases"]["p1"]["held_action_indices"] = []
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="deployed bridge contract"):
        load_config(path, workspace=tmp_path)
    cfg["phases"]["p1"]["held_action_indices"] = V2_PHASE_HELD["p1"]
    cfg["live_enabled"] = True
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="live_enabled=false"):
        load_config(path, workspace=tmp_path)


def test_inventory_never_claims_online_ready(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/gr00t_rlt.py"
    spec = importlib.util.spec_from_file_location("gr00t_cli_v2", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    c = mock_assets(tmp_path)
    monkeypatch.setattr(cli, "check_code", lambda c: None)
    prepared = Path(c["prepared_root"]) / c["phase"] / "manifest.json"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("{}")
    for name in ("features/manifest.json", "token/best.pt", "bc/best.pt", "learner/latest.json"):
        p = Path(c["output"]) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    result = cli.readiness(c)
    assert result["offline_assets_ready"]
    assert not result["online_ready"] and result["blockers"]
