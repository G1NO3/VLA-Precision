"""No GPU/camera/robot: real SQLite + exact saved feature/crop evidence."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt import bootstrap as b
from vla_precision.gr00t_rlt.core import FOCUS, PI05, bridge_profile, load_config, sha256
from vla_precision.gr00t_rlt.replay import Replay
from test_gr00t_rlt_live import record


@pytest.mark.parametrize("phase", ["p1", "p2", "p3", "p4", "p5"])
def test_focus_separate_contract_and_same_reward(phase, tmp_path):
    c = load_config(FOCUS, phase, tmp_path)
    v5 = load_config(PI05, phase, tmp_path)
    assert c["contract_sha256"] != v5["contract_sha256"]
    assert c["reward"] == v5["reward"]
    assert c["views"] == ["rgb", "focus"]
    assert c["baseline"]["step"] == 21600
    assert bridge_profile(c).endswith("egofocus_native_inspire")


def setup(tmp_path, monkeypatch):
    c = load_config(FOCUS, workspace=tmp_path)
    monkeypatch.setattr(b, "check_code", lambda c: None)
    monkeypatch.setattr(b, "validate_assets", lambda c: {"fixture": True})
    monkeypatch.setattr(b.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 10 << 30})())
    return c, Replay(tmp_path / "run")


def add(c, db, outcome="success"):
    r = record(c)
    f = dict(prefix=np.ones((3, 2048), np.float16), mask=np.array([True, True, False]),
             state=r["state"], reference=r["reference"],
             rgb=np.full((8, 9, 3), 37, np.uint8), focus=np.full((4, 4, 3), 13, np.uint8),
             focus_record=np.asarray('{"locked":true}'), physical_state=np.arange(59, dtype=np.float32),
             instruction=np.asarray("fixture"))
    evidence = b.save_observation(db.root, f, c, "request-1")
    r.update({k: np.asarray(v) for k, v in evidence.items()})
    r.update({"next_"+k: np.asarray(v) for k, v in evidence.items()})
    eid = db.begin(dict(phase=c["phase"], contract_sha256=c["contract_sha256"],
                        token_sha256=b.bootstrap_token(c), behavior_sha256="reference"))
    db.append(eid, r)
    if outcome != "running":
        db.finish(eid, outcome, 1000000001)
    return eid, r


def test_cache_preserves_pixels_masks_split_and_discard(tmp_path, monkeypatch):
    c, db = setup(tmp_path, monkeypatch)
    good, row = add(c, db)
    other, _ = add(c, db, "failure")
    add(c, db, "discarded")
    add(c, db, "aborted")
    add(c, db, "running")
    with pytest.raises(ValueError, match="trained token"):
        db.export(c, b.bootstrap_token(c), tmp_path/"bad.npz")
    result = b.cache_bootstrap(c, db.root)
    assert result["episodes"] == 2 and result["observations"] == 2
    root = Path(c["output"])/"features"
    manifest = json.loads((root/"manifest.json").read_text())
    assert {r["split"] for r in manifest["records"]} == {"train", "validation"}
    for r in manifest["records"]:
        with np.load(root/r["file"]) as f:
            assert np.all(f["rgb"] == 37) and np.all(f["focus"] == 13)
            assert np.array_equal(f["demo_valid"], row["valid"])
            assert not f["demo_valid"][3:].any()  # no padded/unpublished BC labels
    db.discard(good)
    with pytest.raises(ValueError, match="discarded"):
        b.source_check(manifest)
    db.close()


def test_capture_rejects_corruption_and_misaligned_state(tmp_path, monkeypatch):
    c, db = setup(tmp_path, monkeypatch)
    _, row = add(c, db)
    changed = {**row, "state": row["state"]+1}
    with pytest.raises(ValueError, match="alignment"):
        b.load_observation(db.root, changed, c)
    changed = {**row, "observation_sha256": np.asarray("wrong")}
    with pytest.raises(ValueError, match="hash"):
        b.load_observation(db.root, changed, c)
    changed = {**row, "observation_file": np.asarray("../../escape.npz")}
    with pytest.raises(ValueError, match="escapes"):
        b.load_observation(db.root, changed, c)
    db.close()


class TinyToken(torch.nn.Module):
    def encode_flat(self, prefix, mask):
        return torch.ones(len(prefix), 256) * 7


def prepare_token(c, monkeypatch):
    from vla_precision.gr00t_rlt import model
    monkeypatch.setattr(model, "token_model", lambda c: TinyToken())
    p = Path(c["output"])/"token/best.pt"
    p.parent.mkdir(parents=True)
    torch.save(dict(contract_sha256=c["contract_sha256"], state={}), p)
    return p


def test_encoding_preserves_ids_outcomes_audits_and_export(tmp_path, monkeypatch):
    c, db = setup(tmp_path, monkeypatch)
    eid, _ = add(c, db)
    discarded, _ = add(c, db, "discarded")
    p = prepare_token(c, monkeypatch)
    result = b.encode_bootstrap(c, db.root)
    assert result["episodes"] == 1
    assert db.counts() == {"success": 1, "discarded": 1}
    assert db.db.execute("SELECT token FROM episodes WHERE id=?", (discarded,)).fetchone()[0] == b.bootstrap_token(c)
    assert b.encode_bootstrap(c, db.root)["episodes"] == 0  # idempotent
    dest = tmp_path/"replay.npz"
    db.export(c, sha256(p), dest)
    with np.load(dest) as f:
        assert np.all(f["token"] == 7) and np.all(f["next_token"] == 7)
    db.discard(eid)  # same episode identity still revokes export AFTER encoding
    with pytest.raises(ValueError, match="No complete"):
        db.export(c, sha256(p), tmp_path/"discarded.npz")
    db.close()


def test_encoding_rolls_back_entire_transaction_on_missing_evidence(tmp_path, monkeypatch):
    c, db = setup(tmp_path, monkeypatch)
    add(c, db)
    bad, row = add(c, db)
    row["observation_sha256"] = np.asarray("bad")
    stream = io.BytesIO()
    np.savez_compressed(stream, **row)
    with db.db:
        db.db.execute("UPDATE transitions SET payload=? WHERE episode=?", (stream.getvalue(), bad))
    prepare_token(c, monkeypatch)
    with pytest.raises(ValueError, match="hash"):
        b.encode_bootstrap(c, db.root)
    assert {r[0] for r in db.db.execute("SELECT token FROM episodes")} == {b.bootstrap_token(c)}
    db.close()


def test_focus_feature_input_uses_crop_not_wrist_and_resets(monkeypatch):
    from vla_precision.gr00t_rlt.features import FrozenFocusFeatures, FrozenPI05Features
    calls = []
    class Focus:
        from threading import Lock
        gpu_lock = Lock()
        def __call__(self, ego, sentence):
            calls.append(sentence)
            return np.full((224,224,3), 42, np.uint8), {"locked": True}
        def reset(self):
            calls.append("reset")
    e = FrozenFocusFeatures.__new__(FrozenFocusFeatures)
    e.focus = Focus()
    def extract(self, sample):
        assert np.all(sample["observation.images.focus"] == 42)
        return {}
    monkeypatch.setattr(FrozenPI05Features, "extract", extract)
    sample = {"observation.images.rgb": np.zeros((720,1280,3), np.uint8),
              "observation.state": np.zeros(59), "task": "pick"}
    result = e.extract(sample)
    assert result["focus"].shape == (224,224,3)
    e.reset()
    assert calls == ["pick", "reset"]
    sample["observation.images.rgb"] = np.zeros((224,224,3), np.uint8)
    with pytest.raises(ValueError, match="1280x720"):
        e.extract(sample)


def test_live_bootstrap_returns_reference_without_using_actor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vla_precision.gr00t_rlt.serving import LivePolicy
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"VLAPolicyBridge"))
    from vla_policy_bridge.pi05_fulltask import synthetic_observation
    c, db = setup(tmp_path, monkeypatch)
    c["action_active"] = [i not in (12,17) for i in range(18)]
    p = LivePolicy.__new__(LivePolicy)
    p.c, p.device, p.bootstrap = c, "cpu", True
    p.episode = None
    p.capture_root = db.root
    p.token_sha, p.behavior, p.actor_kind = b.bootstrap_token(c), "baseline-sha", "frozen_reference"
    p.stats = {"q01": [-1.]*32, "q99": [1.]*32}
    resets = []
    def features(sample):
        assert "observation.images.wrist_left" not in sample
        return dict(prefix=np.ones((3,2048),np.float16), mask=np.ones(3,bool),
                    state=np.zeros(59,np.float32), reference=np.full((30,32), .25, np.float32),
                    rgb=sample["observation.images.rgb"], focus=np.zeros((224,224,3),np.uint8))
    p.features = SimpleNamespace(extract=features, reset=lambda: resets.append(True),
        physical_actions=lambda action, ref: np.concatenate((action, ref[:,18:]), -1))
    p._load_actor = lambda: pytest.fail("Bootstrap must not use a learned/random actor")
    obs = synthetic_observation()
    obs["video"].pop("wrist_left")
    result = p.get_action(obs, {"episode": "test-episode"})
    assert len(resets) == 1
    assert result["info"]["rlt"]["actor_kind"] == "frozen_reference"
    assert np.all(result["right_wrist_rel6"] == .25)
    assert np.all(result["info"]["rlt"]["token"] == 0)
    assert (db.root / result["info"]["rlt"]["observation_file"]).is_file()
    p.get_action(obs, {"episode": "test-episode"})
    assert len(resets) == 1
    p.get_action(obs, {"episode": "new-episode"})
    assert len(resets) == 2
    db.close()


def test_cpu_initialization_and_first_rl_update(tmp_path, monkeypatch):
    """Synthetic temp data only: exercise real token loss, masked BC and RL code."""
    import importlib.util
    from types import SimpleNamespace
    c, db = setup(tmp_path, monkeypatch)
    c["rlinf_root"] = load_config(FOCUS)["rlinf_root"]
    # Small batch/sequence for CPU CI; same production model and mask semantics.
    c["token"]["batch_size"] = 1
    c["token"]["prefix_seq_len"] = 4
    c["learner"]["batch_size"] = 2
    add(c, db)
    add(c, db, "failure")
    b.cache_bootstrap(c, db.root)
    path = Path(__file__).resolve().parents[1]/"scripts/gr00t_rlt.py"
    spec = importlib.util.spec_from_file_location("focus_test_train_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "check_code", lambda c: None)
    monkeypatch.setattr(cli, "validate_assets", lambda c: {"fixture": True})
    args = SimpleNamespace(device="cpu", updates=1)
    cli.fit_token(c, args)
    cli.fit_bc(c, args)
    out = Path(c["output"])
    assert (out/"token/metadata.json").is_file() and (out/"bc/metadata.json").is_file()
    result = b.encode_bootstrap(c, db.root)
    assert result["windows"] == 2
    dest = tmp_path/"rl.npz"
    db.export(c, result["token_sha256"], dest)
    cli.fit_heads(c, SimpleNamespace(device="cpu", updates=2, replay=dest, resume=None))
    assert (out/"learner/latest.json").is_file()
    metrics = [json.loads(s) for s in (out/"learner/metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and metrics[-1]["actor_updated"]
    db.close()
