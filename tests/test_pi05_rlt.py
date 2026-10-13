"""PI05-specific feature and recipe regressions; no CUDA/model download/robot."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import PI05, bridge_profile, load_config
from vla_precision.gr00t_rlt.features import FrozenPI05Features


@pytest.mark.parametrize("phase", ["p1", "p2", "p3", "p4", "p5"])
def test_pi05_contract_is_separate_but_rewards_match(phase, tmp_path):
    c = load_config(PI05, phase, tmp_path)
    g = load_config(phase=phase, workspace=tmp_path)
    assert c["contract_sha256"] != g["contract_sha256"]
    assert c["reward"] == g["reward"]
    assert c["held_action_indices"] == g["held_action_indices"]
    assert c["run_root"].endswith("starvla-pipette-5task-v5-pi05/pi05")
    assert "pi05-fulltask-v5-rlt" in c["output"]
    assert bridge_profile(c) == "starvla_pi05_pipette_5taskv5_inspire"
    assert c["baseline"]["state_input"] is False


class MockPI05:
    def __init__(self, fail=False):
        self.vlm = SimpleNamespace(forward_prefix=self.forward)
        self.fail = fail
        self.calls = 0

    def embed_prefix(self, *args):
        return torch.zeros(1, 4, 2048), torch.tensor([[True, True, False, True]]), None

    def forward(self, **kwargs):
        # Not the input embeddings: final hidden state, which RLToken needs.
        return SimpleNamespace(last_hidden_state=torch.full((1, 4, 2048), 7.))

    def predict_action(self, examples):
        self.calls += 1
        assert "state" not in examples[0]  # exactly the production no-state reference
        assert len(examples[0]["image"]) == 2
        prefix, _, _ = self.embed_prefix()
        self.vlm.forward_prefix(inputs_embeds=prefix)
        if self.fail:
            raise RuntimeError("test inference failure")
        return {"normalized_actions": np.zeros((1, 30, 32), np.float32)}


def extractor(fail=False):
    e = FrozenPI05Features.__new__(FrozenPI05Features)
    e.c = load_config(PI05)
    e.model = MockPI05(fail)
    e.proc = SimpleNamespace(data_config=SimpleNamespace(serve_view_preprocess=lambda x: x),
                             apply_state=lambda x: x * 2)
    return e


def sample():
    return {"observation.state": np.ones(59, np.float32), "task": "pick",
            "observation.images.rgb": np.zeros((4, 4, 3), np.uint8),
            "observation.images.wrist_left": np.zeros((6, 6, 3), np.uint8)}


def test_exact_reference_forward_hidden_features_and_padding_mask():
    e = extractor()
    before = e.model.embed_prefix, e.model.vlm.forward_prefix
    f = e.extract(sample())
    assert e.model.calls == 1
    assert f["prefix"].shape == (4, 2048) and np.all(f["prefix"] == 7)
    assert f["mask"].tolist() == [True, True, False, True]
    assert f["state"].shape == (59,) and np.all(f["state"] == 2)
    assert f["reference"].shape == (30, 32)
    assert (e.model.embed_prefix, e.model.vlm.forward_prefix) == before


def test_hooks_restored_after_reference_failure():
    e = extractor(fail=True)
    before = e.model.embed_prefix, e.model.vlm.forward_prefix
    with pytest.raises(RuntimeError, match="test inference"):
        e.extract(sample())
    assert (e.model.embed_prefix, e.model.vlm.forward_prefix) == before


def test_rng_is_not_changed_by_fixed_seed_reference():
    e = extractor()
    before = torch.get_rng_state().clone()
    e.extract(sample())
    assert torch.equal(before, torch.get_rng_state())


def test_pi05_data_bypasses_gr00t_resize_cache(monkeypatch, tmp_path):
    from vla_precision.data.pipette_fulltask import FullTaskDataset
    import cv2
    ds = FullTaskDataset.__new__(FullTaskDataset)
    ds.raw_images, ds.source = True, tmp_path
    monkeypatch.setattr(ds, "_cached_video", lambda *a: pytest.fail("PI05 must not read resize270 cache"))
    bgr = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    cap = SimpleNamespace(set=lambda *a: None, read=lambda: (True, bgr), release=lambda: None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    result = ds.image(0, 1, "rgb")
    assert result.shape == (7, 11, 3)
    np.testing.assert_array_equal(result, bgr[..., ::-1])
