"""Actual FOCUS loop instrumentation with fake tracker; no SDK/GPU/network."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"starVLA"))
from deployment.model_server import focus_view as fv
from vla_precision.gr00t_rlt.features import FrozenFocusFeatures, FrozenPI05Features

SENTENCE = "Pick up the pipette from the holder on the right with the right hand."


class Tracker:
    def __init__(self, **kwargs):
        self.locked = False

    def reset(self, target):
        self.target = target

    def step(self, frame):
        self.owner._stop = True  # one iteration of the REAL background loop
        return dict(target="holder", window=[0,0,300,300], center=[150,150],
                    acquired=False, locked=self.locked, ms=1.)

    def crop(self, frame, rec):
        return frame[:224, :224].copy()


@pytest.mark.parametrize("locked", [False, True])
def test_background_paths_emit_timing_without_changing_crop(monkeypatch, locked):
    monkeypatch.setattr(fv, "FocusTracker", Tracker)
    monkeypatch.setattr(fv.time, "sleep", lambda _: None)
    focus = fv.FocusViewSynth()
    focus.tracker.owner = focus
    focus.tracker.locked = locked
    ego = np.full((720,1280,3), 91, np.uint8)
    focus._cam = SimpleNamespace(recv=lambda **kw: ego)
    focus._sentence = SENTENCE
    focus._loop()
    info = focus.timing.snapshot()
    assert info["counts"]["tracking" if locked else "detection"] == 1
    crop, rec = focus(ego, SENTENCE)
    np.testing.assert_array_equal(crop, ego[:224,:224])
    assert rec["generation"] == 0 and not rec["using_prior"]
    assert rec["record_age_ms"] >= 0
    focus.reset()
    _, rec = focus(ego, SENTENCE)
    assert rec["using_prior"] and rec["record_age_ms"] is None


def test_live_features_report_active_detection_overlap_and_phase(monkeypatch):
    monkeypatch.setattr(fv, "FocusTracker", Tracker)
    focus = fv.FocusViewSynth()
    focus._cam = object()  # background mode; no socket or tracking thread
    extractor = FrozenFocusFeatures.__new__(FrozenFocusFeatures)
    extractor.focus = focus
    extractor.last_timings = {}
    def model(self, sample):
        self.last_timings = {"model_predict_ms": 12.}
        return {}
    monkeypatch.setattr(FrozenPI05Features, "extract", model)
    token = focus.timing.begin("detection", "holder", 1)
    ego = np.full((720,1280,3), 17, np.uint8)
    result = extractor.extract({"observation.images.rgb":ego,
                                "observation.state":np.zeros(59), "task":SENTENCE})
    focus.timing.end(token)
    d = extractor.last_focus_diagnostics
    assert d["detection_overlap_ms"] > 0 and d["tracking_overlap_ms"] == 0
    assert d["generation"] == 1 and d["using_prior"]
    assert d["active"][0]["target"] == "holder"
    np.testing.assert_array_equal(result["rgb"], ego)
    assert extractor.last_timings["focus_lock_wait_ms"] >= 0
