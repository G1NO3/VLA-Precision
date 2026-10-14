"""Use starVLA's own serving crop, normalization, backbone and flow sampler."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


def frozen_features(c, device="cuda"):
    from .core import is_pi05, is_focus
    cls = FrozenFocusFeatures if is_focus(c) else (FrozenPI05Features if is_pi05(c) else FrozenGR00TFeatures)
    return cls(c, device)


def normalization_processor(c):
    sys.path.insert(0, c["starvla_root"])
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    return PolicyNormProcessor(c["checkpoint"], unnorm_key=c["unnorm_key"])


def normalize_actions(processor, actions):
    """Forward the saved action transforms, including constant-dimension handling."""
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
    values, offset = {}, 0
    for key in processor.action_keys:
        width = processor.action_key_dims[key]
        values[key] = np.asarray(actions[:, offset:offset + width], np.float32)
        offset += width
    if offset != actions.shape[-1]:
        raise ValueError("Action normalization width mismatch")
    for transform in processor.transform.transforms:
        if isinstance(transform, (StateActionToTensor, StateActionTransform)):
            if set(processor.action_keys).intersection(transform.apply_to):
                values = transform(values)
    return np.concatenate([np.asarray(values[k]) for k in processor.action_keys], axis=-1)


class FrozenGR00TFeatures:
    """Single-owner offline extractor. Capture the exact features used for reference.

    It deliberately does not expose a robot action publisher or async inference.
    Inputs use the prepared dataset's RGB arrays and physical state59.
    """
    def __init__(self, c, device="cuda"):
        sys.path.insert(0, c["starvla_root"])
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        self.c = c
        self.wrapper = PolicyServerWrapper(c["checkpoint"], device=device, use_bf16=True,
                                           unnorm_key=c["unnorm_key"])
        self.model = self.wrapper._framework
        self.model.requires_grad_(False).eval()
        self.model._freeze_backbone = True
        self.proc = self.wrapper.get_norm_processor()
        if sum(self.proc.state_key_dims.values()) != 59 or sum(self.proc.action_key_dims.values()) != 32:
            raise ValueError("GR00T serving processor has unexpected layout")

    @torch.inference_mode()
    def extract(self, sample):
        started = time.perf_counter()
        raw_state = np.asarray(sample["observation.state"], np.float32)
        if raw_state.shape != (59,) or not np.isfinite(raw_state).all():
            raise ValueError("Expected finite physical state59")
        images = [sample[f"observation.images.{view}"] for view in self.c["views"]]
        if any(im.ndim != 3 or im.shape[-1] != 3 or im.dtype != np.uint8 for im in images):
            raise ValueError("Expected uint8 RGB images, not BGR")
        images = self.proc.data_config.serve_view_preprocess(images)
        state = self.proc.apply_state(raw_state)
        example = {"image": images, "lang": sample["task"], "state": state}
        prepared = time.perf_counter()
        prediction, captured = self._predict_capture(example)
        predicted = time.perf_counter()
        # Preserve the original float16 cache/actor quantization exactly, but
        # do it on device. Live actors reuse this tensor rather than uploading
        # the complete prefix again after its archival CPU copy.
        prefix_device = captured["prefix"][0].detach().to(torch.float16)
        mask_device = captured["mask"][0].detach().bool()
        prefix = prefix_device.cpu().numpy()
        mask = mask_device.cpu().numpy()
        if (prefix.ndim != 2 or mask.shape != prefix.shape[:1]
                or prefix.shape[1] != self.c["token"]["input_dim"]
                or len(prefix) > self.c["token"]["prefix_seq_len"]
                or not np.isfinite(prefix).all()):
            raise ValueError(f"Feature shape outside configured token capacity: {prefix.shape}")
        if prediction.shape != (30, 32) or not mask.any() or not np.isfinite(prediction).all():
            raise ValueError("Invalid frozen reference/feature mask")
        result = dict(prefix=prefix, mask=mask, state=state.astype(np.float32),
                      reference=prediction.astype(np.float32))
        if getattr(self, "keep_device_features", False):
            result.update(_prefix_device=prefix_device, _mask_device=mask_device)
        if "action" in sample:
            result["demo_action"] = normalize_actions(self.proc, sample["action"])
        self.last_timings = dict(preprocess_ms=(prepared-started)*1000,
                                 model_predict_ms=(predicted-prepared)*1000,
                                 feature_archive_copy_ms=(time.perf_counter()-predicted)*1000)
        return result

    def _predict_capture(self, example):
        captured = {}
        original = self.model._encode_vl

        def encode(images, instructions):
            result = original(images, instructions)
            captured["prefix"], _, captured["mask"] = result
            return result

        # Capture after the production model's resize and processor. No second
        # forward, different sampling seed or alternative vision preprocessing.
        self.model._encode_vl = encode
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(self.c["reference_seed"])
                prediction = self.model.predict_action([example])["normalized_actions"][0]
        finally:
            self.model._encode_vl = original
        return prediction, captured

    def physical_actions(self, actor_action, reference):
        """Retain frozen 14-joint IK seeds; return commands, not motor targets."""
        if actor_action.shape != (30, 18) or reference.shape != (30, 32):
            raise ValueError("Expected actor30x18 and reference30x32")
        action = np.concatenate([actor_action, reference[:, 18:]], axis=-1)
        if not np.isfinite(action).all():
            raise ValueError("Non-finite actor output")
        return self.proc.unapply_actions(action)


class FrozenPI05Features(FrozenGR00TFeatures):
    """PI05 v5: capture final PaliGemma prefix and its real padding/view mask.

    The frozen reference receives NO state, exactly like the BC server. The
    normalized physical state59 remains an explicit input to the new RL head.
    Missing right-wrist images stay masked; do not pool black-image tokens.
    """
    def __init__(self, c, device="cuda"):
        super().__init__(c, device)
        tokenizer = self.model.tokenizer
        if not Path(tokenizer.model_path).is_absolute():
            tokenizer.model_path = str((Path(c["starvla_root"]) / tokenizer.model_path).resolve(strict=True))

    def _predict_capture(self, example):
        example = {k: v for k, v in example.items() if k != "state"}
        captured = {}
        embed, forward = self.model.embed_prefix, self.model.vlm.forward_prefix

        def embed_prefix(*args, **kwargs):
            result = embed(*args, **kwargs)
            captured["mask"] = result[1]
            return result

        def forward_prefix(*args, **kwargs):
            result = forward(*args, **kwargs)
            captured["prefix"] = result.last_hidden_state
            return result

        self.model.embed_prefix = embed_prefix
        self.model.vlm.forward_prefix = forward_prefix
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(self.c["reference_seed"])
                prediction = self.model.predict_action([example])["normalized_actions"][0]
        finally:
            self.model.embed_prefix, self.model.vlm.forward_prefix = embed, forward
        return prediction, captured


class FrozenFocusFeatures(FrozenPI05Features):
    """Same 10-Hz background tracker as IL; save the exact served crop/prefix.

    Offline training does not rerun the temporally stateful tracker. The
    camera endpoint is runtime configuration, not a trained model parameter.
    """
    def __init__(self, c, device="cuda"):
        super().__init__(c, device)
        from deployment.model_server.focus_view import FocusViewSynth
        camera = c.get("focus_camera")
        if not camera:
            raise ValueError("FOCUS live extraction requires --focus-camera for the ego stream")
        self.focus = FocusViewSynth(sam=c["focus"]["sam"], device=device,
                                   camera=camera, rate_hz=c["focus"]["rate_hz"])

    def reset(self):
        self.focus.reset()

    def extract(self, sample):
        import json
        started = time.perf_counter()
        ego = sample["observation.images.rgb"]
        if ego.shape != (720, 1280, 3) or ego.dtype != np.uint8:
            raise ValueError("FOCUS requires original 1280x720 RGB, no pre-resized or missing camera")
        crop, record = self.focus(ego, sample["task"])
        cropped = time.perf_counter()
        with self.focus.gpu_lock:
            acquired = time.perf_counter()
            value = super().extract({**sample, "observation.images.focus": crop})
            completed = time.perf_counter()
        timing = getattr(self.focus, "timing", None)
        self.last_focus_diagnostics = dict(timing.snapshot(acquired, completed) if timing else {},
            available=timing is not None,
            target=record.get("target"), generation=record.get("generation"),
            locked=bool(record.get("locked")), using_prior=record.get("using_prior"),
            record_age_ms=record.get("record_age_ms"), sam_cache=record.get("sam_cache"))
        self.last_timings = {**getattr(self, "last_timings", {}),
                             "focus_crop_ms": (cropped-started)*1000,
                             "focus_lock_wait_ms": (acquired-cropped)*1000}
        if timing is not None:
            self.last_timings.update(
                focus_detection_overlap_ms=self.last_focus_diagnostics["detection_overlap_ms"],
                focus_tracking_overlap_ms=self.last_focus_diagnostics["tracking_overlap_ms"])
        value.update(rgb=ego.copy(), focus=crop.copy(), focus_record=np.asarray(json.dumps(record)),
                     physical_state=np.asarray(sample["observation.state"], np.float32),
                     instruction=np.asarray(sample["task"]))
        return value
