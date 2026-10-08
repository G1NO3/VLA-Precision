"""Use starVLA's own serving crop, normalization, backbone and flow sampler."""
from __future__ import annotations

import sys

import numpy as np
import torch


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
        raw_state = np.asarray(sample["observation.state"], np.float32)
        if raw_state.shape != (59,) or not np.isfinite(raw_state).all():
            raise ValueError("Expected finite physical state59")
        images = [sample[f"observation.images.{view}"] for view in self.c["views"]]
        if any(im.ndim != 3 or im.shape[-1] != 3 or im.dtype != np.uint8 for im in images):
            raise ValueError("Expected uint8 RGB images, not BGR")
        images = self.proc.data_config.serve_view_preprocess(images)
        state = self.proc.apply_state(raw_state)
        example = {"image": images, "lang": sample["task"], "state": state}
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
        prefix = captured["prefix"][0].float().cpu().numpy()
        mask = captured["mask"][0].cpu().numpy().astype(bool)
        if prefix.shape[1] != self.c["token"]["input_dim"] or len(prefix) > self.c["token"]["prefix_seq_len"]:
            raise ValueError(f"Feature shape outside configured token capacity: {prefix.shape}")
        if prediction.shape != (30, 32) or not mask.any() or not np.isfinite(prediction).all():
            raise ValueError("Invalid frozen reference/feature mask")
        result = dict(prefix=prefix.astype(np.float16), mask=mask, state=state.astype(np.float32),
                      reference=prediction.astype(np.float32))
        if "action" in sample:
            result["demo_action"] = normalize_actions(self.proc, sample["action"])
        return result

    def physical_actions(self, actor_action, reference):
        """Retain frozen 14-joint IK seeds; return commands, not motor targets."""
        if actor_action.shape != (30, 18) or reference.shape != (30, 32):
            raise ValueError("Expected actor30x18 and reference30x32")
        action = np.concatenate([actor_action, reference[:, 18:]], axis=-1)
        if not np.isfinite(action).all():
            raise ValueError("Non-finite actor output")
        return self.proc.unapply_actions(action)
