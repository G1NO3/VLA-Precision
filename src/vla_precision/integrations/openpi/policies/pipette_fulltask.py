"""Two-view / 59-state / 32-action adapter for native JAX pi0.5."""
import dataclasses

import numpy as np
from PIL import Image
from openpi import transforms


@dataclasses.dataclass(frozen=True)
class FullTaskInputs(transforms.DataTransformFn):
    training: bool = False

    def with_training(self, training):
        return dataclasses.replace(self, training=training)

    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (59,) or not np.isfinite(state).all():
            raise ValueError("Full-task pi05 requires all 59 raw state channels")
        images = {}
        for key in ("base_0_rgb", "left_wrist_0_rgb"):
            arr = np.asarray(data[key])
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("Full-task camera inputs must be HWC RGB uint8")
            image = Image.fromarray(arr).resize((270, 270), Image.Resampling.BILINEAR)
            y, x = np.random.randint(0, 15, size=2) if self.training else (7, 7)
            image = image.crop((int(x), int(y), int(x) + 256, int(y) + 256))
            images[key] = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
        images["right_wrist_0_rgb"] = np.zeros((224, 224, 3), np.uint8)
        result = {"state": state.copy(), "image": images,
                  "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                                 "right_wrist_0_rgb": np.False_}, "prompt": data["prompt"]}
        if "actions" in data:
            actions = np.asarray(data["actions"], np.float32)
            if actions.shape != (30, 32) or not np.isfinite(actions).all():
                raise ValueError("Full-task actions must be finite [30,32]")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class FullTaskStateDropout(transforms.DataTransformFn):
    """Zero the entire NORMALIZED state, only before training tokenization."""
    probability: float = 0.8
    training: bool = False

    def with_training(self, training):
        return dataclasses.replace(self, training=training)

    def __call__(self, data):
        if self.training and np.random.random() < self.probability:
            return {**data, "state": np.zeros_like(data["state"])}
        return data


@dataclasses.dataclass(frozen=True)
class FullTaskOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.shape[-1] != 32:
            raise ValueError("Expected 32 output channels; do not use the XYZ-only adapter")
        return {"actions": actions}
