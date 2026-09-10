"""Three-camera G1 policy: measured joints/hand XYZ -> pelvis-frame delta XYZ."""
import dataclasses
import numpy as np
from openpi import transforms
from vla_precision.integrations.openpi.policies.dual_ur import _parse_image

@dataclasses.dataclass(frozen=True)
class PipetteInputs(transforms.DataTransformFn):
    def __call__(self, data):
        keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {key: _parse_image(data[key]) for key in keys},
            "image_mask": {key: np.True_ for key in keys},
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result

@dataclasses.dataclass(frozen=True)
class PipetteOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :3]}
