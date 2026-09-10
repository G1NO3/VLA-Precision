"""Prepared pipette samples for native OpenPI, independent of LeRobot versions.

The source is LeRobot v2.1, whereas the pinned reader requires v3. Image caches
preserve source frame indices and native OpenPI resizing. Action chunks are
restricted to complete, real transitions within each episode (no tail repeats).
"""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np

FORMAT = "vla_precision.pipette_sft.v1"
VIEWS = ("rgb", "wrist_left", "wrist_right")
VALIDATION_EPISODES = (9, 10, 12, 18, 21, 23, 28, 29, 33)

class PipetteDataset:
    def __init__(self, root, action_horizon, *, split="train"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != FORMAT:
            raise ValueError("Unsupported pipette cache format")
        if split not in ("train", "validation") or action_horizon < 1:
            raise ValueError("Invalid split or action horizon")
        self.action_horizon = action_horizon
        self.split = split
        self.episodes = {}
        self.samples = []
        for episode in self.manifest["episodes"]:
            if episode["split"] != split:
                continue
            ep = episode["episode"]
            with np.load(self.root / f"episode_{ep:06d}" / "numeric.npz") as arrays:
                self.episodes[ep] = {key: arrays[key] for key in arrays.files}
            n = len(self.episodes[ep]["actions"])
            self.samples.extend((ep, frame) for frame in range(n - action_horizon + 1))
        if not self.samples:
            raise ValueError("No complete action chunks in selected split")

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=9)
    def _image_array(self, ep, view):
        return np.load(self.root / f"episode_{ep:06d}" / f"{view}.npy", mmap_mode="r")

    def numeric_item(self, index):
        ep, frame = self.samples[index]
        data = self.episodes[ep]
        return {"state": data["state"][frame],
                "actions": data["actions"][frame:frame + self.action_horizon]}

    def __getitem__(self, index):
        ep, frame = self.samples[index]
        numeric = self.numeric_item(index)
        return {
            "observation.state": numeric["state"], "action": numeric["actions"],
            "task": self.manifest["prompt"],
            **{f"observation.images.{view}": self._image_array(ep, view)[frame] for view in VIEWS},
        }
