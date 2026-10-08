"""Five-phase, bimanual G1 data contract (independent of LeRobot/starVLA).

Commands, NOT measured motion, supervise actions. Translation is expressed in
pelvis axes; relative rotation is R_anchor.T @ R_target, not rotvec subtraction.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FORMAT = "vla_precision.pipette_fulltask.v1"
DATASET_REPO = "jren313/g1-pipette-2view-teleop0925-5task-eerel"
DATASET_REVISION = "340ebabb48e6d1fb42f92178fba0f6ded87322df"
VIEWS = ("rgb", "wrist_left")
# Published per-phase model cards; never re-randomize pick/return halves.
VALIDATION = {
    1: (10, 12, 32, 40, 68, 78, 82, 102, 118),
    2: (130, 146, 162, 174, 176),
    3: (199, 200, 208, 214, 215, 219, 228),
    4: (131, 147, 163, 175, 177),
    5: (11, 13, 33, 41, 69, 79, 83, 103, 119),
}
TASKS = {
    1: "Pick up the pipette from the holder on the right with the right hand.",
    2: "Pick up the tube from the blue rack on the left with the left hand.",
    3: "Aim the pipette held in the right hand at the tube held in the left hand.",
    4: "Put the tube held in the left hand back into the blue rack on the left.",
    5: "Put the pipette held in the right hand back into the holder on the right.",
}


def action_chunk(commands, frame, horizon=30, *, left_thumb_relative=False):
    """32-D raw targets -> 32-D chunk-relative/absolute mixed labels."""
    commands = np.asarray(commands)
    if commands.ndim != 2 or commands.shape[1] != 32:
        raise ValueError("Expected commands [N,32]")
    if not 0 <= frame < len(commands) or horizon < 1:
        raise ValueError("Invalid chunk anchor/horizon")
    rows = commands[np.minimum(frame + np.arange(horizon), len(commands) - 1)].copy()
    anchor = commands[frame]
    for start in (0, 6):
        rows[:, start:start + 3] -= anchor[start:start + 3]
        r0 = Rotation.from_rotvec(anchor[start + 3:start + 6])
        rt = Rotation.from_rotvec(rows[:, start + 3:start + 6])
        rows[:, start + 3:start + 6] = (r0.inv() * rt).as_rotvec()
    if left_thumb_relative:
        rows[:, 12] -= anchor[12]
    return rows.astype(np.float32)


def moving_anchors(commands, horizon=30, threshold_mm=1.0):
    """Exact starVLA gate: end-to-start norm of concatenated L/R XYZ."""
    xyz = np.asarray(commands, np.float64)[:, [0, 1, 2, 6, 7, 8]]
    end = np.minimum(np.arange(len(xyz)) + horizon - 1, len(xyz) - 1)
    return np.flatnonzero(np.linalg.norm(xyz[end] - xyz, axis=1) * 1000 >= threshold_mm)


def validate_recording_split(provenance, phase_by_episode):
    """Prevent the two halves of any original recording crossing splits."""
    seen = {}
    for row in provenance:
        ep = int(row["episode_index"])
        phase = phase_by_episode[ep]
        source = row["source_episode"].removesuffix("_pick").removesuffix("_return")
        key = (row["source_session"], source)
        split = "validation" if ep in VALIDATION[phase] else "train"
        if key in seen and seen[key] != split:
            raise ValueError(f"Recording leakage: {key}")
        seen[key] = split


class FullTaskDataset:
    def __init__(self, root, action_horizon=30, *, split="train", left_thumb_relative=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != FORMAT or action_horizon != 30:
            raise ValueError("Expected full-task v1 data and 30-step chunks")
        if split not in ("train", "validation"):
            raise ValueError(split)
        self.phase = int(self.manifest["phase"])
        # None preserves the legacy per-phase recipe; GR00T five-task v2
        # explicitly requests relative thumb labels in ALL phases, including P1.
        if left_thumb_relative is not None and not isinstance(left_thumb_relative, bool):
            raise ValueError("left_thumb_relative must be boolean or None")
        self.left_thumb_relative = self.phase != 1 if left_thumb_relative is None else left_thumb_relative
        self.split = split
        self.action_horizon = action_horizon
        self.shared = (self.root / self.manifest["shared_root"]).resolve()
        self.source = Path(self.manifest["source"])
        self.episodes, self.samples = {}, []
        for row in self.manifest["episodes"]:
            if row["split"] != split:
                continue
            ep = row["episode"]
            with np.load(self.shared / f"episode_{ep:06d}" / "numeric.npz") as arrays:
                data = {key: arrays[key] for key in arrays.files}
            if data["state"].shape != (row["frames"], 59) or data["commands"].shape != (row["frames"], 32):
                raise ValueError(f"Invalid numeric cache for episode {ep}")
            self.episodes[ep] = data
            anchors = data["train_anchors"] if split == "train" else range(row["frames"])
            self.samples.extend((ep, int(f)) for f in anchors)
        if not self.samples:
            raise ValueError("Empty phase/split")

    def __len__(self):
        return len(self.samples)

    def numeric_item(self, index):
        ep, frame = self.samples[index]
        data = self.episodes[ep]
        return {"state": data["state"][frame].copy(),
                "actions": action_chunk(data["commands"], frame, self.action_horizon,
                                        left_thumb_relative=self.left_thumb_relative)}

    @lru_cache(maxsize=8)
    def _cached_video(self, ep, view):
        path = self.shared / f"episode_{ep:06d}" / f"{view}.npy"
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def image(self, ep, frame, view):
        cached = self._cached_video(ep, view)
        if cached is not None:
            return np.asarray(cached[frame])
        # Slow but bounded-memory diagnostic fallback; training should cache images.
        import cv2
        from PIL import Image
        path = self.source / "videos/chunk-000" / f"observation.images.{view}" / f"episode_{ep:06d}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise FileNotFoundError(f"Cannot decode {path}, frame {frame}")
            return np.asarray(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((270, 270), Image.Resampling.BILINEAR))
        finally:
            cap.release()

    def __getitem__(self, index):
        ep, frame = self.samples[index]
        item = self.numeric_item(index)
        return {"observation.state": item["state"], "action": item["actions"],
                "task": TASKS[self.phase],
                **{f"observation.images.{v}": self.image(ep, frame, v) for v in VIEWS}}

    def validation_indices(self, count=480, seed=42):
        """Fixed, episode-balanced selection; NOT starVLA's random 480 indices."""
        if self.split != "validation" or count < 1:
            raise ValueError("Validation split and positive sample count required")
        rng = np.random.default_rng(seed)
        buckets = [[i for i, (e, _) in enumerate(self.samples) if e == ep]
                   for ep in sorted(self.episodes)]
        result = []
        for n, ids in enumerate(buckets):
            k = count // len(buckets) + (n < count % len(buckets))
            result.extend(rng.choice(ids, size=k, replace=k > len(ids)).tolist())
        return sorted(result)
