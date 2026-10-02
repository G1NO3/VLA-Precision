#!/usr/bin/env python3
"""Prepare the pinned five-task capture for native OpenPI JAX. Never overwrite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from vla_precision.data.pipette_fulltask import (
    DATASET_REPO, DATASET_REVISION, FORMAT, TASKS, VALIDATION, VIEWS,
    moving_anchors, validate_recording_split,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def poses(fk, q):
    """Batched equivalent of bridge UrdfKinematics.forward; returns XYZ+rotvec."""
    out = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for joint in fk.chain:
        out = out @ joint.origin
        if joint.kind == "fixed":
            continue
        if joint.kind not in ("revolute", "continuous"):
            raise ValueError(joint.kind)
        axis = joint.axis / np.linalg.norm(joint.axis)
        rot = Rotation.from_rotvec(q[:, fk.q_index[joint.name], None] * axis).as_matrix()
        mat = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        mat[:, :3, :3] = rot
        out = out @ mat
    for i in np.unique(np.linspace(0, len(q) - 1, 5, dtype=int)):
        np.testing.assert_allclose(out[i], fk.forward(q[i]), atol=1e-10)
    return np.concatenate([out[:, :3, 3], Rotation.from_matrix(out[:, :3, :3]).as_rotvec()], axis=1)


def cache_video(source, dest, count):
    import cv2
    from PIL import Image
    if dest.exists():
        arr = np.load(dest, mmap_mode="r")
        if arr.shape != (count, 270, 270, 3) or arr.dtype != np.uint8:
            raise ValueError(f"Invalid existing image cache {dest}")
        return
    required = count * 270 * 270 * 3
    if shutil.disk_usage(dest.parent).free < required + (10 << 30):
        raise OSError("Image caching would leave less than 10 GiB free")
    partial = dest.with_suffix(".partial.npy")
    cap = cv2.VideoCapture(str(source))
    arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(count, 270, 270, 3))
    try:
        for frame in range(count):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Truncated video: {source} frame {frame}")
            arr[frame] = np.asarray(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((270, 270), Image.Resampling.BILINEAR))
        if cap.read()[0]:
            raise ValueError(f"Video has more frames than parquet: {source}")
        arr.flush()
        del arr
        partial.replace(dest)
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bridge-root", type=Path, default=Path(__file__).resolve().parents[2] / "VLAPolicyBridge")
    p.add_argument("--download", action="store_true", help="Download pinned HF snapshot (~10.6 GB with videos)")
    p.add_argument("--numeric-only", action="store_true", help="Download/prepare metadata and parquet only")
    p.add_argument("--cache-images", action="store_true", help="270 RGB mmap cache, ~66 GiB for all 237 episodes")
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.numeric_only and args.cache_images:
        p.error("--numeric-only and --cache-images are mutually exclusive")
    if args.download:
        from huggingface_hub import snapshot_download
        patterns = ["meta/*", "data/*/*.parquet", "README.md"]
        if not args.numeric_only:
            patterns.append("videos/*/*/*.mp4")
        snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
                          local_dir=source, allow_patterns=patterns, max_workers=4)
    sys.path.insert(0, str(args.bridge_root.resolve()))
    from vla_policy_bridge.hil.kinematics import UrdfKinematics
    from vla_policy_bridge.twist2_tracker import TRACKER_JOINTS
    urdf = args.bridge_root / "vla_policy_bridge/robots/vendor_twist2/g1_29dof_rev_1_0.urdf"
    fks = [UrdfKinematics(urdf, target_link=f"{side}_rubber_hand", arm_slice=sl)
           for side, sl in (("left", slice(15, 22)), ("right", slice(22, 29)))]
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(x) for x in (source / "meta/episodes.jsonl").read_text().splitlines()]
    provenance = [json.loads(x) for x in (source / "meta/source_provenance.jsonl").read_text().splitlines()]
    notes = json.loads((source / "meta/source_notes.json").read_text())
    phases = {int(ep): int(task[4]) for ep, task in notes["task_types"]["by_episode_index"].items()}
    if info["fps"] != 60 or len(episodes) != 237 or set(phases) != set(range(237)):
        raise ValueError("This recipe requires the pinned 237-episode, 60-Hz dataset")
    for key in ("action", "observation.state"):
        if tuple(info["features"][key]["names"][:29]) != tuple(TRACKER_JOINTS):
            raise ValueError(f"Unexpected joint order: {key}")
    validate_recording_split(provenance, phases)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(format=FORMAT, source=str(source), repo=DATASET_REPO, revision=DATASET_REVISION,
                    urdf_sha256=sha256(urdf), preparation_sha256=sha256(__file__),
                    fps=60, horizon=30, state_dim=59, action_dim=32,
                    images="RGB resize270 -> random/center crop256 -> resize224",
                    pause_filter="train only: concatenated L/R XYZ end-to-start norm >= 1 mm",
                    tail="repeat final command; never cross episodes",
                    metadata_sha256={f: sha256(source / "meta" / f) for f in
                                     ("info.json", "episodes.jsonl", "source_provenance.jsonl", "source_notes.json")})
    contract_path = output / "preparation_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Preparation contract changed; use a new output directory")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    records = []
    for row in episodes:
        ep, n = int(row["episode_index"]), int(row["length"])
        phase = phases[ep]
        dest = output / "shared" / f"episode_{ep:06d}"
        dest.mkdir(parents=True, exist_ok=True)
        numeric = dest / "numeric.npz"
        parquet = source / "data/chunk-000" / f"episode_{ep:06d}.parquet"
        digest = sha256(parquet)
        receipt_path = dest / "receipt.json"
        if numeric.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["parquet_sha256"] != digest or receipt["numeric_sha256"] != sha256(numeric):
                raise ValueError(f"Changed input/cache for episode {ep}")
            kept = receipt["train_anchors"]
        else:
            d = pq.read_table(parquet).to_pydict()
            if len(d["action"]) != n or not np.array_equal(d["frame_index"], np.arange(n)):
                raise ValueError(f"Frame index mismatch in episode {ep}")
            if not np.all(np.asarray(d["task_index"]).reshape(-1) == 3 * (phase - 1)):
                raise ValueError(f"Task mismatch in episode {ep}")
            measured = np.asarray(d["observation.state"], np.float64)[:, :29]
            cmd = np.asarray(d["action"], np.float64)[:, :29]
            mp = [poses(fk, measured) for fk in fks]
            cp = [poses(fk, cmd) for fk in fks]
            np.testing.assert_allclose(cp[1][:, :3], d["action.right_ee_xyz"], atol=1e-6)
            left = np.asarray(d["action.hand_left"], np.float64)[:, 4:5]
            right = np.asarray(d["action.hand_right"], np.float64)[:, :5]
            state = np.concatenate([measured, *mp, *cp, left, right], axis=1).astype(np.float32)
            commands = np.concatenate([*cp, left, right, cmd[:, 15:29]], axis=1).astype(np.float32)
            if not np.isfinite(state).all() or not np.isfinite(commands).all():
                raise ValueError(f"Non-finite data in episode {ep}")
            anchors = moving_anchors(commands)
            np.savez(numeric, state=state, commands=commands, train_anchors=anchors)
            kept = len(anchors)
            receipt_path.write_text(json.dumps(dict(parquet_sha256=digest, numeric_sha256=sha256(numeric), train_anchors=kept)) + "\n")
        if args.cache_images:
            for view in VIEWS:
                path = source / "videos/chunk-000" / f"observation.images.{view}" / f"episode_{ep:06d}.mp4"
                cache_video(path, dest / f"{view}.npy", n)
        records.append(dict(episode=ep, phase=phase, frames=n, train_anchors=kept,
                            split="validation" if ep in VALIDATION[phase] else "train"))
        print(f"episode {ep:03d}: phase {phase}, {n} frames, {kept} moving anchors", flush=True)
    for phase in TASKS:
        dst = output / f"p{phase}"
        dst.mkdir(exist_ok=True)
        selected = [r for r in records if r["phase"] == phase]
        manifest = dict(format=FORMAT, phase=phase, source=str(source), shared_root="../shared",
                        source_revision=DATASET_REVISION, preparation_contract_sha256=sha256(contract_path),
                        episodes=selected, prompt=TASKS[phase])
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"phase {phase}: train={sum(r['split']=='train' for r in selected)}, "
              f"validation={sum(r['split']=='validation' for r in selected)}, "
              f"train anchors={sum(r['train_anchors'] for r in selected if r['split']=='train')}")
    print("Prepared numeric data. Images: " + ("cached" if args.cache_images else "on-demand videos; use --cache-images before training"))


if __name__ == "__main__":
    main()
