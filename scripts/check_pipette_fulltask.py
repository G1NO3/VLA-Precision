#!/usr/bin/env python3
"""CPU preflight of labels, normalization, token lengths and model input shapes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from vla_precision.config import load_stage1_config
from vla_precision.data.pipette_fulltask import FullTaskDataset, TASKS, VIEWS
from vla_precision.integrations.openpi.configs import build_stage1_train_config
from vla_precision.integrations.openpi.data_loader import transform_dataset


def check_model_shapes(cfg):
    import jax
    from flax import nnx, traverse_util
    import orbax.checkpoint as ocp
    from openpi.shared import array_typing as at
    shape = jax.eval_shape(lambda key: nnx.state(cfg.model.create(key)), jax.random.key(0))
    flat = traverse_util.flatten_dict(shape.to_pure_dict())
    meta = ocp.PyTreeCheckpointer().metadata(cfg.weight_loader.params_path)
    base = traverse_util.flatten_dict(meta["params"])
    if set(flat) != set(base) or any(flat[k].shape != base[k].shape for k in flat):
        raise ValueError("Model parameters do not match the native base checkpoint")
    obs, actions = cfg.model.inputs_spec(batch_size=1)
    with at.disable_typechecking():
        obs = obs.replace(state=jax.ShapeDtypeStruct((1, 59), np.float32))
    loss = jax.eval_shape(lambda rng, o, a: cfg.model.create(rng).compute_loss(rng, o, a, train=True),
                          jax.random.key(0), obs, actions)
    if loss.shape != (1, 30):
        raise ValueError(f"Unexpected loss shape {loss.shape}")
    return dict(parameter_leaves=len(flat), missing=0, mismatched_shapes=0,
                loss_shape=list(loss.shape), verification="abstract forward; no model arrays or optimizer allocated")


def check(path, require_cached=False, images=False, model_shapes=False):
    resolved = load_stage1_config(path)
    cfg = build_stage1_train_config(resolved.config)
    if cfg.name != "pi05_full_finetune_pipette_fulltask":
        raise ValueError("Not a full-task JAX recipe")
    train = FullTaskDataset(resolved.config.data.lerobot_root)
    val = FullTaskDataset(resolved.config.data.lerobot_root, split="validation")
    if set(train.episodes) & set(val.episodes):
        raise ValueError("Episode leakage")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if dc.norm_stats is None or dc.norm_stats["state"].mean.shape != (59,) or dc.norm_stats["actions"].mean.shape != (32,):
        raise ValueError("Run norm-stats first: expected 59-state / 32-action train statistics")
    stats_root = Path(cfg.assets_dirs) / dc.asset_id
    receipt = json.loads((stats_root / "norm_contract.json").read_text())
    if (receipt["manifest_sha256"] != hashlib.sha256((train.root / "manifest.json").read_bytes()).hexdigest()
            or receipt["norm_stats_sha256"] != hashlib.sha256((stats_root / "norm_stats.json").read_bytes()).hexdigest()
            or receipt["chunks"] != len(train) or receipt["split"] != "train" or receipt["horizon"] != 30):
        raise ValueError("Normalization contract mismatch; recompute stats for this phase")
    if not Path(resolved.config.openpi.initialization_checkpoint).is_dir():
        raise ValueError("Missing native JAX initialization params")
    missing = []
    for dataset in (train, val):
        for ep, numeric in dataset.episodes.items():
            for v in VIEWS:
                image_path = dataset.shared / f"episode_{ep:06d}" / f"{v}.npy"
                if not image_path.is_file():
                    missing.append(str(image_path))
                    continue
                image = np.load(image_path, mmap_mode="r")
                if image.shape != (len(numeric["state"]), 270, 270, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid image cache {image_path}")
    if require_cached and missing:
        raise ValueError(f"{len(missing)} missing image caches; rerun prepare with --download --cache-images")
    # Numeric coverage throughout each recording, using synthetic images by
    # default so a parquet-only download is sufficient for this preflight.
    class Samples:
        def __len__(self):
            return len(train)
        def __getitem__(self, index):
            if images:
                return train[index]
            item = train.numeric_item(index)
            return {"observation.state": item["state"], "action": item["actions"],
                    "task": TASKS[train.phase],
                    **{f"observation.images.{v}": np.zeros((270, 270, 3), np.uint8) for v in VIEWS}}
    transformed = transform_dataset(Samples(), dc, training=False)
    token_lengths = []
    for i in np.linspace(0, len(train) - 1, 128, dtype=int):
        batch = transformed[int(i)]
        assert batch["state"].shape == (59,) and batch["actions"].shape == (30, 32)
        assert batch["image_mask"]["right_wrist_0_rgb"] == False
        assert all(x.shape == (224, 224, 3) for x in batch["image"].values())
        assert np.isfinite(batch["state"]).all() and np.isfinite(batch["actions"]).all()
        length = int(batch["tokenized_prompt_mask"].sum())
        if length >= cfg.model.max_token_len:
            raise ValueError("Potential state/prompt truncation")
        token_lengths.append(length)
    hand_filtered = 0
    for data in train.episodes.values():
        cmd = data["commands"]
        end = np.minimum(np.arange(len(cmd)) + 29, len(cmd) - 1)
        hand_moves = np.any(np.abs(cmd[end, 12:18] - cmd[:, 12:18]) > 1, axis=1)
        keep = np.zeros(len(cmd), bool)
        keep[data["train_anchors"]] = True
        hand_filtered += int(np.sum(hand_moves & ~keep))
    result = dict(config=str(path), phase=train.phase, train_episodes=len(train.episodes),
                  validation_episodes=len(val.episodes), train_anchors=len(train),
                  validation_anchors=len(val), holdout_samples=len(val.validation_indices()),
                  max_sampled_token_length=max(token_lengths), token_capacity=cfg.model.max_token_len,
                  filtered_hand_motion_anchors=hand_filtered, missing_image_caches=len(missing),
                  image_check="decoded real videos" if images else "synthetic images; numeric labels are real",
                  train_stats_only=True, global_batch=cfg.batch_size, fsdp_devices=cfg.fsdp_devices,
                  train_steps=cfg.num_train_steps,
                  free_disk_gib=round(shutil.disk_usage(train.root).free / 2**30, 1))
    if model_shapes:
        result["model_check"] = check_model_shapes(cfg)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--require-cached", action="store_true")
    p.add_argument("--images", action="store_true", help="Decode actual images (slower without cache)")
    p.add_argument("--model-shapes", action="store_true", help="Abstract full-model forward and base-weight compatibility")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = check(a.config, a.require_cached, a.images, a.model_shapes)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
