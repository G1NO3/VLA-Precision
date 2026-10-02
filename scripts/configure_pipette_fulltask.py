#!/usr/bin/env python3
"""Emit five self-contained JAX configs from verified prepared manifests."""
import argparse
import json
import math
from pathlib import Path

import yaml

from vla_precision.data.pipette_fulltask import FORMAT


def make_config(data_root, phase, *, batch_size, epochs, gpus, fsdp_devices,
                init_params, checkpoint_root, assets_root, cache_root,
                project, run_prefix):
    root = data_root / f"p{phase}"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != FORMAT or manifest["phase"] != phase:
        raise ValueError("Wrong data manifest")
    anchors = sum(r["train_anchors"] for r in manifest["episodes"] if r["split"] == "train")
    # Epoch-equivalent sample budget; round up to a whole optimizer update.
    steps = math.ceil(anchors * epochs / batch_size)
    epoch_steps = math.ceil(anchors / batch_size)
    return dict(
        schema_version=1, cuda_visible_devices=gpus, xla_memory_fraction=0.9,
        openpi=dict(name="pi05_full_finetune_pipette_fulltask", exp_name=f"{run_prefix}_p{phase}",
                    initialization_checkpoint=str(init_params),
                    model=dict(action_dim=32, action_horizon=30, max_token_len=320),
                    batch_size=batch_size, num_train_steps=steps, num_workers=0,
                    resume=False, overwrite=False, fsdp_devices=fsdp_devices,
                    project_name=project, ema_decay=None, seed=42,
                    lr_schedule=dict(warmup_steps=min(100, steps - 1), peak_lr=1e-5,
                                     decay_steps=steps, decay_lr=1e-6),
                    optimizer=dict(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10,
                                   clip_gradient_norm=1.0),
                    log_interval=10, save_interval=epoch_steps, keep_period=None,
                    wandb_enabled=True, eval_interval=epoch_steps, eval_samples=480,
                    eval_batch_size=4, eval_seed=42, num_inference_steps=10),
        data=dict(lerobot_repo_id=f"local/g1-pipette-fulltask-jax-v1-p{phase}",
                  lerobot_root=str(root), state_indices=[], action_indices=[],
                  state_key="observation.state", action_key="action", extra_delta_transform=False,
                  prompt_from_task=False,
                  image_key_map=dict(base_0_rgb="observation.images.rgb",
                                     left_wrist_0_rgb="observation.images.wrist_left")),
        paths=dict(checkpoint_root=str(checkpoint_root), openpi_assets_root=str(assets_root),
                   cache_root=str(cache_root)),
    )


def main():
    workspace = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-params", type=Path, default=workspace / "models/openpi/openpi-assets/checkpoints/pi05_base/params")
    p.add_argument("--checkpoint-root", type=Path, default=workspace / "checkpoints/pipette-fulltask-jax")
    p.add_argument("--assets-root", type=Path, default=workspace / "assets/pipette-fulltask-jax")
    p.add_argument("--cache-root", type=Path, default=workspace / ".cache/pipette-fulltask-jax")
    p.add_argument("--batch-size", type=int, default=128, help="GLOBAL batch, not per GPU")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--fsdp-devices", type=int, default=4)
    p.add_argument("--project", default="pipette-fulltask-pi05-jax")
    p.add_argument("--run-prefix", default="pi05_fulltask_v1")
    a = p.parse_args()
    devices = a.gpus.split(",")
    if a.batch_size < 1 or a.epochs <= 0 or a.fsdp_devices < 1 or len(devices) % a.fsdp_devices or a.batch_size % len(devices):
        p.error("Positive batch/epochs/FSDP required; batch and device counts must be divisible")
    if not a.init_params.is_dir():
        p.error("--init-params must point at the native OpenPI params/ directory, NOT .pt")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for phase in range(1, 6):
        cfg = make_config(a.data_root.resolve(), phase, batch_size=a.batch_size, epochs=a.epochs,
                          gpus=a.gpus, fsdp_devices=a.fsdp_devices, init_params=a.init_params.resolve(),
                          checkpoint_root=a.checkpoint_root.resolve(), assets_root=a.assets_root.resolve(),
                          cache_root=a.cache_root.resolve(), project=a.project, run_prefix=a.run_prefix)
        if cfg["openpi"]["num_train_steps"] < 2:
            p.error("Training budget must include at least two optimizer updates")
        path = a.output_dir / f"pi05_fulltask_p{phase}.yaml"
        payload = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite changed configuration: {path}")
        path.write_text(payload)
        print(f"{path}: {cfg['openpi']['num_train_steps']} steps, eval every {cfg['openpi']['eval_interval']}")


if __name__ == "__main__":
    main()
