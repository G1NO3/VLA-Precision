"""Fixed holdout evaluation and best-weights-only Orbax snapshots for pi0.5."""
from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from openpi import transforms
from openpi.models import model as model_api
from openpi.shared import normalize
from openpi.training import checkpoints
import orbax.checkpoint as ocp
from scipy.spatial.transform import Rotation

from vla_precision.data.pipette_fulltask import FullTaskDataset
from vla_precision.integrations.openpi.data_loader import transform_dataset


def physical_metrics(pred, target, baseline):
    out = {}
    for suffix, value in (("", pred), ("_hold", baseline)):
        for side, start in (("left", 0), ("right", 6)):
            delta = value[..., start:start + 3] - target[..., start:start + 3]
            out[f"{side}_xyz_rmse_mm{suffix}"] = float(np.sqrt(np.mean(delta ** 2)) * 1000)
            r1 = Rotation.from_rotvec(value[..., start + 3:start + 6].reshape(-1, 3))
            r2 = Rotation.from_rotvec(target[..., start + 3:start + 6].reshape(-1, 3))
            out[f"{side}_rotation_rms_deg{suffix}"] = float(np.rad2deg(np.sqrt(np.mean((r1.inv() * r2).magnitude() ** 2))))
        out[f"left_thumb_rmse_registers{suffix}"] = float(np.sqrt(np.mean((value[..., 12] - target[..., 12]) ** 2)))
        out[f"right_hand_rmse_registers{suffix}"] = float(np.sqrt(np.mean((value[..., 13:18] - target[..., 13:18]) ** 2)))
        out[f"arm_joints_rmse_deg{suffix}"] = float(np.rad2deg(np.sqrt(np.mean((value[..., 18:] - target[..., 18:]) ** 2))))
    return out


class FullTaskEvaluator:
    def __init__(self, config, root_config):
        self.config = config
        self.options = root_config.openpi
        if self.options.eval_samples < 1 or self.options.eval_batch_size < 1 or self.options.num_inference_steps < 1:
            raise ValueError("Invalid holdout evaluation settings")
        if self.options.eval_samples % self.options.eval_batch_size:
            raise ValueError("eval_samples must be divisible by eval_batch_size")
        dataset = FullTaskDataset(root_config.data.lerobot_root, config.model.action_horizon, split="validation")
        self.dc = config.data.create(config.assets_dirs, config.model)
        processed = transform_dataset(dataset, self.dc, training=False)
        indices = dataset.validation_indices(self.options.eval_samples, self.options.eval_seed)
        self.root = Path(config.checkpoint_dir)
        selection = {"seed": self.options.eval_seed,
                     "manifest_sha256": hashlib.sha256((dataset.root / "manifest.json").read_bytes()).hexdigest(),
                     "norm_stats_sha256": hashlib.sha256((Path(config.assets_dirs) / self.dc.asset_id / "norm_stats.json").read_bytes()).hexdigest(),
                     "samples": [dict(episode=dataset.samples[i][0], frame=dataset.samples[i][1]) for i in indices],
                     "selection": "episode-balanced; not the starVLA 480-sample subset"}
        path = self.root / "holdout_samples.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError("Holdout selection changed on resume")
        path.write_text(json.dumps(selection, indent=2) + "\n")
        self.batches, self.targets, self.raw_targets = [], [], []
        for start in range(0, len(indices), self.options.eval_batch_size):
            ids = indices[start:start + self.options.eval_batch_size]
            batch = jax.tree.map(lambda *xs: np.stack(xs), *[processed[i] for i in ids])
            self.targets.append(batch.pop("actions"))
            self.raw_targets.append(np.stack([dataset.numeric_item(i)["actions"] for i in ids]))
            self.batches.append(model_api.Observation.from_dict(batch))
        self.targets = np.concatenate(self.targets)
        self.raw_targets = np.concatenate(self.raw_targets)
        self.baseline = np.broadcast_to(self.raw_targets[:, :1], self.raw_targets.shape)
        self.best_score = float("inf")
        self.best_path = self.root / "best_selection.json"
        if self.best_path.exists():
            self.best_score = float(json.loads(self.best_path.read_text())["mse"])
        self.manager = ocp.CheckpointManager(
            self.root / "best",
            item_handlers={"params": ocp.PyTreeCheckpointHandler(), "assets": checkpoints.CallbackHandler()},
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True,
                                                async_options=ocp.AsyncOptions(timeout_secs=7200)),
        )

        def predict(params, graphdef, obs, rng):
            model = nnx.merge(graphdef, params)
            model.eval()
            return model.sample_actions(rng, obs, num_steps=self.options.num_inference_steps)

        self.predict = jax.jit(predict, static_argnums=(1,))

    def evaluate(self, state, step):
        params = state.ema_params if state.ema_params is not None else state.params
        preds = []
        for i, batch in enumerate(self.batches):
            # Same flow noise on every evaluation; never consume training RNG.
            rng = jax.random.fold_in(jax.random.key(self.options.eval_seed), i)
            preds.append(np.asarray(self.predict(params, state.model_def, batch, rng)))
        pred = np.concatenate(preds)
        if not np.isfinite(pred).all():
            raise FloatingPointError("Non-finite holdout predictions")
        mse = float(np.mean((pred - self.targets) ** 2))
        hold_mse = float(np.mean((self.targets[:, :1] - self.targets) ** 2))
        raw = transforms.Unnormalize({"actions": self.dc.norm_stats["actions"]},
                                     use_quantiles=self.dc.use_quantile_norm)({"actions": pred})["actions"]
        metrics = {"eval/mse": mse, "eval/hold_mse": hold_mse,
                   "eval/mse_ratio": mse / max(hold_mse, 1e-12),
                   **{f"eval/{k}": v for k, v in physical_metrics(raw, self.raw_targets, self.baseline).items()}}
        with (self.root / "holdout_metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if mse < self.best_score:
            def save_assets(directory):
                normalize.save(directory / self.dc.asset_id, self.dc.norm_stats)
            self.manager.save(step, {"params": {"params": params}, "assets": save_assets})
            self.manager.wait_until_finished()
            self.best_score = mse
            self.best_path.write_text(json.dumps({"step": step, "mse": mse,
                "checkpoint": f"best/{step}", "note": "inference weights + train stats; resume from latest full-state checkpoint"}, indent=2) + "\n")
        return metrics

    def close(self):
        self.manager.close()
