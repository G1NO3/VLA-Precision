#!/usr/bin/env python3
"""Fixed-seed held-out native JAX action RMSE, with per-episode results.

This measures open-loop prediction, not attachment success. Defaults to 16
uniformly spaced complete chunks per validation episode for a pilot check.
Use --samples-per-episode 0 for every complete validation chunk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat
install_lerobot_import_compat()
from openpi import transforms
from openpi.shared import normalize
from openpi.policies.policy_config import create_trained_policy
from vla_precision.config import load_stage1_config
from vla_precision.data.pipette import PipetteDataset
from vla_precision.integrations.openpi.configs import build_stage1_train_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/stage1/pipette_pi05.yaml'))
    p.add_argument('--checkpoint', type=Path, required=True, help='Native checkpoint directory containing params/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples-per-episode', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    resolved = load_stage1_config(args.config)
    c = build_stage1_train_config(resolved.config)
    dc = c.data.create(c.assets_dirs, c.model)
    dataset = PipetteDataset(resolved.config.data.lerobot_root, c.model.action_horizon, split='validation')
    # Use checkpoint assets when present and reject a different data contract.
    # The official base has no pipette assets, so it uses the train-only stats.
    stats_path = c.assets_dirs / dc.repo_id / 'norm_stats.json'
    checkpoint_stats = args.checkpoint / 'assets' / dc.asset_id / 'norm_stats.json'
    if checkpoint_stats.is_file():
        if checkpoint_stats.read_bytes() != stats_path.read_bytes():
            raise ValueError('Checkpoint normalization differs from the configured training assets')
        stats_path = checkpoint_stats
    norm_stats = normalize.load(stats_path.parent)
    inference_repack = transforms.Group(inputs=[transforms.RepackTransform({
        key: value for key, value in dc.repack_transforms.inputs[0].structure.items() if key != 'actions'
    })])
    policy = create_trained_policy(c, args.checkpoint, repack_transforms=inference_repack,
                                   norm_stats=norm_stats, sample_kwargs={'num_steps': 10})
    rng = np.random.default_rng(args.seed)
    rows, predictions = [], []
    started = time.monotonic()
    for ep in sorted(dataset.episodes):
        indices = [i for i, (sample_ep, _) in enumerate(dataset.samples) if sample_ep == ep]
        if args.samples_per_episode > 0:
            selected = np.unique(np.linspace(0, len(indices) - 1, min(len(indices), args.samples_per_episode), dtype=int))
            indices = [indices[i] for i in selected]
        errors, first_errors, zeros = [], [], []
        for index in indices:
            obs = dataset[index]
            target = obs.pop('action')
            noise = rng.standard_normal((c.model.action_horizon, c.model.action_dim)).astype(np.float32)
            predicted = np.asarray(policy.infer(obs, noise=noise)['actions'])
            if not np.isfinite(predicted).all():
                raise FloatingPointError('Non-finite policy actions')
            errors.append((predicted - target)**2)
            first_errors.append((predicted[0] - target[0])**2)
            zeros.append(target**2)
            predictions.append({'episode': ep, 'frame': dataset.samples[index][1],
                                'prediction': predicted.tolist(), 'target': target.tolist()})
        row = {'episode': ep, 'chunks': len(indices), 'chunk_mse_m2': float(np.mean(errors)),
               'first_action_mse_m2': float(np.mean(first_errors)), 'zero_action_mse_m2': float(np.mean(zeros))}
        rows.append(row)
        print(f'episode {ep:02d}: chunk RMSE {np.sqrt(row["chunk_mse_m2"]) * 1000:.4f} mm/step', flush=True)
    result = {'checkpoint': str(args.checkpoint.resolve()), 'config_sha256': resolved.config_sha256,
              'norm_stats_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
              'seed': args.seed, 'flow_steps': 10, 'samples_per_episode': args.samples_per_episode,
              'episode_equal_chunk_rmse_mm': float(np.sqrt(np.mean([r['chunk_mse_m2'] for r in rows])) * 1000),
              'episode_equal_first_action_rmse_mm': float(np.sqrt(np.mean([r['first_action_mse_m2'] for r in rows])) * 1000),
              'episode_equal_zero_action_rmse_mm': float(np.sqrt(np.mean([r['zero_action_mse_m2'] for r in rows])) * 1000),
              'episodes': rows, 'elapsed_seconds': time.monotonic() - started,
              'note': 'Open-loop validation; not task success. Pilot subset unless samples_per_episode=0.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.with_suffix('.predictions.json').write_text(json.dumps(predictions) + '\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
