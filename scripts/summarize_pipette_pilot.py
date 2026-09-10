#!/usr/bin/env python3
"""Summarize the completed pilot using recorded results, never inferred success."""
import argparse
import csv
import json
from pathlib import Path
import re
import math
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('../outputs/pi05-lora-sft-20260909'))
    parser.add_argument('--base', type=Path, default=Path('../outputs/pi05-sft-20260909/base-validation.json'))
    args = parser.parse_args()
    if (args.run / 'status.txt').read_text().strip() != 'complete':
        raise SystemExit('Pilot has not completed training and validation')
    config = yaml.safe_load((args.run / 'resolved.yaml').read_text())
    steps = config['openpi']['num_train_steps']
    batch_size = config['openpi']['batch_size']
    base = json.loads(args.base.read_text())
    result = json.loads((args.run / 'validation.json').read_text())
    for key in ('norm_stats_sha256', 'seed', 'flow_steps', 'samples_per_episode'):
        if base[key] != result[key]:
            raise ValueError(f'Incomparable validation settings: {key}')
    # Verify exact episode/frame selection and targets, not just sample counts.
    before = json.loads(args.base.with_suffix('.predictions.json').read_text())
    after = json.loads((args.run / 'validation.predictions.json').read_text())
    if [(x['episode'], x['frame'], x['target']) for x in before] != [(x['episode'], x['frame'], x['target']) for x in after]:
        raise ValueError('Before/after validation targets differ')
    log = (args.run / 'train.log').read_text()
    metrics = [dict(step=int(m[1]), grad_norm=float(m[2]), loss=float(m[3]), param_norm=float(m[4]))
               for m in re.finditer(r'Step (\d+): grad_norm=([\d.e+-]+), loss=([\d.e+-]+), param_norm=([\d.e+-]+)', log)]
    if not metrics or any(x in log for x in ('loss=nan', 'grad_norm=nan', 'Traceback')):
        raise ValueError('Training log is incomplete or contains errors')
    with (args.run / 'gpu-usage.csv').open() as f:
        memory = [float(row['memory_used_MiB']) for row in csv.DictReader(f)]
    key = 'episode_equal_chunk_rmse_mm'
    summary = {
        'status': 'completed', 'method': 'native JAX pi05 LoRA plus vision/projection SFT',
        'optimizer_steps': steps, 'batch_size': batch_size, 'checkpoint': result['checkpoint'],
        'validation_chunks': len(after), 'validation_episodes': len(result['episodes']),
        'base_chunk_rmse_mm': base[key], 'sft_chunk_rmse_mm': result[key],
        'zero_action_chunk_rmse_mm': result['episode_equal_zero_action_rmse_mm'],
        'relative_rmse_reduction_percent': (1 - result[key] / base[key]) * 100,
        'base_first_action_rmse_mm': base['episode_equal_first_action_rmse_mm'],
        'sft_first_action_rmse_mm': result['episode_equal_first_action_rmse_mm'],
        'sampled_peak_gpu_used_GiB': max(memory) / 1024,
        'first_logged_loss': metrics[0]['loss'], 'last_logged_loss': metrics[-1]['loss'],
        'note': '144-chunk held-out pilot subset; open-loop action prediction, not attachment success.',
    }
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.run / 'training-metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    lines = ['# Native JAX pi05 pipette LoRA pilot', '',
             f'Completed {steps:,} optimizer updates, batch {batch_size}. Discrete state input; three views; fixed 37/9 episode split.', '',
             '| Held-out metric (mm/step) | Base | SFT |', '| --- | ---: | ---: |',
             f'| Chunk XYZ RMSE | {base[key]:.6f} | {result[key]:.6f} |',
             f'| First-action XYZ RMSE | {base["episode_equal_first_action_rmse_mm"]:.6f} | {result["episode_equal_first_action_rmse_mm"]:.6f} |', '',
             f'Zero-action chunk RMSE: {summary["zero_action_chunk_rmse_mm"]:.6f} mm/step.',
             f'Sampled peak GPU memory used: {summary["sampled_peak_gpu_used_GiB"]:.2f} GiB.', '',
             'Both policies used exactly the same 144 chunks, target arrays, noise seed, normalization and flow steps.',
             'This is a pilot subset, not a full validation sweep or a robot success-rate evaluation.', '',
             '| Episode | Base chunk RMSE | SFT chunk RMSE | Zero-action RMSE |', '| --- | ---: | ---: | ---: |']
    base_episodes = {x['episode']: x for x in base['episodes']}
    for row in result['episodes']:
        old = base_episodes[row['episode']]
        values = [math.sqrt(x) * 1000 for x in (old['chunk_mse_m2'], row['chunk_mse_m2'], row['zero_action_mse_m2'])]
        lines.append(f'| {row["episode"]} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    lines.extend(['', f'Checkpoint: `{result["checkpoint"]}`', '',
                  'Load this checkpoint with `configs/stage1/pipette_pi05_lora.yaml`; it contains LoRA parameters.', ''])
    (args.run / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
