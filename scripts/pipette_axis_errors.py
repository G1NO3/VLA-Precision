#!/usr/bin/env python3
"""Per-axis held-out Cartesian delta-action errors, in mm per 30 Hz step."""
import csv
import json
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
RUN = WORKSPACE / 'outputs/pi05-lora-sft-20260909'
BASE = WORKSPACE / 'outputs/pi05-sft-20260909'

def metrics(predictions, targets, episode_ids):
    rows = []
    for ep in np.unique(episode_ids):
        error = (predictions[episode_ids == ep] - targets[episode_ids == ep]) * 1000
        rows.append({'episode': int(ep), 'mse_mm2': np.mean(error**2, axis=(0, 1)),
                     'mae_mm': np.mean(np.abs(error), axis=(0, 1)), 'bias_mm': np.mean(error, axis=(0, 1))})
    mse = np.mean([r['mse_mm2'] for r in rows], axis=0)
    mae = np.mean([r['mae_mm'] for r in rows], axis=0)
    bias = np.mean([r['bias_mm'] for r in rows], axis=0)
    return {'axes': {axis: {'rmse_mm': float(np.sqrt(mse[i])), 'mae_mm': float(mae[i]),
                            'bias_mm': float(bias[i])} for i, axis in enumerate('xyz')},
            'xyz_rmse_mm': float(np.sqrt(np.mean(mse))),
            'xy_component_rmse_mm': float(np.sqrt(np.mean(mse[:2]))),
            'xy_vector_rms_mm': float(np.sqrt(np.sum(mse[:2]))),
            'episodes': [{'episode': r['episode'],
                          'rmse_mm': dict(zip('xyz', np.sqrt(r['mse_mm2']).tolist()))} for r in rows]}

def main():
    before = json.loads((BASE / 'base-validation.predictions.json').read_text())
    after = json.loads((RUN / 'validation.predictions.json').read_text())
    assert [(r['episode'], r['frame'], r['target']) for r in before] == [(r['episode'], r['frame'], r['target']) for r in after]
    target = np.asarray([r['target'] for r in after], dtype=np.float64)
    episode_ids = np.asarray([r['episode'] for r in after])
    predictions = {'base': np.asarray([r['prediction'] for r in before], dtype=np.float64),
                   'sft': np.asarray([r['prediction'] for r in after], dtype=np.float64),
                   'zero': np.zeros_like(target)}
    result = {'units': 'mm per 1/30 second action step', 'coordinate_frame': 'pelvis',
              'aggregation': 'mean squared error within each episode, then equal-episode mean, then square root',
              'chunks': len(after), 'horizon': target.shape[1], 'episodes': len(np.unique(episode_ids)),
              'note': 'Errors of predicted Cartesian delta actions, not absolute hand/tip position or task success.',
              'all_chunk_actions': {}, 'first_action': {}}
    for name, predicted in predictions.items():
        result['all_chunk_actions'][name] = metrics(predicted, target, episode_ids)
        result['first_action'][name] = metrics(predicted[:, :1], target[:, :1], episode_ids)
    saved = json.loads((RUN / 'validation.json').read_text())
    np.testing.assert_allclose(result['all_chunk_actions']['sft']['xyz_rmse_mm'],
                               saved['episode_equal_chunk_rmse_mm'], rtol=1e-6)
    (RUN / 'axis-errors.json').write_text(json.dumps(result, indent=2) + '\n')
    with (RUN / 'axis-errors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['selection', 'model', 'axis', 'rmse_mm', 'mae_mm', 'bias_mm'])
        writer.writeheader()
        for selection in ('all_chunk_actions', 'first_action'):
            for name, values in result[selection].items():
                for axis, values in values['axes'].items():
                    writer.writerow({'selection': selection, 'model': name, 'axis': axis, **values})
    print(json.dumps({selection: {name: values['axes'] for name, values in result[selection].items()}
                      for selection in ('all_chunk_actions', 'first_action')}, indent=2))

if __name__ == '__main__':
    main()
