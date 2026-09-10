#!/usr/bin/env python3
"""Check real data splits, chunk boundaries, camera mapping and discrete state."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from openpi import transforms
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi_client.image_tools import resize_with_pad
from vla_precision.config import load_stage1_config
from vla_precision.data.pipette import PipetteDataset, VIEWS, VALIDATION_EPISODES
from vla_precision.integrations.openpi.configs import build_stage1_train_config
from vla_precision.integrations.openpi.data_loader import create_data_loader, transform_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/stage1/pipette_pi05.yaml'))
    p.add_argument('--output', type=Path, default=Path('../outputs/pi05-sft-20260909/data-check.json'))
    args = p.parse_args()
    root = load_stage1_config(args.config).config
    c = build_stage1_train_config(root)
    dc = c.data.create(c.assets_dirs, c.model)
    assert c.model.pi05 and c.model.discrete_state_input and dc.use_quantile_norm
    train = PipetteDataset(root.data.lerobot_root, c.model.action_horizon)
    val = PipetteDataset(root.data.lerobot_root, c.model.action_horizon, split='validation')
    expected_validation = set(train.manifest.get('validation_episodes', VALIDATION_EPISODES))
    expected_episodes = {episode['episode'] for episode in train.manifest['episodes']}
    assert set(val.episodes) == expected_validation
    assert set(train.episodes) == expected_episodes - expected_validation
    assert not set(train.episodes) & set(val.episodes)
    max_tokens = 0
    tokenizer = PaligemmaTokenizer(1024)
    normalize = transforms.Normalize(dc.norm_stats, use_quantiles=True)
    for dataset in (train, val):
        for ep, numeric in dataset.episodes.items():
            assert numeric['state'].shape[1] == 32 and numeric['actions'].shape[1] == 3
            assert np.isfinite(numeric['state']).all() and np.isfinite(numeric['actions']).all()
            frames, next_frames = numeric['source_frame'], numeric['source_next_frame']
            assert np.array_equal(next_frames[:-1], frames[1:])
            # Check every state's full token length, including out-of-quantile states.
            for state in numeric['state']:
                state = normalize({'state': state})['state']
                _, mask = tokenizer.tokenize(dataset.manifest['prompt'], state)
                max_tokens = max(max_tokens, int(mask.sum()))
            for view in VIEWS:
                cache = dataset._image_array(ep, view)
                assert cache.shape == (len(frames), 224, 224, 3) and cache.dtype == np.uint8
        for i, (ep, frame) in enumerate(dataset.samples):
            end = frame + c.model.action_horizon
            assert end <= len(dataset.episodes[ep]['actions'])
            np.testing.assert_array_equal(dataset.numeric_item(i)['actions'], dataset.episodes[ep]['actions'][frame:end])
    assert max_tokens <= c.model.max_token_len, (max_tokens, c.model.max_token_len)
    # Independently decode first/middle/last camera frames in one train and one val episode.
    camera_checks = 0
    camera_episodes = [(train, min(train.episodes)), (val, min(val.episodes))]
    # The HIL194 union has a second recording session starting at episode 46.
    if len(expected_episodes) == 194:
        camera_episodes += [(ds, min(ep for ep in ds.episodes if ep >= 46)) for ds in (train, val)]
    for dataset, ep in camera_episodes:
        numeric = dataset.episodes[ep]
        for view in VIEWS:
            path = Path(dataset.manifest['source']) / f'videos/chunk-000/observation.images.{view}/episode_{ep:06d}.mp4'
            cap = cv2.VideoCapture(str(path))
            for index in (0, len(numeric['actions']) // 2, len(numeric['actions']) - 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(numeric['source_frame'][index]))
                ok, frame = cap.read()
                assert ok
                rgb = resize_with_pad(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 224, 224)
                np.testing.assert_array_equal(rgb, dataset._image_array(ep, view)[index])
                camera_checks += 1
            cap.release()
    sample = transform_dataset(train, dc)[0]
    assert sample['actions'].shape == (c.model.action_horizon, 32)
    np.testing.assert_array_equal(sample['actions'][:, 3:], 0)
    normalized = normalize(train.numeric_item(0))
    restored = transforms.Unnormalize(dc.norm_stats, use_quantiles=True)(normalized)
    np.testing.assert_allclose(restored['actions'], train.numeric_item(0)['actions'], atol=1e-8)
    batch = next(iter(create_data_loader(c, root, shuffle=False, num_batches=1)))
    assert batch[1].shape == (c.batch_size, c.model.action_horizon, 32)
    report = {'train_episodes': sorted(train.episodes), 'validation_episodes': sorted(val.episodes),
              'train_chunks': len(train), 'validation_chunks': len(val), 'max_prompt_tokens': max_tokens,
              'camera_frame_checks_exact': camera_checks, 'discrete_state_input': True,
              'quantile_normalization': True, 'action_shape': list(batch[1].shape), 'status': 'passed'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
