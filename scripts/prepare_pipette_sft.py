#!/usr/bin/env python3
"""Build an auditable native-OpenPI cache from the corrected HF snapshot.

Read-only inputs; resumable episode caches. Existing completed output must have
identical input hashes. Images use the exact OpenPI PIL bilinear resize+pad.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pyarrow.parquet as pq
from openpi_client.image_tools import resize_with_pad
from vla_precision.data.pipette import FORMAT, VIEWS, VALIDATION_EPISODES

WORKSPACE = Path(__file__).resolve().parents[2]
REVISION = "e74439f3eb2257661145571571b2dfaa5592f734"

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def cache_video(source, output, frames):
    target = np.lib.format.open_memmap(str(output) + '.partial', mode='w+', dtype=np.uint8,
                                     shape=(len(frames), 224, 224, 3))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {source}')
    wanted = 0
    try:
        for frame in range(int(frames[-1]) + 1):
            if not cap.grab():
                raise RuntimeError(f'Truncated video {source}: frame {frame}')
            if frame == frames[wanted]:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise RuntimeError(f'Cannot decode {source}: frame {frame}')
                target[wanted] = resize_with_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 224, 224)
                wanted += 1
        assert wanted == len(frames)
        target.flush()
        del target
        Path(str(output) + '.partial').replace(output)
    finally:
        cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    w = args.workspace.resolve()
    source = w / 'datasets/g1-pipette-3view-hgdagger-20260904-e74439f3eb22'
    labels = w / 'datasets/g1-pipette-3view-hgdagger-20260904-expert-xyz-30hz/actions'
    output = args.output or w / 'datasets/g1-pipette-pi05-30hz-e74439f3eb22'
    output.mkdir(parents=True, exist_ok=True)
    fk_path = w / 'VLAPolicyBridge/scripts/dataset/prepare_pipette_hil_candidates.py'
    spec = importlib.util.spec_from_file_location('pipette_fk_source', fk_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    urdf = w / 'TWIST2/assets/g1/g1_29dof_rev_1_0.urdf'
    fk = mod.UrdfFk(urdf, 'right_rubber_hand')
    assert fk.base_link == 'pelvis'
    manifest = {
        'format': FORMAT, 'source': str(source), 'source_revision': REVISION,
        'prompt': 'Attach a pipette tip to the pipette, lift it, then move to the disposal box and eject the tip.',
        'scope': 'whole recorded episodes; not attachment-only cropped demonstrations',
        'fps': 30, 'action_units': 'metres per 1/30 second step in pelvis frame',
        'action': 'FK(action[t+2,:29]) - FK(action[t,:29]) at audited source frame pairs',
        'state': list(mod.TRACKER_JOINTS) + ['right_hand_x', 'right_hand_y', 'right_hand_z'],
        'state_source': 'observation.state[:29] and FK thereof; no future information',
        'urdf': str(urdf), 'urdf_sha256': sha256(urdf), 'fk_sha256': sha256(fk_path),
        'images': 'source camera names already repaired; RGB uint8, OpenPI PIL resize+pad 224',
        'chunk_boundary': 'only full horizon windows of real transitions; no terminal padding',
        'validation_episodes': list(VALIDATION_EPISODES), 'episodes': [],
    }
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for ep in range(46):
            dest = output / f'episode_{ep:06d}'
            dest.mkdir(exist_ok=True)
            numeric_file = source / f'data/chunk-000/episode_{ep:06d}.parquet'
            label_file = labels / f'episode_{ep:06d}.parquet'
            videos = {v: source / f'videos/chunk-000/observation.images.{v}/episode_{ep:06d}.mp4' for v in VIEWS}
            provenance = {str(p): sha256(p) for p in [numeric_file, label_file, urdf, fk_path, *videos.values()]}
            record_path = dest / 'record.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['input_sha256'] != provenance:
                    raise ValueError(f'Input changed for episode {ep}; use a fresh output directory')
                for name, digest in record['output_sha256'].items():
                    if sha256(dest / name) != digest:
                        raise ValueError(f'Cache corrupted: {dest / name}')
                manifest['episodes'].append(record)
                print(f'episode {ep:02d} verified cached', flush=True)
                continue
            raw = pq.read_table(numeric_file).to_pydict()
            label = pq.read_table(label_file).to_pydict()
            frames = np.asarray(label['source_frame'], dtype=np.int64)
            next_frames = np.asarray(label['source_next_frame'], dtype=np.int64)
            assert np.all(next_frames - frames == 2) and np.all(np.diff(frames) == 2)
            assert np.array_equal(frames[1:], next_frames[:-1])
            reference = np.asarray(raw['action'], dtype=np.float64)[:, :29]
            positions, _ = fk.poses(reference[np.r_[frames, next_frames[-1]]])
            actions = np.asarray(label['expert_delta_xyz'], dtype=np.float64)
            np.testing.assert_array_equal(np.diff(positions, axis=0).astype(np.float32), actions.astype(np.float32))
            joints = np.asarray(raw['observation.state'], dtype=np.float64)[frames, :29]
            measured, _ = fk.poses(joints)
            state = np.concatenate((joints, measured), axis=-1).astype(np.float32)
            assert state.shape == (len(frames), 32)
            assert np.isfinite(state).all() and np.isfinite(actions).all()
            np.savez(dest / 'numeric.npz', state=state, actions=actions.astype(np.float32),
                     source_frame=frames, source_next_frame=next_frames)
            futures = [pool.submit(cache_video, path, dest / f'{v}.npy', frames) for v, path in videos.items()]
            for f in futures:
                f.result()
            record = {'episode': ep, 'split': 'validation' if ep in VALIDATION_EPISODES else 'train',
                      'transitions': len(frames), 'input_sha256': provenance,
                      'output_sha256': {p.name: sha256(p) for p in [dest / 'numeric.npz', *[dest / f'{v}.npy' for v in VIEWS]]}}
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            manifest['episodes'].append(record)
            print(f'episode {ep:02d}: {len(frames)} transitions ({record["split"]})', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {sum(e["transitions"] for e in manifest["episodes"])} transitions in {output}', flush=True)

if __name__ == '__main__':
    main()
