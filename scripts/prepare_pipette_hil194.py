#!/usr/bin/env python3
"""Prepare pinned HIL194 as native pipette cache; preserve all reference labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

REVISION = '643ece1b2e93f15022c83832302848c9978d5f05'
PROMPT = 'Attach a green pipette tip from the tip rack to the pipette.'
WORKSPACE = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def poses(fk, q):
    t = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for joint in fk.chain:
        t = t @ joint.origin
        if joint.kind == 'fixed':
            continue
        if joint.kind not in ('revolute', 'continuous'):
            raise ValueError(joint.kind)
        axis = joint.axis / np.linalg.norm(joint.axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        theta = q[:, fk.q_index[joint.name]]
        rot = np.eye(3) + np.sin(theta)[:, None, None] * cross + (1-np.cos(theta))[:, None, None] * (cross @ cross)
        r = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        r[:, :3, :3] = rot
        t = t @ r
    return t[:, :3, 3]


def cache_video(source, target, frames, expected_frames):
    temp = target.with_suffix('.npy.partial')
    data = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint8, shape=(len(frames), 224, 224, 3))
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
            raise ValueError(f'Invalid video: {source}')
        wanted = 0
        for frame in range(int(frames[-1]) + 1):
            if not cap.grab():
                raise ValueError(f'Truncated video: {source}, frame {frame}')
            if frame == frames[wanted]:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise ValueError(f'Decode failed: {source}, frame {frame}')
                data[wanted] = resize_with_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 224, 224)
                wanted += 1
        assert wanted == len(frames)
        data.flush()
        del data
        temp.replace(target)
        return sha256(target)
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    receipt = json.loads(args.receipt.read_text())
    if receipt['dataset_revision'] != REVISION or not receipt['all_verified'] or Path(receipt['root']).resolve() != source:
        raise ValueError('Expected verified pinned HIL194 download receipt')
    files = {r['path']: r for r in receipt['files']}
    fk_path = WORKSPACE / 'VLAPolicyBridge/scripts/dataset/prepare_pipette_hil_candidates.py'
    spec = importlib.util.spec_from_file_location('pipette_fk_hil194', fk_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    urdf = WORKSPACE / 'TWIST2/assets/g1/g1_29dof_rev_1_0.urdf'
    fk = mod.UrdfFk(urdf, 'right_rubber_hand')
    info = json.loads((source / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (source / 'meta/episodes.jsonl').read_text().splitlines()]
    assert info['fps'] == 60 and len(episodes) == 194 and fk.base_link == 'pelvis'
    for key in ['action', 'observation.state']:
        assert info['features'][key]['names'][:29] == list(mod.TRACKER_JOINTS)
    validation = sorted([*VALIDATION_EPISODES, *map(int, np.random.default_rng(42).choice(np.arange(46,194), 30, replace=False))])
    contract = dict(format=FORMAT, source=str(source), source_revision=REVISION, prompt=PROMPT,
                    fps=30, source_fps=60, resampling='even source frames; real (f,f+2) pairs',
                    state=list(mod.TRACKER_JOINTS)+['right_hand_x','right_hand_y','right_hand_z'],
                    state_source='measured observation.state[:29] plus its FK; no hand feedback',
                    action='FK(reference_action[f+2,:29])-FK(reference_action[f,:29])',
                    action_units='metres per 1/30 second step in pelvis frame',
                    action_outliers='retained unchanged; audit stored separately',
                    urdf=str(urdf), urdf_sha256=sha256(urdf), fk_sha256=sha256(fk_path),
                    preparation_sha256=sha256(__file__), validation_episodes=validation,
                    images='RGB uint8, native OpenPI PIL resize+pad 224; source names already corrected',
                    scope='whole recorded episodes; no success/phase filtering',
                    chunk_boundary='complete real transitions only; no terminal padding')
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / 'preparation_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Preparation contract changed; use a fresh output directory')
    contract_path.write_text(json.dumps(contract, indent=2)+'\n')
    records, futures = {}, {}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for episode in episodes:
            ep = episode['episode_index']
            dest = output / f'episode_{ep:06d}'
            dest.mkdir(exist_ok=True)
            numeric_rel = f'data/chunk-000/episode_{ep:06d}.parquet'
            video_rels = {v:f'videos/chunk-000/observation.images.{v}/episode_{ep:06d}.mp4' for v in VIEWS}
            inputs = {r:files[r]['sha256'] for r in [numeric_rel, *video_rels.values()]}
            record_path = dest / 'record.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['input_sha256'] != inputs:
                    raise ValueError(f'Input changed for {ep}')
                for name, digest in record['output_sha256'].items():
                    if sha256(dest/name) != digest:
                        raise ValueError(f'Corrupt cache {ep}/{name}')
                records[ep] = record
                print(f'episode {ep}: verified existing cache', flush=True)
                continue
            if sha256(source/numeric_rel) != inputs[numeric_rel]:
                raise ValueError(f'Corrupt numeric input {ep}')
            raw = pq.read_table(source/numeric_rel, columns=['action','observation.state','frame_index']).to_pydict()
            reference = np.asarray(raw['action'], np.float64)[:, :29]
            measured = np.asarray(raw['observation.state'], np.float64)[:, :29]
            n = len(reference)
            assert n == episode['length'] and np.array_equal(raw['frame_index'], np.arange(n))
            sequence = np.arange(0,n,2)
            reference_xyz = poses(fk, reference[sequence])
            measured_xyz = poses(fk, measured[sequence[:-1]])
            select = np.linspace(0, len(sequence)-1, 5, dtype=int)
            np.testing.assert_allclose(reference_xyz[select], fk.poses(reference[sequence[select]])[0], atol=1e-12, rtol=1e-12)
            actions = np.diff(reference_xyz, axis=0).astype(np.float32)
            state = np.concatenate([measured[sequence[:-1]], measured_xyz],axis=1).astype(np.float32)
            assert state.shape == (len(actions),32) and np.isfinite(state).all() and np.isfinite(actions).all()
            np.savez(dest/'numeric.npz', state=state, actions=actions, source_frame=sequence[:-1], source_next_frame=sequence[1:])
            records[ep] = dict(episode=ep, split='validation' if ep in validation else 'train', transitions=len(actions),
                               input_sha256=inputs, output_sha256={'numeric.npz':sha256(dest/'numeric.npz')})
            for view, rel in video_rels.items():
                if (source/rel).stat().st_size != files[rel]['bytes']:
                    raise ValueError(f'Changed video {rel}')
                future = pool.submit(cache_video, source/rel, dest/f'{view}.npy', sequence[:-1],n)
                futures[future] = (ep, view)
        for i, future in enumerate(as_completed(futures), 1):
            ep, view = futures[future]
            records[ep]['output_sha256'][view+'.npy'] = future.result()
            if len(records[ep]['output_sha256']) == 4:
                path = output/f'episode_{ep:06d}/record.json'
                path.write_text(json.dumps(records[ep],indent=2)+'\n')
            if i%12 == 0 or i == len(futures):
                print(f'cached {i}/{len(futures)} videos',flush=True)
    manifest = {**contract, 'episodes':[records[i] for i in sorted(records)]}
    temp = output/'manifest.json.partial'
    temp.write_text(json.dumps(manifest,indent=2)+'\n')
    temp.replace(output/'manifest.json')
    print(f'COMPLETE {len(records)} episodes, {sum(x["transitions"] for x in records.values())} transitions',flush=True)

if __name__ == '__main__':
    main()
