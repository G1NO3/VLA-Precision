"""Read-only pipette inference for the operator GUI; no robot command writer."""
from __future__ import annotations

import hashlib
import json
import socket
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np

VIEWS = ("rgb", "wrist_left", "wrist_right")


def read_live_packet(host: str, port: int) -> bytes:
    """The only Redis operation supported by this observer is GET."""
    key = b"hil:observation"
    request = b"*2\r\n$3\r\nGET\r\n$" + str(len(key)).encode() + b"\r\n" + key + b"\r\n"
    with socket.create_connection((host, port), timeout=1.0) as sock:
        sock.sendall(request)
        with sock.makefile('rb') as stream:
            line = stream.readline(128)
            if line == b'$-1\r\n':
                raise ValueError('No live HIL observation. Preview does not start the actor or robot.')
            if not line.startswith(b'$'):
                raise ValueError('Redis did not return an observation')
            size = int(line[1:])
            if not 0 < size <= 16 * 1024 * 1024:
                raise ValueError('Invalid observation size')
            data = stream.read(size)
            if len(data) != size or stream.read(2) != b'\r\n':
                raise ValueError('Incomplete live observation')
            return data


def decode_live_packet(raw: bytes, *, now_ms: int | None = None):
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    packet = msgpack.unpackb(raw, raw=False)
    if not isinstance(packet, dict) or packet.get('schema_version') != 1:
        raise ValueError('Unsupported HIL observation schema')
    if packet.get('action_frame') != 'pelvis':
        raise ValueError('Expected pelvis-frame observation')
    if packet.get('action_units') != 'metres_per_10hz_step':
        raise ValueError('Unexpected HIL wire units; this adapter reads the current 10 Hz wire')
    if packet.get('image_color_after_decode') != 'BGR':
        raise ValueError('Unknown JPEG color convention')
    stamp = packet.get('timestamp_ms')
    if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise ValueError('Missing observation timestamp')
    age = now_ms - stamp
    if age < -100 or age > 1000:
        raise ValueError(f'Stale live observation ({age:.0f} ms)')
    seq = packet.get('observation_seq')
    if type(seq) is not int or seq < 0:
        raise ValueError('Invalid observation sequence')
    fields = packet.get('state', {})
    joints = np.asarray(fields.get('joint_position', []), np.float32)
    xyz = np.asarray(fields.get('eef_position', []), np.float32)
    if joints.shape != (29,) or xyz.shape != (3,):
        raise ValueError('Expected measured joints (29) and measured hand XYZ (3)')
    state = np.concatenate((joints, xyz))
    if not np.isfinite(state).all():
        raise ValueError('Non-finite measured state')
    observation = {'observation.state': state}
    for view in VIEWS:
        encoded = packet.get('images_jpeg', {}).get(view)
        if not isinstance(encoded, bytes):
            raise ValueError(f'Missing live camera: {view}')
        bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or min(bgr.shape[:2]) < 8:
            raise ValueError(f'Invalid live JPEG: {view}')
        observation[f'observation.images.{view}'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return observation, {'source': 'live_hil', 'observation_seq': seq,
                         'timestamp_ms': int(stamp), 'age_at_receive_ms': age,
                         'episode': packet.get('episode_id'), 'frame': packet.get('step_index'),
                         'source_action_hz': 10}


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


class PreviewModel:
    def __init__(self, config_path: Path, checkpoint: Path):
        from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat
        install_lerobot_import_compat()
        from openpi import transforms
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import normalize
        from vla_precision.config import load_stage1_config
        from vla_precision.data.pipette import PipetteDataset
        from vla_precision.integrations.openpi.configs import build_stage1_train_config

        self.resolved = load_stage1_config(config_path)
        self.config = build_stage1_train_config(self.resolved.config)
        c = self.config
        if c.name != 'pi05_lora_finetune_pipette' or not c.model.discrete_state_input:
            raise ValueError('Preview requires the pipette pi05 LoRA configuration with discrete state')
        if c.model.action_dim != 32 or c.model.action_horizon != 10:
            raise ValueError('Expected the trained 32-D, 10-step action contract')
        self.datasets = [PipetteDataset(self.resolved.config.data.lerobot_root, 10, split=split)
                         for split in ('train', 'validation')]
        self.samples = {(ep, frame): (dataset, i) for dataset in self.datasets
                        for i, (ep, frame) in enumerate(dataset.samples)}
        self.manifest = self.datasets[0].manifest
        dc = c.data.create(c.assets_dirs, c.model)
        stats_path = checkpoint / 'assets' / dc.asset_id / 'norm_stats.json'
        expected = c.assets_dirs / dc.repo_id / 'norm_stats.json'
        if not (checkpoint / 'params/_METADATA').is_file():
            raise ValueError('Checkpoint must contain native Orbax params/')
        if stats_path.read_bytes() != expected.read_bytes():
            raise ValueError('Checkpoint normalization does not match the selected training configuration')
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            key: value for key, value in dc.repack_transforms.inputs[0].structure.items() if key != 'actions'
        })])
        self.policy = create_trained_policy(c, checkpoint, repack_transforms=repack,
                                           norm_stats=normalize.load(stats_path.parent),
                                           sample_kwargs={'num_steps': 10})
        self.metadata = {
            'checkpoint': str(checkpoint.resolve()), 'config': str(config_path.resolve()),
            'config_sha256': self.resolved.config_sha256,
            'norm_stats_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            'model': c.name, 'action_hz': 30, 'action_horizon': 10, 'flow_steps': 10, 'frame': 'pelvis',
            'dataset_root': str(self.datasets[0].root.resolve()),
            'source_revision': self.manifest['source_revision'],
            'state_source': self.manifest['state_source'],
            'state_names': self.manifest['state'], 'prompt': self.manifest['prompt'],
            'mode': 'preview_only', 'robot_output': False,
            'episodes': [{'episode': ep, 'split': ds.split,
                          'max_frame': len(ds.episodes[ep]['actions']) - 10}
                         for ds in self.datasets for ep in ds.episodes],
        }

    def infer(self, request: dict, run_dir: Path):
        from openpi_client.image_tools import resize_with_pad
        source = request.get('source')
        target = None
        if source == 'dataset':
            ep, frame = int(request.get('episode', 0)), int(request.get('frame', 0))
            if (ep, frame) not in self.samples:
                raise ValueError('Episode/frame does not have a complete 10-step chunk')
            dataset, index = self.samples[ep, frame]
            observation = dataset[index]
            target = observation.pop('action')
            meta = {'source': 'dataset', 'episode': ep, 'frame': frame, 'split': dataset.split,
                    'source_frame_60hz': int(dataset.episodes[ep]['source_frame'][frame])}
        elif source == 'live_hil':
            observation, meta = decode_live_packet(read_live_packet(
                str(request.get('redis_host', '127.0.0.1')), int(request.get('redis_port', 6379))))
            observation['task'] = self.manifest['prompt']
        else:
            raise ValueError('Source must be dataset or live_hil')
        # Display and record the exact RGB pixels passed to native preprocessing.
        for view in VIEWS:
            key = f'observation.images.{view}'
            observation[key] = resize_with_pad(observation[key], 224, 224)
        seed = int(request.get('seed', 42))
        noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        start = time.perf_counter()
        actions = np.asarray(self.policy.infer(observation, noise=noise)['actions'], np.float32)
        elapsed = (time.perf_counter() - start) * 1000
        if actions.shape != (10, 3) or not np.isfinite(actions).all():
            raise ValueError('Model produced invalid XYZ actions')
        folder = run_dir / request['id']
        folder.mkdir()
        inputs = {'state': observation['observation.state'],
                  **{v: observation[f'observation.images.{v}'] for v in VIEWS}}
        np.savez_compressed(folder / 'inputs.npz', **inputs)
        for view in VIEWS:
            cv2.imwrite(str(folder / f'{view}.png'), cv2.cvtColor(inputs[view], cv2.COLOR_RGB2BGR))
        result = {'id': request['id'], 'source': meta, 'timestamp_ms': int(time.time() * 1000),
                  'inference_ms': elapsed, 'seed': seed, 'action_hz': 30, 'robot_output': False,
                  'actions_m': actions.tolist(), 'actions_mm': (actions * 1000).tolist(),
                  'state': inputs['state'].tolist(), 'target_mm': None}
        if target is not None:
            result['target_mm'] = (target * 1000).tolist()
            result['axis_rmse_mm'] = np.sqrt(np.mean(((actions - target) * 1000)**2, axis=0)).tolist()
        if source == 'live_hil':
            result['source_age_at_result_ms'] = result['timestamp_ms'] - meta['timestamp_ms']
        write_json(folder / 'result.json', result)
        return result
