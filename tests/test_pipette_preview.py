"""Preview input semantics and recordings; no GPU or robot required."""
import json
import socket
import threading
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import pytest

from vla_precision.integrations.openpi.pipette_preview import (
    VIEWS, PreviewModel, decode_live_packet, read_live_packet,
)


def packet():
    ok, jpeg = cv2.imencode('.jpg', np.full((16, 24, 3), [10, 80, 230], np.uint8))
    assert ok
    return {'schema_version': 1, 'action_frame': 'pelvis',
            'action_units': 'metres_per_10hz_step', 'image_color_after_decode': 'BGR',
            'timestamp_ms': 10000, 'observation_seq': 7,
            'state': {'joint_position': list(range(29)), 'eef_position': [.1, .2, .3]},
            'images_jpeg': {v: jpeg.tobytes() for v in VIEWS}}


def test_live_measured_state_and_rgb():
    obs, meta = decode_live_packet(msgpack.packb(packet()), now_ms=10200)
    np.testing.assert_allclose(obs['observation.state'], [*range(29), .1, .2, .3])
    for view in VIEWS:
        np.testing.assert_allclose(obs[f'observation.images.{view}'][0, 0], [230, 80, 10], atol=3)
    assert meta['age_at_receive_ms'] == 200
    assert meta['source_action_hz'] == 10


@pytest.mark.parametrize('change', [
    lambda p: p.update(timestamp_ms=8000),
    lambda p: p.update(timestamp_ms=12000),
    lambda p: p.update(action_frame='world'),
    lambda p: p.update(schema_version=2),
    lambda p: p.update(action_units='millimetres'),
    lambda p: p.update(image_color_after_decode='RGB'),
    lambda p: p['state'].update(joint_position=[0]*34),
    lambda p: p['state'].update(eef_position=[float('nan'), 0, 0]),
    lambda p: p['images_jpeg'].pop('wrist_right'),
    lambda p: p['images_jpeg'].update(rgb=b'invalid'),
])
def test_bad_live_packets_rejected(change):
    p = packet()
    change(p)
    with pytest.raises((ValueError, cv2.error)):
        decode_live_packet(msgpack.packb(p), now_ms=10200)


def test_redis_is_get_only():
    seen = []
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        def respond():
            with server.accept()[0] as client:
                seen.append(client.recv(4096))
                client.sendall(b'$3\r\nabc\r\n')
        thread = threading.Thread(target=respond)
        thread.start()
        assert read_live_packet(*server.getsockname()) == b'abc'
        thread.join(timeout=3)
    assert seen == [b'*2\r\n$3\r\nGET\r\n$15\r\nhil:observation\r\n']


def test_prediction_records_exact_inputs_and_mm(tmp_path):
    obs = {'observation.state': np.arange(32, dtype=np.float32), 'task': 'attach',
           'action': np.zeros((10, 3), np.float32),
           **{f'observation.images.{v}': np.full((100, 200, 3), i*60, np.uint8) for i, v in enumerate(VIEWS)}}
    class Dataset:
        split = 'validation'
        episodes = {9: {'source_frame': [12]}}
        def __getitem__(self, i):
            return dict(obs)
    def infer(observation, *, noise):
        assert 'action' not in observation
        assert noise.shape == (10, 32)
        assert all(observation[f'observation.images.{v}'].shape == (224, 224, 3) for v in VIEWS)
        return {'actions': np.tile(np.array([.001, -.002, .003], np.float32), (10, 1))}
    model = PreviewModel.__new__(PreviewModel)
    model.samples = {(9, 0): (Dataset(), 0)}
    model.policy = SimpleNamespace(infer=infer)
    result = model.infer({'id': '0123456789abcdef', 'source': 'dataset', 'episode': 9, 'frame': 0}, tmp_path)
    assert result['action_hz'] == 30 and result['robot_output'] is False
    np.testing.assert_allclose(result['actions_mm'][0], [1, -2, 3])
    np.testing.assert_allclose(result['axis_rmse_mm'], [1, 2, 3])
    folder = tmp_path / result['id']
    assert json.loads((folder / 'result.json').read_text()) == result
    with np.load(folder / 'inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['state'], obs['observation.state'])
        for v in VIEWS:
            png = cv2.cvtColor(cv2.imread(str(folder / f'{v}.png')), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(png, inputs[v])


def test_native_30hz_live_packet():
    p=packet();p.update(action_hz=30,action_units='metres_per_30hz_step')
    _,meta=decode_live_packet(msgpack.packb(p),now_ms=10200)
    assert meta['source_action_hz']==30
    p['action_hz']=10
    with pytest.raises(ValueError):decode_live_packet(msgpack.packb(p),now_ms=10200)
