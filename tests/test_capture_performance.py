"""CPU-only async capture, durability and exact feature parity; no robot I/O."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt import capture_writer as cw
from vla_precision.gr00t_rlt.bootstrap import load_observation
from vla_precision.gr00t_rlt.core import representation_sha
from vla_precision.gr00t_rlt.features import FrozenGR00TFeatures
from vla_precision.gr00t_rlt.model import spatial_features
from vla_precision.gr00t_rlt.replay import Replay
from test_direct_editor import recipe, editor_record, bare_policy


def features():
    mask = np.zeros(800, bool)
    mask[:512] = True
    return dict(prefix=np.ones((800, 2048), np.float16), mask=mask,
                state=np.zeros(59, np.float32), reference=np.zeros((30,32), np.float32),
                rgb=np.full((8,9,3),37,np.uint8), focus=np.full((4,4,3),13,np.uint8))


def row_with(evidence, c):
    row = editor_record(c)
    for prefix in ("", "next_"):
        row.update({prefix+k:np.asarray(v) for k,v in evidence.items()})
    return row


def test_nonblocking_snapshot_bounded_queue_and_durable_export(tmp_path, monkeypatch):
    c = recipe(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = cw.save_observation
    def blocked(*a, **kw):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test release")
        return original(*a, **kw)
    monkeypatch.setattr(cw, "save_observation", blocked)
    writer = cw.CaptureWriter(tmp_path, max_pending=1)
    db = Replay(tmp_path)
    try:
        f = features()
        f['_prefix_device'] = torch.ones(2)  # must not enter the CPU archive
        ev = writer.submit(f, c, "rollout")
        assert started.wait(2)
        f['rgb'][:] = 0  # queued evidence has an independent immutable copy
        assert writer.status()['pending'] == 1
        with pytest.raises(RuntimeError, match="queue full"):
            writer.submit(f,c,"rollout")
        row = row_with(ev,c)
        eid = db.begin(dict(phase='p1',contract_sha256=c['contract_sha256'],
                            token_sha256=representation_sha(c),behavior_sha256='test'))
        db.append(eid,row)
        db.finish(eid,'failure',1_000_000_001)
        with pytest.raises(ValueError,match='pending or failed'):
            db.export(c,representation_sha(c),tmp_path/'early.npz')
        assert not (tmp_path/'early.npz').exists()
    finally:
        release.set()
        writer.close()
    saved = load_observation(tmp_path,row,c)
    assert np.all(saved['rgb']==37) and '_prefix_device' not in saved
    assert db.export(c,representation_sha(c),tmp_path/'ready.npz')['transitions']==1
    with (tmp_path/ev['observation_file']).open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError,match='hash'):
        db.export(c,representation_sha(c),tmp_path/'corrupted.npz')
    db.close()


def test_writer_failure_latches_and_never_issues_receipt(tmp_path, monkeypatch):
    def fail(*a,**kw):
        raise OSError('disk full')
    monkeypatch.setattr(cw,'save_observation',fail)
    writer=cw.CaptureWriter(tmp_path)
    ev=writer.submit(features(),recipe(tmp_path),'rollout')
    with pytest.raises(RuntimeError,match='writer failed'):
        writer.close()
    assert not (tmp_path/ev['observation_receipt']).exists()
    with pytest.raises(RuntimeError,match='writer failed'):
        writer.submit(features(),recipe(tmp_path),'next')


def test_receipt_path_and_identity_are_checked(tmp_path):
    c=recipe(tmp_path);writer=cw.CaptureWriter(tmp_path)
    ev=writer.submit(features(),c,'r');writer.close();row=row_with(ev,c)
    row['observation_receipt']=np.asarray('../../escape')
    with pytest.raises(ValueError,match='receipt path'):
        load_observation(tmp_path,row,c)
    row=row_with(ev,c)
    p=tmp_path/ev['observation_receipt'];receipt=json.loads(p.read_text())
    receipt['contract_sha256']='wrong';p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt mismatch'):
        load_observation(tmp_path,row,c)


def test_device_prefix_matches_legacy_fp16_roundtrip_exactly():
    e=FrozenGR00TFeatures.__new__(FrozenGR00TFeatures)
    e.c=dict(views=['rgb'],token=dict(input_dim=2048,prefix_seq_len=1024))
    e.proc=SimpleNamespace(data_config=SimpleNamespace(serve_view_preprocess=lambda x:x),apply_state=lambda x:x)
    prefix=torch.randn(1,800,2048)
    mask=torch.zeros(1,800,dtype=torch.bool);mask[:,:512]=True
    e._predict_capture=lambda example:(np.zeros((30,32),np.float32),dict(prefix=prefix,mask=mask))
    e.keep_device_features=True
    f=e.extract({'observation.state':np.zeros(59),'observation.images.rgb':np.zeros((4,4,3),np.uint8),'task':'test'})
    legacy=torch.from_numpy(prefix[0].float().numpy().astype(np.float16)).float()[None]
    torch.testing.assert_close(f['_prefix_device'].float()[None],legacy,rtol=0,atol=0)
    torch.testing.assert_close(spatial_features(f['_prefix_device'].float()[None],mask),
                               spatial_features(legacy,mask),rtol=0,atol=0)
    assert all(v>=0 for v in e.last_timings.values())


def test_live_action_returns_while_archive_is_blocked_and_reports_timings(tmp_path,monkeypatch):
    from vla_policy_bridge.pi05_fulltask import synthetic_observation
    c=recipe(tmp_path);p=bare_policy(c);p._load_actor()
    p.episode=None;p.capture_root=tmp_path;p.stats=dict(q01=[-1]*32,q99=[1]*32)
    f=features()
    f['_prefix_device']=torch.from_numpy(f['prefix'])
    f['_mask_device']=torch.from_numpy(f['mask'])
    p.features=SimpleNamespace(extract=lambda sample:f,
        physical_actions=lambda a,r:np.concatenate([a,r[:,18:]],axis=-1))
    entered,release=threading.Event(),threading.Event()
    original=cw.save_observation
    def blocked(*a,**kw):
        entered.set()
        if not release.wait(5):raise TimeoutError('test release')
        return original(*a,**kw)
    monkeypatch.setattr(cw,'save_observation',blocked)
    p.capture_writer=cw.CaptureWriter(tmp_path)
    try:
        result=p.get_action(synthetic_observation(),{'episode':'r'})
        assert entered.wait(2)
        meta=result['info']['rlt']
        assert 'observation_receipt' in meta
        assert not (tmp_path/meta['observation_receipt']).exists()
        assert meta['capture_writer']['pending']==1
        assert meta['timing_ms']['server_total_ms']>=meta['timing_ms']['actor_ms']>=0
        np.testing.assert_array_equal(meta['action'],f['reference'][:,:18])
    finally:
        release.set();p.close()
