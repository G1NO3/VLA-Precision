import json
import sqlite3
from pathlib import Path

import numpy as np

from vla_precision.pipette_rl.correction_data import build_manifest


def test_loss_ignores_unlabelled_steps_and_extra_channels():
    from vla_precision.pipette_rl.correction_data import masked_xyz_mse
    p=np.ones((2,10,32));target=np.zeros_like(p);mask=np.zeros((2,10));mask[:,0]=1
    assert masked_xyz_mse(p,target,mask)==1
    target[:,1:,:]=999;target[:,0,3:]=-999
    assert masked_xyz_mse(p,target,mask)==1
    assert masked_xyz_mse(p,target,mask*0)==0


def test_native_loss_supervises_padding_only_on_labelled_human_steps():
    from vla_precision.pipette_rl.correction_data import masked_native_mse
    p=np.ones((1,10,32));target=np.zeros_like(p);mask=np.zeros((1,10));mask[:,0]=1
    assert masked_native_mse(p,target,mask)==1
    target[:,1:,:]=999
    assert masked_native_mse(p,target,mask)==1
    target[:,0,3:]=1
    assert masked_native_mse(p,target,mask)==3/32


def test_only_human_actions_supervised_without_crossing_takeover(tmp_path):
    db=sqlite3.connect(tmp_path/'replay.sqlite3')
    db.execute('create table discarded_episodes (episode_id)')
    db.execute('create table transitions (id,buffer,episode_id,step_index,observation_id,timestamp_ns,intervention,info_json,terminated,truncated,reward)')
    wanted=set()
    # Include autonomous outliers adjacent to corrections, a centered R1
    # waiting frame, release debounce and discarded/unfinished episodes.
    stamp=1789420000000000000
    for ep in range(7):
        for step in range(16):
            human=2<=step<=11;source='human' if human else 'policy'
            eligible=True
            if step==12:source='release_hold';eligible=False
            a=[0,0,-.0005] if human and step!=6 else ([0,0,0] if human else [99,99,99])
            info=dict(source=source,training_bc_eligible=eligible,training_intervention=human,
                      safety_reason='ok',rl_contract='ours',action_hz=30,commanded_action=a)
            tid=ep*100+step
            db.execute('insert into transitions values (?,?,?,?,?,?,?,?,?,?,?)',
                       (tid,'online',f'ep{ep}',step,tid,stamp+ep*1000000000+step*33333333,human,json.dumps(info),step==15 and ep!=6,False,1))
            if human and step!=6 and ep<5:wanted.add(tid)
    db.execute("insert into discarded_episodes values ('ep5')");db.commit();db.close()
    c=dict(replay_db=str(tmp_path/'replay.sqlite3'),timezone='America/New_York',start_local='2026-09-14T00:00:00',
           end_local='2026-09-15T06:00:00',source_contract='ours',max_gap_s=.15,minimum_action_mm=.01,horizon=10,
           seed=42,validation_fraction=.2)
    m=build_manifest(c)
    ids=[i for x in m['chunks'] for i in x['human_transition_ids'] if i is not None]
    assert len(ids)==len(wanted) and set(ids)==wanted
    for x in m['chunks']:
        a=np.asarray(x['actions_m']);mask=np.asarray(x['loss_mask'])
        assert abs(a).max()<.001
        assert np.all(a[~mask]==0)
        assert x['step']==2
    train={x['episode'] for x in m['chunks'] if x['split']=='train'}
    val={x['episode'] for x in m['chunks'] if x['split']=='validation'}
    assert train and val and not train.intersection(val)
