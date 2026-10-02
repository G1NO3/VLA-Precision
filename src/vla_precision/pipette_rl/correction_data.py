"""Supervised human corrections; autonomous actions never become targets."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'VLA-Precision/configs/pipette_rl/human_bc_20260914_native.json'


def masked_xyz_mse(prediction,target,mask):
    """Works with NumPy or JAX; no loss on padding or non-XYZ channels."""
    error=((prediction[...,:3]-target[...,:3])**2).mean(-1)
    return (error*mask).sum()/mask.sum().clip(1)


def masked_native_mse(prediction,target,mask):
    """Native pi05 includes the 29 zero-padded channels in flow denoising."""
    return (((prediction-target)**2).mean(-1)*mask).sum()/mask.sum().clip(1)


def load_config(path=DEFAULT):
    c=json.loads(Path(path).read_text())
    c['config_sha256']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    for key in ('baseline','replay_db','output'):
        c[key]=str((ROOT/c[key]).resolve())
    if 'context_cache' in c:c['context_cache']=str((ROOT/c['context_cache']).resolve())
    if 'context_cache' in c:
        prior=json.loads((Path(c['context_cache']).parent/'manifest.json').read_text())['config']
        if (prior['config_sha256']!=c['context_config_sha256'] or
                any(prior[k]!=c[k] for k in ('baseline','replay_db','norm_sha256'))):
            raise ValueError('Reused context cache belongs to another baseline or replay')
    stats=list((Path(c['baseline'])/'assets').rglob('norm_stats.json'))
    if len(stats)!=1 or hashlib.sha256(stats[0].read_bytes()).hexdigest()!=c['norm_sha256']:
        raise ValueError('Original HIL293 normalization does not match')
    c['norm_stats']=str(stats[0])
    return c


def read_db(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    return db


def human_frame(row, info, c):
    return (bool(row['intervention']) and info.get('source')=='human'
            and info.get('training_bc_eligible',False)
            and info.get('training_intervention',True)
            and not info.get('release_debounce_active',False)
            and info.get('safety_reason')=='ok'
            and info.get('rl_contract')==c['source_contract']
            and info.get('action_hz')==30)


def build_manifest(c):
    zone=ZoneInfo(c['timezone'])
    start=dt.datetime.fromisoformat(c['start_local']).replace(tzinfo=zone).timestamp()*1e9
    end=dt.datetime.fromisoformat(c['end_local']).replace(tzinfo=zone).timestamp()*1e9
    counts=collections.Counter();chunks=[];outcomes={};valid_frame_ids=set()
    with read_db(c['replay_db']) as db:
        discarded={r[0] for r in db.execute('select episode_id from discarded_episodes')}
        episodes=[r[0] for r in db.execute("select distinct episode_id from transitions where buffer='online' and timestamp_ns>=? and timestamp_ns<?",(start,end))]
        for ep in episodes:
            rows=db.execute("select * from transitions where buffer='online' and episode_id=? order by step_index",(ep,)).fetchall()
            if (ep in discarded or not rows[-1]['terminated'] or rows[-1]['truncated']
                    or [r['step_index'] for r in rows]!=list(range(len(rows)))
                    or any(r['terminated'] or r['truncated'] for r in rows[:-1])):
                counts['excluded_episodes']+=1;continue
            groups=[];group=[]
            for row in rows:
                info=json.loads(row['info_json'])
                human=start<=row['timestamp_ns']<end and human_frame(row,info,c)
                counts['autonomous_frames']+=info.get('source')=='policy'
                counts['release_frames']+=info.get('source')=='release_hold'
                counts['human_frames']+=bool(row['intervention'])
                if group and (not human or not 0<(row['timestamp_ns']-group[-1][0]['timestamp_ns'])/1e9<=c['max_gap_s']):
                    groups.append(group);group=[]
                if human:
                    a=np.asarray(info['commanded_action'],np.float32)
                    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Invalid human target')
                    active=np.linalg.norm(a)*1000>=c['minimum_action_mm'] or info.get('intentional_hold',False)
                    counts['human_waiting_excluded']+=not active
                    group.append((row,a,active))
                    if active:valid_frame_ids.add(row['id'])
            if group:groups.append(group)
            before=len(chunks)
            for group in groups:
                i=0
                while i<len(group):
                    if not group[i][2]:i+=1;continue
                    window=group[i:i+c['horizon']]
                    actions=np.zeros((c['horizon'],3),np.float32)
                    mask=np.zeros(c['horizon'],bool);ids=[None]*c['horizon']
                    for k,(row,a,active) in enumerate(window):
                        if active:actions[k]=a;mask[k]=True;ids[k]=row['id']
                    chunks.append(dict(episode=ep,observation_id=window[0][0]['observation_id'],
                        step=window[0][0]['step_index'],timestamp_ns=window[0][0]['timestamp_ns'],
                        actions_m=actions.tolist(),loss_mask=mask.tolist(),human_transition_ids=ids))
                    i+=c['horizon']
            if len(chunks)>before:outcomes[ep]='success' if rows[-1]['reward']>0 else 'failure'
    # An active human target appears exactly once, with no inferred/pseudo label.
    used=[i for x in chunks for i in x['human_transition_ids'] if i is not None]
    if len(used)!=len(set(used)) or set(used)!=valid_frame_ids:raise ValueError('Human target coverage mismatch')
    if len(outcomes)<5:raise ValueError('Need at least five labelled correction episodes for episode-held-out validation')
    rng=np.random.default_rng(c['seed']);val=set()
    for outcome in ('success','failure'):
        eps=sorted(ep for ep,v in outcomes.items() if v==outcome)
        rng.shuffle(eps)
        if len(eps)>1:val.update(eps[:max(1,round(len(eps)*c['validation_fraction']))])
    for x in chunks:x['split']='validation' if x['episode'] in val else 'train'
    counts.update(active_human_targets=len(used),chunks=len(chunks),episodes=len(outcomes),
                  train_chunks=sum(x['split']=='train' for x in chunks),validation_chunks=sum(x['split']=='validation' for x in chunks))
    return dict(config=c,counts=dict(counts),outcomes=outcomes,validation_episodes=sorted(val),chunks=chunks)
