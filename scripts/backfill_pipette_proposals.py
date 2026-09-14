#!/usr/bin/env python3
"""Annotate missing human-vs-BC pairs from saved observations; no robot I/O.

Default is a read-only coverage check. --apply loads the pinned HIL293 BC,
never a robot controller, and adds proposal rows without changing transitions.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from vla_precision.pipette_rl.config import load_config,DEFAULT_CONFIG,WORKSPACE
sys.path.insert(0,str(WORKSPACE/'VLAPolicyBridge'))
sys.path.insert(0,str(WORKSPACE/'VLAPolicyBridge/scripts'))
from vla_policy_bridge.hil.policy_proposals import connect_writer,lookup,offline_key,write_proposal
from vla_policy_bridge.hil.replay import _decode_observation
from vla_precision.pipette_rl.replay import Replay


def competing_processes():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if p.parent.name==str(os.getpid()):continue
        try:
            args=p.read_bytes().split(b'\0')
            names=[Path(x.decode(errors='replace')).name for x in args[:4] if len(x)<300]
            if ('run_pipette_pi05_policy.py' in names
                    or ('backfill_pipette_proposals.py' in names and b'--apply' in args)
                    or ('pipette_rl.py' in names and (b'train' in args or b'smoke' in args))):
                result.append(int(p.parent.name))
        except OSError:pass
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--source-contract',type=Path)
    p.add_argument('--replay-db',type=Path)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--episode',action='append')
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    c=load_config(args.config,source_contract=args.source_contract,replay_db=args.replay_db)
    replay=Replay(c);replay.refresh();db=replay.connect()
    if db is None:raise ValueError('Replay does not exist')
    missing=[];human=direct=paired=0
    for idx,ep,is_human in replay.index:
        if not is_human or (args.episode and ep not in args.episode):continue
        row=db.execute('SELECT t.step_index,t.info_json,o.payload FROM transitions t JOIN observations o ON o.id=t.observation_id WHERE t.id=?',(idx,)).fetchone()
        step,info,blob=row;info=json.loads(info);human+=1
        if info.get('policy_valid'):direct+=1;continue
        if lookup(db,info=info,contract=c['contract_sha256'],episode=ep,step=step,replay_blob=blob):paired+=1;continue
        missing.append((idx,ep,step))
    print(json.dumps(dict(robot_output=False,human=human,direct=direct,paired=paired,missing=len(missing),episodes=list(replay.closed)),ensure_ascii=False),flush=True)
    if not args.apply or not missing:db.close();return
    busy=competing_processes()
    if busy:raise RuntimeError(f'Stop policy/learner/backfill processes first; GPU in use by PIDs {busy}')
    from pi05_standalone_policy import StandalonePi05Policy
    from vla_policy_bridge.hil.pi05_policy import adapt_actions
    base=Path(c['baseline']);model=StandalonePi05Policy(base.parents[1],step=base.name)
    writer=connect_writer(c['replay_db']);written=0;start=time.monotonic()
    provenance={**model.metadata,'source':'offline_baseline_counterfactual','seed':42,'chunk_index':0,
                'norm_sha256':c['norm_sha256'],'gain':1.,'max_step_mm':c['max_step_mm']}
    try:
        for idx,ep,step in missing[:args.limit]:
            busy=competing_processes()
            if busy:raise RuntimeError(f'Concurrent GPU process detected: {busy}; stopping backfill')
            if ep in Replay.discarded_ids(db):continue
            blob=db.execute('SELECT o.payload FROM transitions t JOIN observations o ON o.id=t.observation_id WHERE t.id=?',(idx,)).fetchone()[0]
            raw=_decode_observation(blob)
            obs={'observation.state':np.concatenate([raw['joint_position'],raw['eef_position']]).astype(np.float32),
                 'task':model.prompt,**{f'observation.images.{v}':raw[v] for v in ('rgb','wrist_left','wrist_right')}}
            actions,_=model.predict_timed(obs,seed=42)
            action,_=adapt_actions(actions,index=0,gain=1.,max_step_m=c['max_step_mm']/1000)
            if ep in Replay.discarded_ids(db):continue
            current=db.execute('SELECT o.payload FROM transitions t JOIN observations o ON o.id=t.observation_id WHERE t.id=?',(idx,)).fetchone()
            if current is None or current[0]!=blob:raise ValueError('Replay observation changed during inference')
            write_proposal(writer,key=offline_key(c['contract_sha256'],ep,step,blob),episode=ep,step=step,
                           contract=c['contract_sha256'],kind='offline',action=action,observation=blob,model=provenance)
            written+=1
            if written==1 or written%100==0:
                print(json.dumps(dict(written=written,remaining=len(missing[:args.limit])-written,seconds=round(time.monotonic()-start,1))),flush=True)
    finally:writer.close();db.close()
    print(json.dumps(dict(completed=True,written=written,robot_output=False)),flush=True)

if __name__=='__main__':main()
