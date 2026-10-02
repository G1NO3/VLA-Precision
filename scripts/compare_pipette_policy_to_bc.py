#!/usr/bin/env python3
"""Paired offline deployment-path comparison, original BC versus ACoB.

Reads finalized replay only. No Redis/DDS connection or robot output. Uses
identical observations and flow noise for both models; outputs metres, not
normalized losses. This is a replay probe, not a task-success evaluation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE/'VLAPolicyBridge/scripts'))
sys.path.insert(0, str(WORKSPACE/'VLAPolicyBridge'))


def assert_gpu_idle():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = p.read_bytes().split(b'\0')
            names = [v.rsplit(b'/', 1)[-1] for v in argv[:4]]
            if (b'run_pipette_pi05_policy.py' in names or
                    (b'pipette_rl.py' in names and b'train' in argv) or
                    (b'backfill_pipette_proposals.py' in names and b'--apply' in argv)):
                raise RuntimeError('Policy/learner/backfill became active; stop offline comparison: '+str(p.parent))
        except OSError:
            pass


def paired_stats(bc, rl):
    b, r = bc.reshape(-1, 3)*1000, rl.reshape(-1, 3)*1000
    diff = r-b
    distance = np.linalg.norm(diff, axis=-1)
    bn, rn = np.linalg.norm(b, axis=-1), np.linalg.norm(r, axis=-1)
    valid = (bn>.05)&(rn>.05)
    cosine = np.sum(b[valid]*r[valid], axis=-1)/(bn[valid]*rn[valid])
    return dict(axis_rmse_mm=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
                axis_mean_delta_mm=diff.mean(axis=0).tolist(),
                l2_mean_mm=float(distance.mean()), l2_median_mm=float(np.median(distance)),
                l2_p95_mm=float(np.quantile(distance,.95)), l2_max_mm=float(distance.max()),
                bc_action_norm_mean_mm=float(bn.mean()), rl_action_norm_mean_mm=float(rn.mean()),
                relative_rms_difference=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(b**2)),1e-12)),
                mean_cosine_nontrivial=float(cosine.mean()) if len(cosine) else None,
                cosine_eligible_count=int(valid.sum()),
                bc_mean_xyz_mm=b.mean(axis=0).tolist(), rl_mean_xyz_mm=r.mean(axis=0).tolist())


def seed_diagnostics(bc, rl, seeds):
    result={}
    for i,seed in enumerate(seeds):
        b,r=bc[:,i],rl[:,i]
        item=dict(first_action=paired_stats(b[:,0],r[:,0]),whole_chunk=paired_stats(b,r))
        for name,bb,rr in [('first',b[:,0],r[:,0]),('whole_chunk',b.reshape(-1,3),r.reshape(-1,3))]:
            mask=bb[:,2]<-.00001  # BC predicts >0.01 mm down, not numerical zero.
            item[name+'_bc_downward_subset']=dict(count=int(mask.sum()),total=len(mask),
                threshold_mm=.01,
                bc_mean_z_mm=float(bb[mask,2].mean()*1000) if mask.any() else None,
                rl_mean_z_mm=float(rr[mask,2].mean()*1000) if mask.any() else None,
                fraction_rl_less_downward=float((rr[mask,2]>bb[mask,2]).mean()) if mask.any() else None)
        result[str(seed)]=item
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes-per-outcome',type=int,default=6)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    args=parser.parse_args()
    if args.episodes_per_outcome<1 or not args.seeds:parser.error('Need episodes and seeds')
    assert_gpu_idle()
    args.output.mkdir(parents=True,exist_ok=False)
    from vla_precision.pipette_rl.config import load_config, DEFAULT_CONFIG
    from vla_precision.pipette_rl.replay import Replay
    from vla_policy_bridge.hil.replay import _decode_observation
    from pi05_standalone_policy import StandalonePi05Policy, VIEWS, PROMPT

    config=load_config(DEFAULT_CONFIG)
    metadata=json.loads((args.checkpoint/'metadata.json').read_text())
    assert metadata['contract_sha256']==config['contract_sha256']
    assert Path(metadata['baseline'])==Path(config['baseline'])
    replay=Replay(config);replay.refresh()
    observations=[];samples=[]
    db=replay.connect()
    try:
        for success in (True,False):
            episodes=sorted((ep for ep,m in replay.closed.items() if m['success']==success),
                            key=lambda ep:replay.closed[ep]['last_timestamp_ns'])
            indices=np.unique(np.linspace(0,len(episodes)-1,min(len(episodes),args.episodes_per_outcome),dtype=int))
            for index in indices:
                ep=episodes[index]
                rows=db.execute("SELECT t.step_index,o.payload,t.info_json,t.intervention FROM transitions t JOIN observations o ON o.id=t.observation_id WHERE t.buffer='online' AND t.episode_id=? ORDER BY t.step_index",(ep,)).fetchall()
                # Early task state, mid-final-5s state, and last real observation.
                positions=sorted(set([int(.2*(len(rows)-1)),max(0,len(rows)-1-75),len(rows)-1]))
                for position in positions:
                    step,blob,info,intervention=rows[position];raw=_decode_observation(blob)
                    state=np.concatenate([np.asarray(raw['joint_position'])[:29],np.asarray(raw['eef_position'])]).astype(np.float32)
                    assert state.shape==(32,) and np.isfinite(state).all()
                    observations.append({'observation.state':state,'task':PROMPT,
                                         **{'observation.images.'+v:raw[v] for v in VIEWS}})
                    samples.append(dict(episode_id=ep,step=step,steps_to_terminal=len(rows)-1-step,
                                        success=success,intervention=bool(intervention),
                                        observation_sha256=hashlib.sha256(blob).hexdigest(),
                                        measured_xyz=state[-3:].tolist()))
    finally:db.close()
    if not samples:raise ValueError('No finalized observations')
    (args.output/'samples.json').write_text(json.dumps(samples,indent=2))
    print(json.dumps(dict(stage='prepared',observations=len(samples),episodes=len({s['episode_id'] for s in samples}),seeds=args.seeds)),flush=True)
    baseline=Path(config['baseline']);predictions={}
    for label,adapter in [('bc',None),('rl',args.checkpoint)]:
        assert_gpu_idle(); print(json.dumps(dict(stage='loading',model=label)),flush=True)
        model=StandalonePi05Policy(baseline.parents[1],step=baseline.name,acob_checkpoint=adapter)
        output=np.zeros((len(samples),len(args.seeds),10,3),np.float32)
        start=time.monotonic()
        for i,obs in enumerate(observations):
            assert_gpu_idle()
            for j,seed in enumerate(args.seeds):
                noise=np.random.default_rng(seed).standard_normal((10,32)).astype(np.float32)
                output[i,j]=model.infer(dict(obs),noise=noise)
                if not np.isfinite(output[i,j]).all():raise ValueError('Nonfinite policy output')
            if i%6==0 or i==len(samples)-1:
                print(json.dumps(dict(stage='inference',model=label,done=i+1,total=len(samples),seconds=round(time.monotonic()-start,1))),flush=True)
        np.save(args.output/(label+'_actions_m.npy'),output)
        predictions[label]=output
        del model
        import jax
        jax.clear_caches();gc.collect()

    bc,rl=predictions['bc'],predictions['rl']
    # Policy norm cap then actor per-axis rate cap only; no simulated IK/workspace.
    def limited(a):
        norm=np.linalg.norm(a,axis=-1,keepdims=True)
        a=a*np.minimum(1.,.002/np.maximum(norm,1e-30))
        return np.clip(a,-np.array([.05,.05,.04])/30,np.array([.05,.05,.04])/30)
    report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=metadata['state_sha256'],
                step=metadata['step'],baseline=str(baseline),norm_sha256=config['norm_sha256'],
                contract_sha256=config['contract_sha256'],seeds=args.seeds,observations=len(samples),
                episodes=len({s['episode_id'] for s in samples}),
                first_action=paired_stats(bc[:,:,0],rl[:,:,0]),whole_chunk=paired_stats(bc,rl),
                first_action_after_rate_limits=paired_stats(limited(bc[:,:,0]),limited(rl[:,:,0])),
                per_seed=seed_diagnostics(bc,rl,args.seeds),
                limitations=['Outcome-stratified replay probe, not deployment-distribution weighted.',
                             'Same noise for both models; no robot motion or closed-loop success estimate.',
                             'Rate-limit comparison omits workspace/IK/orientation/history effects.'])
    if len(args.seeds)>1:
        report['bc_between_noise_seeds_first_action']=paired_stats(bc[:,0,0],bc[:,1,0])
    for label,mask in [('success',[s['success'] for s in samples]),('failure',[not s['success'] for s in samples]),
                       ('tail5s',[s['steps_to_terminal']<150 for s in samples])]:
        if any(mask):report[label+'_first_action']=paired_stats(bc[mask,:,0],rl[mask,:,0])
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='completed',report=report)),flush=True)


if __name__=='__main__':main()
