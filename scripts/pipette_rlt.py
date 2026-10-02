#!/usr/bin/env python3
"""Prepare native-JAX RL-token features, train readout, and train chunk actor/critic.

All commands are offline and never connect to the robot. Live deployment uses
the existing policy bridge with an explicit RLT checkpoint after validation.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VLA-Precision/src'))
sys.path.insert(0,str(ROOT/'VLAPolicyBridge'))
import numpy as np
from vla_precision.pipette_rlt.core import DEFAULT,bc_targets,load_config,open_replay,replay_chunks,sample_indices


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(value),indent=2));tmp.replace(path)


def assert_idle():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0')
            names=[a.rsplit(b'/',1)[-1] for a in args[:4]]
            if (b'run_pipette_hil_actor.py' in names or b'run_pipette_pi05_policy.py' in names or (b'pipette_rl.py' in names and b'train' in args)
                    or (int(p.parent.name)!=os.getpid() and b'pipette_rlt.py' in names and
                        any(x in args for x in [b'cache',b'token',b'train']))):
                raise RuntimeError('Stop HIL actor, policy and other learners before offline RLT GPU work')
        except OSError:pass


def feature_key(db,obs):
    return hashlib.sha256((str(Path(db).resolve())+':'+str(obs)).encode()).hexdigest()[:24]


def prepare(c,args):
    assert_idle()
    from vla_precision.pipette_rlt.model import FrozenFeatures
    from vla_policy_bridge.hil.replay import _decode_observation
    chunks,outcomes=replay_chunks(c)
    recent,more=replay_chunks(c,source=False);chunks+=recent;outcomes.update(more)
    if not chunks:raise ValueError('No eligible replay chunks')
    rng=np.random.default_rng(c['seed'])
    # Uniform capped subset without replacement; training subsequently applies
    # its explicit tail/full mixture within this recorded subset.
    chosen=np.sort(rng.choice(len(chunks),min(len(chunks),args.max_chunks),replace=False))
    selected=[chunks[int(i)] for i in chosen]
    cache=Path(c['output'])/'features';cache.mkdir(parents=True,exist_ok=True)
    old=cache/'manifest.json'
    if old.exists() and json.loads(old.read_text())['contract_sha256']!=c['contract_sha256']:
        raise ValueError('Feature cache belongs to another RLT experiment')
    model=FrozenFeatures(c)
    needed={}
    for x in selected:
        for field in ['observation_id','next_observation_id']:
            key=feature_key(x['db'],x[field]);x[field+'_key']=key
            needed[key]=(x['db'],x[field])
    for n,(key,(dbpath,oid)) in enumerate(needed.items()):
        path=cache/(key+'.npz')
        if path.exists():continue
        assert_idle()
        with open_replay(dbpath) as db:
            blob=db.execute('select payload from observations where id=?',(oid,)).fetchone()[0]
        raw=_decode_observation(blob)
        obs={'observation.state':np.concatenate([raw['joint_position'][:29],raw['eef_position']]).astype(np.float32),
             **{'observation.images.'+k:raw[k] for k in ['rgb','wrist_left','wrist_right']}}
        features=model.extract(obs)
        if features['prefix'].shape!=(c['token_prefix_len'],c['token_input_dim']):
            raise ValueError('Unexpected native prefix shape '+str(features['prefix'].shape))
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,**features,observation_sha256=np.array(hashlib.sha256(blob).hexdigest()))
        tmp.replace(path)
        if n%10==0:print(json.dumps(dict(stage='features',done=n+1,total=len(needed))),flush=True)
    manifest=dict(contract_sha256=c['contract_sha256'],config=c,chunks=selected,
                  outcomes=outcomes,available_chunks=len(chunks),selected_chunks=len(selected),
                  observations=len(needed),created_time=time.time())
    write(old,manifest)
    print(json.dumps(dict(stage='features_complete',chunks=len(selected),observations=len(needed))),flush=True)


def manifest(c):
    cache=Path(c['output'])/'features'
    m=json.loads((cache/'manifest.json').read_text())
    if m['contract_sha256']!=c['contract_sha256']:raise ValueError('Stale feature cache')
    # Revoke deleted/discarded episodes even after feature extraction.
    _,old=replay_chunks(c);_,new=replay_chunks(c,source=False)
    valid=set(old)|set(new)
    m['chunks']=[x for x in m['chunks'] if x['episode'] in valid]
    if not m['chunks']:raise ValueError('No non-discarded feature chunks')
    return cache,m


def fit_token(c,args):
    assert_idle()
    import torch
    from vla_precision.pipette_rlt.model import token_model
    torch.set_num_threads(4);torch.manual_seed(c['seed'])
    cache,m=manifest(c)
    episodes=sorted({x['episode'] for x in m['chunks']})
    val_eps=set(episodes[::5]) if len(episodes)>1 else set()
    trainkeys=sorted({x[k] for x in m['chunks'] if x['episode'] not in val_eps for k in ['observation_id_key','next_observation_id_key']})
    valkeys=sorted({x['observation_id_key'] for x in m['chunks'] if x['episode'] in val_eps})[:32]
    model=token_model(c).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['token_lr'])
    rng=np.random.default_rng(c['seed'])
    out=Path(c['output'])/('token-smoke' if args.smoke else 'token')
    if (out/'READY').exists():raise FileExistsError('Finished token exists; use it or create a new experiment')
    out.mkdir(parents=True,exist_ok=True)
    def batch(keys):
        records=[]
        for k in keys:
            with np.load(cache/(k+'.npz')) as f:records.append((f['prefix'].copy(),f['mask'].copy()))
        return torch.tensor(np.stack([r[0] for r in records]),device=args.device,dtype=torch.float32),torch.tensor(np.stack([r[1] for r in records]),device=args.device)
    for step in range(1,args.updates+1):
        assert_idle();model.train()
        prefix,mask=batch(rng.choice(trainkeys,size=c['token_batch_size']))
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
            loss,_=model.loss(prefix,mask)
        if not torch.isfinite(loss):raise ValueError('Nonfinite token loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%10==0 or step==1 or step==args.updates:
            metric=dict(step=step,reconstruction_mse=float(loss.detach()))
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            print(json.dumps(dict(stage='token',**metric)),flush=True)
    model.eval();validation=[]
    with torch.inference_mode():
        for key in valkeys:
            prefix,mask=batch([key])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                loss,_=model.loss(prefix,mask)
            validation.append(float(loss))
    torch.save(dict(state=model.cpu().state_dict(),contract_sha256=c['contract_sha256'],smoke_only=args.smoke,updates=args.updates),out/'token.pt')
    write(out/'metadata.json',dict(contract_sha256=c['contract_sha256'],smoke_only=args.smoke,updates=args.updates,
        train_episodes=sorted(set(episodes)-val_eps),validation_episodes=sorted(val_eps),
        validation_mse=float(np.mean(validation)) if validation else None))
    (out/'READY').write_text('complete\n')


def fit_heads(c,args):
    assert_idle()
    import torch
    from vla_precision.pipette_rlt.model import Heads,token_model,load_torch
    torch.set_num_threads(4);torch.manual_seed(c['seed'])
    cache,m=manifest(c);chunks=m['chunks']
    out=Path(c['output'])/('learner-smoke' if args.smoke else 'learner');out.mkdir(parents=True,exist_ok=True)
    resume=None
    if args.resume and (out/'latest.json').exists():
        resume=Path(json.loads((out/'latest.json').read_text())['checkpoint'])
        from vla_precision.pipette_rlt.model import validate_checkpoint
        resume_meta=validate_checkpoint(resume,c,allow_smoke=args.smoke)
    tokenpath=(resume/'token.pt' if resume else
               Path(args.token or Path(c['output'])/('token-smoke' if args.smoke else 'token'))/'token.pt')
    tokenstate=load_torch(tokenpath)
    if tokenstate['contract_sha256']!=c['contract_sha256'] or (tokenstate['smoke_only'] and not args.smoke):
        raise ValueError('Token checkpoint configuration/smoke mismatch')
    token=token_model(c).to(args.device);token.load_state_dict(tokenstate['state']);token.eval()
    latent={}
    with torch.inference_mode():
        for key in sorted({x[k] for x in chunks for k in ['observation_id_key','next_observation_id_key']}):
            assert_idle()
            with np.load(cache/(key+'.npz')) as f:
                p=torch.tensor(f['prefix'],device=args.device,dtype=torch.float32)[None]
                mask=torch.tensor(f['mask'],device=args.device)[None]
                z=token.encode_flat(p,mask)
                z=torch.nn.functional.layer_norm(z,(c['token_dim'],))
                latent[key]=(z[0].cpu().numpy(),np.clip(f['proprio'],-5,5),np.clip(f['reference_m']/np.asarray(c['action_scale_m']),-1,1))
    del token
    model=Heads(c).to(args.device);target=copy.deepcopy(model)
    ao=torch.optim.Adam(model.actor.parameters(),lr=c['actor_lr'])
    qo=torch.optim.Adam(model.critics.parameters(),lr=c['critic_lr'])
    rng=np.random.default_rng(c['seed']);scale=np.asarray(c['action_scale_m'],np.float32)
    start_step=0
    if resume:
        saved=load_torch(resume/'heads.pt')
        model.load_state_dict(saved['state']);target.load_state_dict(saved['target'])
        ao.load_state_dict(saved['actor_optimizer']);qo.load_state_dict(saved['critic_optimizer'])
        start_step=resume_meta['step']
    final_step=start_step+args.updates
    checkpoint=out/f'step-{final_step:08d}'
    if checkpoint.exists():raise FileExistsError(checkpoint)
    tensor=lambda x:torch.tensor(np.asarray(x),device=args.device,dtype=torch.float32)
    warmup=min(c['actor_warmup_updates'],max(1,args.updates//2)) if args.smoke else c['actor_warmup_updates']
    for step in range(start_step+1,final_step+1):
        assert_idle()
        if step%100==0:
            _,updated=manifest(c)
            allowed={x['episode'] for x in updated['chunks']}
            chunks=[x for x in chunks if x['episode'] in allowed]
        batch=[chunks[i] for i in sample_indices(chunks,c['batch_size'],rng,c)]
        z,p,ref=map(tensor,zip(*(latent[x['observation_id_key']] for x in batch)))
        zn,pn,rn=map(tensor,zip(*(latent[x['next_observation_id_key']] for x in batch)))
        actions=tensor([np.asarray(x['actions_m'])/scale for x in batch])
        valid=tensor([x['valid'] for x in batch])
        rewards=tensor([x['reward'] for x in batch]);discount=tensor([x['discount'] for x in batch])
        for _ in range(c['critic_actor_ratio']):
            with torch.no_grad():
                nxt=target.act(zn,pn,rn,sample=True)
                y=rewards+discount*target.q(zn,pn,nxt).min(-1).values
            q=model.q(z,p,actions,valid)
            qloss=((q-y[:,None])**2).mean()
            if not torch.isfinite(qloss):raise ValueError('Nonfinite critic loss')
            qo.zero_grad();qloss.backward();torch.nn.utils.clip_grad_norm_(model.critics.parameters(),1.);qo.step()
        for param in model.critics.parameters():param.requires_grad_(False)
        pred=model.act(z,p,ref,dropout=c['reference_dropout'],sample=True)
        h=c['credit_horizon']
        targets,weights=bc_targets(ref[:,:h].detach().cpu().numpy()*scale,
            np.asarray([x['actions_m'] for x in batch]),[x['human'] for x in batch],
            [x['eligible'] for x in batch],c['human_min_action_mm'],[x['hold'] for x in batch])
        weights=tensor(weights)*valid
        bc=((pred-tensor(targets/scale))**2).mean(-1)
        bc=(bc*weights).sum()/weights.sum().clamp(min=1)
        qpi=model.q(z,p,pred,valid)[:,0].mean()
        loss=c['reference_weight']*bc-(c['q_weight']*qpi if step>warmup else 0.)
        if not torch.isfinite(loss):raise ValueError('Nonfinite actor loss')
        ao.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.);ao.step()
        for param in model.critics.parameters():param.requires_grad_(True)
        with torch.no_grad():
            for dst,src in zip(target.parameters(),model.parameters()):dst.lerp_(src,c['target_tau'])
        if step%10==0 or step==start_step+1 or step==final_step:
            metric=dict(step=step,critic_loss=float(qloss.detach()),actor_loss=float(loss.detach()),bc_loss=float(bc.detach()),q_pi=float(qpi.detach()),
                bc_weight_fraction=float(weights.mean()),rl_active=step>warmup,
                sampled_success_fraction=float(np.mean([x['success'] for x in batch])))
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            print(json.dumps(dict(stage='heads',**metric)),flush=True)
    deployable=final_step>warmup and len(chunks)>=512 and tokenstate['updates']>=1000
    if not args.smoke and not deployable:
        raise ValueError('Round no longer meets deployment requirements; latest checkpoint unchanged')
    import tempfile
    staging=Path(tempfile.mkdtemp(prefix='.saving-',dir=out))
    torch.save(dict(state=model.cpu().state_dict(),target=target.cpu().state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=qo.state_dict()),staging/'heads.pt')
    import shutil
    shutil.copyfile(tokenpath,staging/'token.pt')
    hashes={name:hashlib.sha256((staging/name).read_bytes()).hexdigest() for name in ['heads.pt','token.pt']}
    meta=dict(format='pipette.rlt.v1',contract_sha256=c['contract_sha256'],norm_sha256=c['norm_sha256'],baseline=c['baseline'],
        config=c,step=final_step,smoke_only=bool(args.smoke),sha256=hashes,
        training_episodes=sorted({x['episode'] for x in chunks}),training_chunks=len(chunks),
        token_updates=tokenstate['updates'],warmup_updates=warmup,credit_horizon=c['credit_horizon'])
    write(staging/'metadata.json',meta);(staging/'READY').write_text('complete\n')
    staging.rename(checkpoint)
    write(out/'latest.json',dict(checkpoint=str(checkpoint.resolve()),step=final_step))
    print(json.dumps(dict(stage='completed',checkpoint=str(checkpoint),step=final_step,smoke_only=args.smoke)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inspect','preflight','cache','token','train'])
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--max-chunks',type=int,default=2048)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--token',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true',help='Continue latest heads and optimizers; keep its frozen readout')
    args=p.parse_args();c=load_config(args.config)
    if args.updates<1 or args.max_chunks<1:p.error('Counts must be positive')
    if args.command in ('inspect','preflight'):
        if args.command=='preflight':assert_idle()
        chunks,outcomes=replay_chunks(c);recent,new=replay_chunks(c,source=False)
        summary=dict(source_episodes=len(outcomes),source_chunks=len(chunks),rlt_episodes=len(new),rlt_chunks=len(recent))
        print(json.dumps(dict(config=c,**summary) if args.command=='inspect' else summary,indent=2),flush=True)
        if args.command=='preflight' and min(len(chunks)+len(recent),args.max_chunks)<512:
            raise ValueError('Need at least 512 eligible chunks for a deployable RLT round; collect more labelled episodes. No training started.')
        if args.command=='preflight':
            from gui.pi05_rlt import resolve_latest
            latest=resolve_latest(c)
            print(json.dumps(dict(resume_checkpoint=str(latest) if latest else None)),flush=True)
    elif args.command=='cache':prepare(c,args)
    elif args.command=='token':fit_token(c,args)
    else:fit_heads(c,args)

if __name__=='__main__':main()
