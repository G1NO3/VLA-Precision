#!/usr/bin/env python3
"""Native JAX flow BC on labelled human corrections only. No robot commands."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'VLA-Precision/src'),str(ROOT/'VLAPolicyBridge'),str(ROOT/'VLAPolicyBridge/scripts')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(ROOT/'cache/jax'))
import numpy as np
from vla_precision.pipette_rl.correction_data import DEFAULT,load_config,build_manifest,read_db,masked_xyz_mse,masked_native_mse


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def idle():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0');names=[x.rsplit(b'/',1)[-1] for x in args[:4]]
            if int(p.parent.name)==os.getpid():continue
            if (b'run_pipette_pi05_policy.py' in names or b'run_pipette_hil_actor.py' in names
                    or (b'pipette_rl.py' in names and b'train' in args)
                    or (b'pipette_rlt.py' in names and any(x in args for x in (b'cache',b'token',b'train')))
                    or (b'pipette_correction_bc.py' in names and any(x in args for x in (b'cache',b'train',b'evaluate')))):
                raise RuntimeError('Stop actor, policy and other learners before correction-only BC')
        except OSError:pass


def manifest(c):
    m=json.loads((Path(c['output'])/'manifest.json').read_text())
    if m['config']['config_sha256']!=c['config_sha256']:raise ValueError('Training manifest configuration mismatch')
    with read_db(c['replay_db']) as db:
        revoked={r[0] for r in db.execute('select episode_id from discarded_episodes')}
    if revoked.intersection(x['episode'] for x in m['chunks']):
        raise ValueError('A selected episode was discarded; rebuild the manifest before training')
    return m


def cache(c,args):
    idle()
    import jax
    import jax.numpy as jnp
    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU is required; refusing CPU fallback')
    from flax import nnx
    from openpi.models import model as om
    from vla_precision.integrations.openpi.context import encode_context
    from pi05_standalone_policy import StandalonePi05Policy
    from vla_policy_bridge.hil.replay import _decode_observation
    m=manifest(c);folder=Path(c.get('context_cache',str(Path(c['output'])/'contexts')));folder.mkdir(exist_ok=True)
    base=Path(c['baseline']);policy=StandalonePi05Policy(base.parents[1],step=base.name)
    encode=nnx.jit(encode_context)
    for n,x in enumerate(m['chunks']):
        idle();path=folder/f"{x['observation_id']}.npz"
        if path.exists():continue
        with read_db(c['replay_db']) as db:blob=db.execute('select payload from observations where id=?',(x['observation_id'],)).fetchone()[0]
        raw=_decode_observation(blob)
        obs={'observation.state':np.concatenate([raw['joint_position'][:29],raw['eef_position']]).astype(np.float32),
             **{'observation.images.'+k:raw[k] for k in ('rgb','wrist_left','wrist_right')}}
        transformed=policy.policy._input_transform(obs)
        inp=om.Observation.from_dict(jax.tree.map(lambda v:jnp.asarray(v)[None],transformed))
        context,mask,offset=encode(policy.policy._model,inp)
        bits=np.asarray(jax.lax.bitcast_convert_type(context[0],jnp.uint16))
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez(f,context=bits,mask=np.asarray(mask[0]),offset=np.asarray(offset[0]),
                                         state=np.asarray(transformed['state']),config_sha256=np.array(c.get('context_config_sha256',c['config_sha256'])))
        tmp.replace(path)
        if n%20==0 or n==len(m['chunks'])-1:print(json.dumps(dict(stage='cache',done=n+1,total=len(m['chunks']))),flush=True)
    print('Human correction context cache complete',flush=True)


def initialize(c):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.shared import nnx_utils
    from openpi.training import optimizer,weight_loaders
    from vla_precision.integrations.openpi.configs import get_config
    from vla_precision.integrations.openpi.training_state import initialize_train_state
    os.environ['OPENPI_DATA_HOME']=str(Path(c['baseline']).parents[1]/'tokenizer')
    cfg=get_config('pi05_acob_pipette')  # architecture/filter only; no RL agent/critic
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,action_horizon=10,action_dim=32,max_token_len=200),
        batch_size=c['batch_size'],ema_decay=None,exp_name='human_corrections_only',
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(c['baseline'])/'params')),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10,peak_lr=c['learning_rate'],decay_steps=1000,decay_lr=c['learning_rate']))
    mesh=jax.sharding.Mesh(np.array(jax.devices()),('data',))
    state=initialize_train_state(cfg,jax.random.key(c['seed']),mesh,resume=False)
    params=nnx_utils.state_map(state.params,nnx_utils.PathRegex('.*lora_b.*'),lambda p:p.replace(jnp.zeros_like(p.value)))
    return cfg,dataclasses.replace(state,params=params)


def batches(c,chunks,order):
    from openpi import transforms
    from openpi.shared import normalize
    norm=transforms.Normalize(normalize.load(Path(c['norm_stats']).parent),use_quantiles=True)
    for start in range(0,len(order),c['batch_size']):
        ids=list(order[start:start+c['batch_size']]);real=len(ids)
        ids+=ids[:1]*(c['batch_size']-len(ids))
        records=[]
        for i in ids:
            x=chunks[int(i)]
            folder=Path(c.get('context_cache',str(Path(c['output'])/'contexts')))
            with np.load(folder/f"{x['observation_id']}.npz") as f:
                if str(f['config_sha256'])!=c.get('context_config_sha256',c['config_sha256']):raise ValueError('Stale context cache')
                data={k:f[k].copy() for k in ('context','mask','offset','state')}
            a=np.asarray(x['actions_m'],np.float32)
            a=norm({'actions':a})['actions']
            padded=np.zeros((10,32),np.float32);padded[:,:3]=a
            data.update(actions=padded,loss_mask=np.asarray(x['loss_mask'],np.float32))
            records.append(data)
        batch={k:np.stack([r[k] for r in records]) for k in records[0]}
        batch['loss_mask'][real:]=0
        yield batch


def functions(cfg,loss_dimensions=32):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.models import model as om
    from vla_precision.integrations.openpi.context import velocity_from_context
    def loss(model,batch,rng):
        nr,tr=jax.random.split(rng)
        noise=jax.random.normal(nr,batch['actions'].shape)
        t=jax.random.beta(tr,1.5,1,(noise.shape[0],))*.999+.001
        a=batch['actions'];xt=t[:,None,None]*noise+(1-t[:,None,None])*a
        obs=om.Observation(images={},image_masks={},state=batch['state'])
        v=velocity_from_context(model,obs,batch['context'],batch['mask'],batch['offset'],xt,t)
        objective=masked_native_mse if loss_dimensions==32 else masked_xyz_mse
        return objective(v,noise-a,batch['loss_mask'])
    @jax.jit
    def evaluate(state,batch,rng):
        model=nnx.merge(state.model_def,state.params);model.eval()
        return loss(model,batch,rng)
    @jax.jit
    def update(state,batch,rng):
        model=nnx.merge(state.model_def,state.params);model.train()
        value,grads=nnx.value_and_grad(loss,argnums=nnx.DiffState(0,cfg.trainable_filter))(model,batch,rng)
        params=state.params.filter(cfg.trainable_filter)
        updates,opt=state.tx.update(grads,state.opt_state,params)
        nnx.update(model,optax.apply_updates(params,updates))
        return dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=opt),value
    return update,evaluate


def save(c,cfg,state,epoch,val_loss):
    import jax
    from flax import serialization
    out=Path(c['output'])/'checkpoints';out.mkdir(exist_ok=True)
    path=out/f'epoch-{epoch:02d}';path.mkdir()
    payload=serialization.msgpack_serialize(jax.device_get(dict(lora=state.params.filter(cfg.trainable_filter).to_pure_dict())))
    (path/'lora.msgpack').write_bytes(payload)
    meta=dict(format='pipette.human_bc.v1',config=c,baseline=c['baseline'],norm_sha256=c['norm_sha256'],
              config_sha256=c['config_sha256'],step=int(state.step),epoch=epoch,validation_loss=val_loss,
              action_representation='delta_xyz',state_sha256=hashlib.sha256(payload).hexdigest(),
              training_mode='action_expert_lora',objective='masked_human_only_native_flow_matching' if c.get('loss_dimensions',3)==32 else 'masked_human_only_xyz_flow_matching',
              rollout_targets=0,reward_loss=False,critic=False)
    write(path/'metadata.json',meta);(path/'READY').write_text('complete\n')
    return path


def train(c,args):
    idle()
    import jax
    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU is required; refusing CPU fallback')
    m=manifest(c);out=Path(c['output'])
    if (out/'training.started').exists():raise ValueError('Training already started; use a separate output for another run')
    (out/'training.started').write_text(str(time.time()))
    cfg,state=initialize(c);update,evaluate=functions(cfg,c.get('loss_dimensions',3))
    train_ids=[i for i,x in enumerate(m['chunks']) if x['split']=='train']
    val_ids=[i for i,x in enumerate(m['chunks']) if x['split']=='validation']
    print(json.dumps(dict(stage='initialized',trainable_parameters=int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter)))))),flush=True)
    def validation():
        values=[];weights=[]
        for i,b in enumerate(batches(c,m['chunks'],val_ids)):
            idle();values.append(float(evaluate(state,b,jax.random.fold_in(jax.random.key(1234),i))))
            weights.append(float(b['loss_mask'].sum()))
        return float(np.average(values,weights=weights))
    val=validation();best=val;best_path=None
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=0,step=0,validation_loss=val))+'\n')
    print(json.dumps(dict(stage='baseline_validation',loss=val)),flush=True)
    rng=np.random.default_rng(c['seed'])
    for epoch in range(1,c['epochs']+1):
        order=rng.permutation(train_ids)
        for b in batches(c,m['chunks'],order):
            idle();state,value=update(state,b,jax.random.fold_in(jax.random.key(c['seed']),state.step))
            value=float(value);step=int(state.step)
            if not np.isfinite(value):raise ValueError('Nonfinite supervised loss')
            if step%10==0 or step==1:
                metric=dict(epoch=epoch,step=step,train_loss=value,timestamp=time.time())
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                print(json.dumps(metric),flush=True)
        manifest(c);val=validation();path=save(c,cfg,state,epoch,val)
        metric=dict(epoch=epoch,step=int(state.step),validation_loss=val)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        print(json.dumps(metric),flush=True)
        if best_path is None or val<best:
            best=val;best_path=path
            write(out/'best.json',dict(checkpoint=str(path),epoch=epoch,validation_loss=val))
    write(out/'completed.json',dict(step=int(state.step),epochs=c['epochs'],best_checkpoint=str(best_path),validation_loss=best))
    print('Correction-only BC completed: '+str(best_path),flush=True)


def evaluate_actions(c,args):
    """Use the actual deployment loader, identical inputs and deployment seed."""
    idle()
    import jax
    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU is required')
    from pi05_standalone_policy import StandalonePi05Policy
    from vla_policy_bridge.hil.replay import _decode_observation
    out=Path(c['output']);m=manifest(c)
    cases=[x for x in m['chunks'] if x['split']=='validation']
    baseline=Path(c['baseline']);adapter=None
    if args.which=='trained':adapter=json.loads((out/'completed.json').read_text())['best_checkpoint']
    policy=StandalonePi05Policy(baseline.parents[1],step=baseline.name,human_bc_checkpoint=adapter)
    actions=[];timing=[]
    for n,x in enumerate(cases):
        idle()
        with read_db(c['replay_db']) as db:blob=db.execute('select payload from observations where id=?',(x['observation_id'],)).fetchone()[0]
        raw=_decode_observation(blob)
        obs={'observation.state':np.concatenate([raw['joint_position'][:29],raw['eef_position']]).astype(np.float32),
             **{'observation.images.'+k:raw[k] for k in ('rgb','wrist_left','wrist_right')}}
        a,elapsed=policy.predict_timed(obs,seed=42)
        if not np.isfinite(a).all():raise ValueError('Nonfinite deployment prediction')
        actions.append(a);timing.append(elapsed)
        if n%25==0:print(json.dumps(dict(stage='evaluate',model=args.which,done=n+1,total=len(cases))),flush=True)
    a=np.stack(actions);target=np.asarray([x['actions_m'] for x in cases]);mask=np.asarray([x['loss_mask'] for x in cases])
    error=(a-target)[mask]*1000
    limited=a*np.minimum(1.,.002/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12))
    down=mask & (target[:,:,2]<-.00001)
    summary=dict(model=args.which,checkpoint=adapter or str(baseline),observations=len(cases),human_targets=int(mask.sum()),
        seed=42,rmse_mm=np.sqrt(np.mean(error**2,axis=0)).tolist(),xyz_l2_rmse_mm=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        clipped_rmse_mm=np.sqrt(np.mean(((limited-target)[mask]*1000)**2,axis=0)).tolist(),
        over_2mm_fraction=float(np.mean(np.linalg.norm(a[mask],axis=-1)>.002)),
        downward_targets=int(down.sum()),downward_target_mean_z_mm=float(target[:,:,2][down].mean()*1000),
        downward_predicted_mean_z_mm=float(a[:,:,2][down].mean()*1000),
        downward_sign_fraction=float(np.mean(a[:,:,2][down]<-.00001)),
        warm_inference_median_ms=float(np.median(timing[1:])),
        note='Held-out episode action-label error; not measured physical TCP tracking or task success')
    np.savez(out/(args.which+'-actions.npz'),actions_m=a,targets_m=target,mask=mask,
             observation_ids=np.array([x['observation_id'] for x in cases]))
    write(out/(args.which+'-evaluation.json'),summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','cache','train','evaluate'])
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--which',choices=['baseline','trained'],default='trained')
    args=p.parse_args();c=load_config(args.config)
    if args.command=='prepare':
        out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
        if (out/'training.started').exists():raise ValueError('Cannot replace a training manifest after training started')
        m=build_manifest(c);write(out/'manifest.json',m);print(json.dumps(m['counts'],indent=2))
    elif args.command=='cache':cache(c,args)
    elif args.command=='train':train(c,args)
    else:evaluate_actions(c,args)


if __name__=='__main__':main()
