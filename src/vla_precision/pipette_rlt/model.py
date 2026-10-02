"""Frozen JAX pi05 features with the upstream RLinf token and small torch heads."""
from pathlib import Path
import importlib.util
import hashlib
import sys
import time
import numpy as np
import torch
from torch import nn
from .core import ROOT


def token_model(c):
    # This upstream module has no RLinf runtime dependencies. Loading it directly
    # avoids importing Ray/robot SDKs into the native-JAX inference process.
    path = ROOT/'RLinf/rlinf/models/embodiment/modules/rlt_token_transformer.py'
    spec = importlib.util.spec_from_file_location('pipette_upstream_rlt_token', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RLTTokenTransformer(input_dim=c['token_input_dim'],embed_dim=c['token_dim'],
        prefix_seq_len=c['token_prefix_len'],num_layers=c['token_layers'],
        num_heads=c['token_heads'],mlp_ratio=c['token_mlp_ratio'])


class FrozenFeatures:
    """Use exactly the deployed BC preprocessing and native final prefix states."""
    def __init__(self, c):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as om
        from openpi.models.pi0 import make_attn_mask
        sys.path.insert(0,str(ROOT/'VLAPolicyBridge/scripts'))
        from pi05_standalone_policy import StandalonePi05Policy
        baseline=Path(c['baseline'])
        self.base=StandalonePi05Policy(baseline.parents[1],step=baseline.name)
        self.c=c

        @nnx.jit
        def encode(model, obs):
            obs=om.preprocess_observation(None,obs,train=False)
            tokens,mask,ar=model.embed_prefix(obs)
            outputs,_=model.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
            return outputs[0],mask
        self._encode=encode

    def extract(self, observation):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as om
        transformed=self.base.policy._input_transform(jax.tree.map(lambda x:x,observation))
        batched=jax.tree.map(lambda x:jnp.asarray(x)[None],transformed)
        prefix,mask=self._encode(self.base.policy._model,om.Observation.from_dict(batched))
        noise=np.random.default_rng(self.c['seed']).standard_normal((10,32)).astype(np.float32)
        reference=self.base.infer(observation,noise=noise)
        return dict(prefix=np.asarray(prefix[0],np.float16),mask=np.asarray(mask[0],bool),
                    proprio=np.asarray(transformed['state'],np.float32),reference_m=reference)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,256),nn.LayerNorm(256),nn.Tanh(),
        nn.Linear(256,256),nn.LayerNorm(256),nn.Tanh(),
        nn.Linear(256,256),nn.LayerNorm(256),nn.Tanh(),nn.Linear(256,outputs))


class Heads(nn.Module):
    """RLT conditional Gaussian actor and twin chunk critics, without flow BC."""
    def __init__(self,c):
        super().__init__()
        self.h=c['credit_horizon'];self.std=c['fixed_std']
        self.register_buffer('action_scale',torch.tensor(c['action_scale_m']),persistent=False)
        state_dim=c['token_dim']+32
        self.actor=mlp(state_dim+self.h*3,self.h*3)
        self.critics=nn.ModuleList([mlp(state_dim+self.h*3,1) for _ in range(2)])

    def act(self,z,proprio,reference,*,dropout=0.,sample=False):
        ref=reference[:,:self.h].flatten(1)
        if dropout:
            ref=ref*(torch.rand((len(ref),1),device=ref.device)>=dropout)
        mean=self.actor(torch.cat([z,proprio,ref],dim=-1))
        raw=mean+torch.randn_like(mean)*self.std if sample else mean
        action=raw.tanh().reshape(-1,self.h,3)
        physical_norm=torch.linalg.vector_norm(action*self.action_scale,dim=-1,keepdim=True)
        return action*torch.clamp(.002/physical_norm.clamp(min=1e-12),max=1.)

    def q(self,z,proprio,actions,valid=None):
        if valid is not None:
            actions=actions*valid[...,None]
        x=torch.cat([z,proprio,actions.flatten(1)],dim=-1)
        return torch.cat([net(x) for net in self.critics],dim=-1)


def load_torch(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def validate_checkpoint(path,c,*,allow_smoke=False):
    import json
    path=Path(path)
    m=json.loads((path/'metadata.json').read_text())
    if not (path/'READY').is_file() or m.get('format')!='pipette.rlt.v1':
        raise ValueError('Incomplete RLT checkpoint')
    if m.get('smoke_only') and not allow_smoke:
        raise ValueError('RLT smoke checkpoint cannot control the robot')
    if m['contract_sha256']!=c['contract_sha256'] or m['norm_sha256']!=c['norm_sha256']:
        raise ValueError('RLT checkpoint configuration mismatch')
    for name in ['heads.pt','token.pt']:
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=m['sha256'][name]:
            raise ValueError('RLT checkpoint hash mismatch: '+name)
    return m


class RLTPolicy:
    """Native measured observations -> five RLT actions, padded for the wire."""
    def __init__(self,c,checkpoint,*,device='cuda',allow_smoke=False):
        self.c=c
        self.metadata=validate_checkpoint(checkpoint,c,allow_smoke=allow_smoke)
        self.features=FrozenFeatures(c)
        self.token=token_model(c).to(device).eval()
        self.token.load_state_dict(load_torch(Path(checkpoint)/'token.pt')['state'])
        self.heads=Heads(c).to(device).eval()
        self.heads.load_state_dict(load_torch(Path(checkpoint)/'heads.pt')['state'])
        self.device=device;self.prompt=self.features.base.prompt
        self.metadata={**self.features.base.metadata,**self.metadata,'rlt_checkpoint':str(Path(checkpoint).resolve()),
                       'rlt_horizon':c['credit_horizon'],'action_representation':'delta_xyz'}

    @torch.inference_mode()
    def predict_timed(self,observation,*,seed=42):
        if seed!=self.c['seed']:
            raise ValueError('RLT reference seed must match feature cache')
        start=time.perf_counter();f=self.features.extract(observation)
        tensor=lambda x:torch.tensor(np.asarray(x),device=self.device)
        z=self.token.encode_flat(tensor(f['prefix'])[None].float(),tensor(f['mask'])[None])
        z=nn.functional.layer_norm(z,(self.c['token_dim'],))
        scale=np.asarray(self.c['action_scale_m'],np.float32)
        ref=np.clip(f['reference_m']/scale,-1,1)
        a=self.heads.act(z,tensor(np.clip(f['proprio'],-5,5))[None],tensor(ref)[None])[0].cpu().numpy()*scale
        result=np.zeros((10,3),np.float32);result[:self.c['credit_horizon']]=a
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite RLT action')
        return result,(time.perf_counter()-start)*1000
