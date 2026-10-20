#!/usr/bin/env python3
"""Offline GPU parity: zero v6d Editor equals native IL for all five phases.

Synthetic images and measured-velocity inputs only; no camera, policy socket,
robot commands, production replay or learner output. Requires cached model assets.
"""
import argparse
from pathlib import Path
import tempfile
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'VLAPolicyBridge'))
from vla_precision.gr00t_rlt.core import load_config, is_wrist_velocity
from vla_precision.gr00t_rlt.serving import LivePolicy
from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pipette_rl/pi05_fulltask_v6d_focusvel_continuous_30hz.yaml')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    c=load_config(args.config)
    if not is_wrist_velocity(c):
        parser.error('Expected the v6d wrist-velocity editor recipe')
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='v6d-editor-parity-') as temp:
        # Explicitly test a fresh zero actor without reading/publishing trained actors.
        c.update(output=str(Path(temp)/'output'),focus_camera=None,offline_smoke_test=True)
        policy=LivePolicy(c,args.device,run=temp)
        try:
            policy.capture_root=None
            assert policy.actor_kind=='editor_zero'
            from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
            native=Gr00tCompatPolicy(policy.features.wrapper,unnorm_key=c['unnorm_key'],send_state=True)
            captured={}
            extract=policy.features.extract
            def save_features(sample):
                result=extract(sample)
                captured.clear();captured.update(result)
                return result
            policy.features.extract=save_features
            for i,instruction in enumerate(INSTRUCTIONS):
                obs=synthetic_observation()
                obs['video']={'rgb':np.zeros((1,1,720,1280,3),np.uint8)}
                obs['state']['wrist_vel']=np.asarray([.01,.02,.03,-.01,-.02,-.03],np.float32)[None,None]*(i+1)
                obs['language']['annotation.human.task_description']=[[instruction]]
                actual=policy.get_action(obs,{'episode':f'offline-{i}','rollout':'offline-parity'})
                meta=actual['info']['rlt']
                assert meta['state'].shape==(65,) and meta['physical_state'].shape==(65,)
                np.testing.assert_array_equal(meta['action'],meta['reference'][:,:18])
                # Reuse exactly the crop the editor saw, so tracking cannot confound parity.
                native_obs={**obs,'video':{**obs['video'],'focus':captured['focus'][None,None]}}
                example=native.obs_to_example(native_obs)
                assert example['state'].shape==(1,6)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    torch.manual_seed(c['reference_seed'])
                    expected,_=native.get_action(native_obs)
                for key,value in expected.items():
                    assert np.isfinite(actual[key]).all()
                    np.testing.assert_allclose(actual[key],value,rtol=1e-5,atol=1e-5,err_msg=f'p{i+1}/{key}')
                print(f'PASS p{i+1}: velocity6 prompt, state65 editor, zero residual/native 30x32 parity',flush=True)
            print('PASS: all five phases; no robot I/O or production replay writes',flush=True)
        finally:
            policy.close()


if __name__=='__main__':
    main()
