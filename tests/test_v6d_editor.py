"""Velocity-conditioned reference, shared state65 editor and replay/learner isolation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import numpy as np
import pytest
import torch
import yaml

from vla_precision.gr00t_rlt.core import load_config, bridge_profile, phase_active, representation_sha, validate_replay, validate_assets, sha256
from vla_precision.gr00t_rlt.features import FrozenPI05Features
from vla_precision.gr00t_rlt.model import make_heads
from vla_precision.gr00t_rlt.replay import Replay
from test_direct_editor import editor_record, bare_policy
from test_pi05_rlt import MockPI05

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/pipette_rl/pi05_fulltask_v6d_focusvel_continuous_30hz.yaml'
OLD = ROOT/'configs/pipette_rl/pi05_fulltask_v6_focus_continuous_30hz.yaml'


def recipe(tmp_path):
    c = load_config(CONFIG, workspace=tmp_path)
    c['action_active'] = [True]*18
    c['learner']['batch_size'] = 5
    return c


def test_model_replay_and_output_identity_are_distinct(tmp_path):
    recipes = [load_config(CONFIG, f'p{i}', tmp_path) for i in range(1,6)]
    c = recipes[0]
    old = load_config(OLD, workspace=tmp_path)
    assert len({r['contract_sha256'] for r in recipes}) == 1
    assert c['contract_sha256'] != old['contract_sha256']
    assert c['output'] != old['output'] and c['state_dim'] == 65
    assert c['checkpoint'].endswith('pi05/final_model/pytorch_model.pt')
    assert bridge_profile(c).endswith('v6d_egofocusvel_native_inspire')
    assert c['scheduler'] == old['scheduler'] and c['scheduler']['action_hz'] == 30
    assert c['reward'] == old['reward'] and c['learner'] == old['learner']


class VelocityPI05(MockPI05):
    def predict_action(self, examples):
        self.seen = examples[0]['state'].copy()
        self.calls += 1
        prefix, _, _ = self.embed_prefix()
        self.vlm.forward_prefix(inputs_embeds=prefix)
        return {'normalized_actions': np.zeros((1,30,32),np.float32)}


def test_reference_gets_only_packed_velocity_actor_retains_state65(tmp_path):
    e = FrozenPI05Features.__new__(FrozenPI05Features)
    e.c, e.model = recipe(tmp_path), VelocityPI05()
    e.proc = SimpleNamespace(data_config=SimpleNamespace(serve_view_preprocess=lambda x:x),
        apply_state=lambda x:x*.123456, state_keys=['state.anchor','state.wrist_vel'],
        state_key_dims={'state.anchor':59,'state.wrist_vel':6}, state_input_keys=['state.wrist_vel'])
    raw = np.r_[np.full(59,999),[.01,.02,.03,-.04,-.05,-.06]].astype(np.float32)
    sample = {'observation.state':raw,'task':'fixture',
              'observation.images.rgb':np.zeros((4,4,3),np.uint8),'observation.images.focus':np.zeros((4,4,3),np.uint8)}
    f = e.extract(sample)
    np.testing.assert_array_equal(e.model.seen, (raw[-6:]*.123456).astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(f['state'], raw*.123456)
    assert f['state'].shape==(65,) and e.model.seen.shape==(6,)
    with pytest.raises(ValueError,match='state65'):
        e.extract({**sample,'observation.state':raw[:59]})


def test_zero_editor_preserves_reference_and_masks_with_state65(tmp_path):
    c=recipe(tmp_path);m=make_heads(c)
    z,s,ref=torch.randn(5,16384),torch.randn(5,65),torch.randn(5,30,32)
    phases=torch.arange(5)
    assert torch.equal(m.act(z,s,ref,phase=phases),ref[...,:18])
    assert m.actor[0].in_features==16384+65+30*32+5
    with torch.no_grad():
        m.actor[-1].bias.fill_(1)
    action=m.act(z,s,ref,phase=phases)
    mask=m.active_for(phases).expand(5,30,18)
    assert torch.equal(action[~mask],ref[...,:18][~mask])
    assert (action-ref[...,:18]).abs().max()<=.20001


def test_state65_replay_finite_training_publication_and_old_weight_rejection(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    c=recipe(tmp_path);db=Replay(tmp_path/'replay')
    eid=db.begin(dict(phase='p1',contract_sha256=c['contract_sha256'],token_sha256=representation_sha(c),behavior_sha256='fixture'))
    for phase in range(5):
        row=editor_record(c)
        row.update(state=np.arange(65,dtype=np.float32)/100,next_state=np.arange(65,dtype=np.float32)/100,
            phase=np.int64(phase),next_phase=np.int64(min(phase+1,4)),
            start_ns=np.int64((phase+1)*1_000_000_000),end_ns=np.int64((phase+2)*1_000_000_000),
            phase_reward_ns=np.int64((phase+2)*1_000_000_000-1 if phase<4 else 0))
        row['valid'][:]=phase_active(c,f'p{phase+1}');row['bc_eligible'][:]=row['valid']
        db.append(eid,row);db.phase_success(eid,f'p{phase+1}',int(row['end_ns'])-1)
    db.finish(eid,'success',6_000_000_001)
    dest=tmp_path/'snapshot.npz';db.export(c,representation_sha(c),dest)
    meta=json.loads(dest.with_suffix('.json').read_text())
    with np.load(dest) as f:data={k:f[k].copy() for k in f.files}
    assert data['state'].shape==(5,65)
    validate_replay(data,meta,c,token_sha256=representation_sha(c))
    with pytest.raises(ValueError,match='Invalid replay array'):
        validate_replay({**data,'state':data['state'][:,:59]},meta,c,token_sha256=representation_sha(c))
    old=load_config(OLD,workspace=tmp_path)
    with pytest.raises(ValueError,match='Replay mismatch'):
        validate_replay(data,meta,old,token_sha256=representation_sha(old))
    spec=importlib.util.spec_from_file_location('v6d_train',ROOT/'scripts/gr00t_rlt.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'check_code',lambda c:None);monkeypatch.setattr(cli,'validate_assets',lambda c:None)
    cli.fit_heads(c,SimpleNamespace(device='cpu',replay=dest,resume=None,updates=2))
    entry=json.loads((Path(c['output'])/'learner/latest.json').read_text())
    policy=bare_policy(c);policy._load_actor()
    assert policy.actor_kind=='editor_rl'
    saved=torch.load(entry['checkpoint'],weights_only=True)
    assert saved['learner']['step']==2
    # Even if pointed at the new publication, an old recipe cannot load its actor.
    old['output']=c['output']
    with pytest.raises(ValueError,match='contract mismatch'):
        bare_policy(old)._load_actor()
    db.close()


def test_asset_contract_checks_full_state_metadata_and_final_checkpoint(tmp_path):
    c=recipe(tmp_path);root=Path(c['run_root']);root.mkdir(parents=True)
    cfg=dict(framework=dict(name='PI05',discrete_state_input=True,max_state_dim=6,action_dim=32,action_horizon=30,
             paligemma_config=dict(width=2048),image_resolution=[224,224],num_inference_steps=10),
             datasets=dict(vla_data=dict(action_mode_apply_keys=['action.left_hand_rel'],
                 data_mix='unitree_g1_pipette_egofocusvel_5task447v12_pi05sq224_train_mix',
                 include_state=True,state_input_keys=['state.wrist_vel'])))
    for name in ['config.yaml','config.full.yaml']:(root/name).write_text(yaml.safe_dump(cfg))
    stats={k:dict(q01=[-1]*width,q99=[1]*width) for k,width in [('state',65),('action',32)]}
    (root/'dataset_statistics.json').write_text(json.dumps({'new_embodiment':stats}))
    weight=Path(c['checkpoint']);weight.parent.mkdir();weight.write_bytes(b'fixture')
    c['baseline']['weights_sha256']=sha256(weight)
    names=['config.yaml','config.full.yaml','dataset_statistics.json','final_model/pytorch_model.pt']
    receipt=dict(repo=c['baseline']['repo'],revision=c['baseline']['revision'],files={n:dict(bytes=(root/n).stat().st_size,sha256=sha256(root/n)) for n in names})
    (root/'download_receipt.json').write_text(json.dumps(receipt))
    assert validate_assets(c)['checkpoint']==str(weight)
    cfg['datasets']['vla_data']['state_input_keys']=['state.left_arm']
    (root/'config.full.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='six wrist velocity'):
        validate_assets(c)
