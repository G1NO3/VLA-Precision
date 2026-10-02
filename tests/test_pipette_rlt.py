"""Physical units, human waiting masks, and terminal chunk boundaries for RLT."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from vla_precision.pipette_rlt.core import bc_targets,chunk_return,replay_chunks,sample_indices,load_config


def test_human_wait_not_demonstration_but_intentional_hold_is():
    ref=np.full((4,3),.0005)
    action=np.array([[0,0,0],[0,0,-.0005],[0,0,0],[0,0,0]])
    target,w=bc_targets(ref,action,[True,True,False,True],[True]*4,intentional_hold=[False,False,False,True])
    np.testing.assert_array_equal(w,[False,True,True,True])
    np.testing.assert_allclose(target[1],action[1])
    np.testing.assert_allclose(target[2],ref[2])


def test_release_hold_and_padding_excluded_even_with_motion():
    _,w=bc_targets(np.ones((2,3)),np.ones((2,3)),[False,True],[False,False])
    assert not w.any()


def test_discount_is_primitive_step_discount():
    assert chunk_return([-.01]*4+[10],.999)==pytest.approx(sum(-.01*.999**i for i in range(4))+10*.999**4)


def db_fixture(tmp_path,*,human=None,gap=False,discard=False,success=True,terminal=True):
    path=tmp_path/'replay.sqlite3'
    c=sqlite3.connect(path)
    c.execute('create table transitions (id,buffer,episode_id,step_index,observation_id,next_observation_id,intervention,reward,terminated,truncated,timestamp_ns,info_json)')
    c.execute('create table discarded_episodes (episode_id)')
    for j in range(8):
        hu=bool(human and human[j]);stamp=j*33333333+(1000000000 if gap and j>=4 else 0)
        info=dict(commanded_action=[0,0,-.0001],source='human' if hu else 'policy',rl_contract='contract',action_hz=30)
        c.execute('insert into transitions values (?,?,?,?,?,?,?,?,?,?,?,?)',(j,'online','ep',j,j,j+1,hu,1 if j==7 and success else 0,int(j==7 and terminal),0,stamp,json.dumps(info)))
    if discard:c.execute("insert into discarded_episodes values ('ep')")
    c.commit();c.close()
    return dict(source_replay=str(path),source_contract='contract',replay_db=str(path),contract_sha256='contract',
        credit_horizon=5,action_hz=30,max_recording_gap_s=.15,gamma=.999,success_reward=10,failure_reward=-.01,time_reward=-.01)


def test_terminal_partial_chunk_is_masked_and_not_bootstrapped(tmp_path):
    chunks,_=replay_chunks(db_fixture(tmp_path))
    assert [x['length'] for x in chunks]==[5,3]
    assert chunks[0]['discount']==pytest.approx(.999**5)
    assert chunks[1]['discount']==0
    assert chunks[1]['next_observation_id']==8
    np.testing.assert_array_equal(chunks[1]['valid'],[1,1,1,0,0])
    np.testing.assert_allclose(chunks[1]['actions_m'][3:],0)
    assert chunks[1]['reward']==pytest.approx(-.01-.01*.999+10*.999**2)


@pytest.mark.parametrize('kind',['human','gap'])
def test_chunks_do_not_cross_takeover_or_recording_gap(tmp_path,kind):
    c=db_fixture(tmp_path,human=[False]*4+[True]*4 if kind=='human' else None,gap=kind=='gap')
    chunks,_=replay_chunks(c)
    assert len(chunks)==1 and chunks[0]['step']==4 and chunks[0]['length']==4


@pytest.mark.parametrize('kw',[{'discard':True},{'terminal':False}])
def test_discard_and_unlabelled_excluded(tmp_path,kw):
    chunks,eps=replay_chunks(db_fixture(tmp_path,**kw))
    assert not chunks and not eps


def test_failed_terminal_cost_and_mask(tmp_path):
    chunks,_=replay_chunks(db_fixture(tmp_path,success=False))
    assert not chunks[-1]['success'] and chunks[-1]['discount']==0
    assert chunks[-1]['reward']==pytest.approx(chunk_return([-.01]*3,.999))


def test_tail_sampler_has_no_success_filter():
    c=dict(tail_seconds=5,action_hz=30,tail_fraction=1,latest_collection_fraction=0,replay_db='new')
    chunks=[dict(steps_to_terminal=0,success=s,db='old') for s in [False,True]]
    sampled=sample_indices(chunks,1000,np.random.default_rng(42),c)
    assert .4<np.mean(sampled)<.6


def test_real_config_keeps_measured_joints_and_relative_actions():
    c=load_config()
    assert c['state_dim']==32 and c['action_representation']=='delta_xyz'
    assert c['credit_horizon']==5 and c['gamma']==.999
    assert c['replay_db']!=c['source_replay']


def test_upstream_token_reconstruction_and_actor_gradients():
    import torch
    from vla_precision.pipette_rlt.model import token_model,Heads
    torch.set_num_threads(1)
    c=load_config();c.update(token_input_dim=8,token_dim=8,token_prefix_len=6,token_layers=1,token_heads=2,token_mlp_ratio=2)
    token=token_model(c)
    prefix=torch.randn(2,6,8);mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1]],dtype=torch.bool)
    loss,info=token.loss(prefix,mask);loss.backward()
    assert torch.isfinite(loss) and token.encoder.rl_token_embed.grad.abs().sum()>0
    heads=Heads(c);z=info['z_rl'].detach();p=torch.zeros(2,32);ref=torch.zeros(2,10,3)
    actions=heads.act(z,p,ref)
    assert actions.shape==(2,5,3) and actions.abs().max()<=1
    q=heads.q(z,p,actions);assert q.shape==(2,2)
    q.mean().backward();assert heads.actor[-1].weight.grad.abs().sum()>0
