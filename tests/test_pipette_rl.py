from pathlib import Path
from dataclasses import replace
import sys, json
import numpy as np
import pytest
from vla_precision.pipette_rl.config import WORKSPACE, load_config
from vla_precision.pipette_rl.replay import Replay

sys.path.insert(0, str(WORKSPACE / "VLAPolicyBridge"))
from vla_policy_bridge.hil.replay import DualReplayStore, Transition


def obs():
    return {
        "joint_position": np.zeros(29, np.float32),
        "eef_position": np.array([0.25, -0.03, 0.08], np.float32),
        **{k: np.full((12, 16, 3), n, np.uint8) for n, k in enumerate(["rgb", "wrist_left", "wrist_right"])},
    }


def row(ep, step, *, reward=0.0, terminal=False, intervention=False, contract="test", truncated=False):
    return Transition(
        ep,
        step,
        obs(),
        obs(),
        np.array([0.001, 0, 0], np.float32),
        np.array([0.002, 0, 0], np.float32),
        reward,
        terminal,
        truncated,
        intervention,
        np.zeros(3, np.float32) if intervention else None,
        info={"rl_contract": contract, "action_hz": 30, "policy_valid": True, "commanded_action": [0, 0, -0.0003]},
    )


def config(tmp_path):
    return {
        "replay_db": str(tmp_path / "replay.sqlite3"),
        "contract_sha256": "test",
        "time_reward": -0.001,
        "success_reward": 1.0,
        "failure_reward": 0.0,
        "correction_fraction": 0.5,
        "action_hz": 30,
    }


def test_discount_override_preserves_source_and_inherits_resumed_setting(tmp_path):
    from vla_precision.pipette_rl.config import resolve_training_discount,check_discount_fork
    c={'discount':.99,'contract_sha256':'original'}
    assert resolve_training_discount(c)==.99
    assert resolve_training_discount(c,.999)==.999
    assert resolve_training_discount(c,resumed_metadata={'discount':.999})==.999
    assert resolve_training_discount(c,resumed_metadata={})==.99
    assert c=={'discount':.99,'contract_sha256':'original'}
    source=tmp_path/'old/step-1000'
    with pytest.raises(ValueError,match='separate'):
        check_discount_fork(source.parent,source,.99,.999)
    check_discount_fork(tmp_path/'new',source,.99,.999)
    check_discount_fork(source.parent,source,.99,.99)
    for bad in (0.,1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):resolve_training_discount(c,bad)


def test_reward_override_changes_training_only_and_preserves_terminal_mask(tmp_path):
    from vla_precision.pipette_rl.rewards import resolve_training_rewards,check_reward_fork
    c=config(tmp_path);before=dict(c)
    new=dict(schema_version=1,success_terminal=10.,failure_terminal=-.01,time_reward=-.01)
    legacy=resolve_training_rewards(c)
    assert legacy['success_terminal']==pytest.approx(.999)
    assert resolve_training_rewards(c,resumed_metadata={'training_rewards':new})==new
    with pytest.raises(ValueError,match='separate'):
        check_reward_fork(tmp_path/'old',tmp_path/'old/step-4000',legacy,new)
    check_reward_fork(tmp_path/'new',tmp_path/'old/step-4000',legacy,new)
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(row('success',0))
        store.append_online(row('success',1,terminal=True,reward=1.))
        store.append_online(row('failure',0,terminal=True))
    replay=Replay(c,rewards=new)
    samples=replay.sample(100,np.random.default_rng(0))
    assert {float(round(float(s['rewards'][0]),2)) for s in samples}=={10.,-.01}
    for s in samples:
        assert s['rewards'][0]==pytest.approx(10. if s['dones'] and s['episode_succeed'] else -.01)
        assert bool(s['masks'])==(not s['dones'])
    db=replay.connect()
    assert [r[0] for r in db.execute("SELECT reward FROM transitions WHERE buffer='online' ORDER BY id")]==[0.,1.,0.]
    db.close()
    assert c==before


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'10'])
def test_invalid_reward_override(value):
    from vla_precision.pipette_rl.rewards import validate_rewards
    with pytest.raises(ValueError):
        validate_rewards(dict(schema_version=1,success_terminal=value,failure_terminal=-.01,time_reward=-.01))


def tail_sampling(**overrides):
    return dict(schema_version=1,success_tail_fraction=.5,success_tail_seconds=1.,
                terminal_fraction_within_tail=.2,**overrides)


def terminal_sampling(**overrides):
    settings=dict(schema_version=4,terminal_tail_fraction=.5,terminal_tail_seconds=1.,
                  latest_collection_fraction=.5)
    settings.update(overrides)
    return settings


def test_all_terminal_tails_include_failure_without_outcome_or_length_bias(tmp_path):
    c=config(tmp_path);c['correction_fraction']=0
    with DualReplayStore(c['replay_db']) as store:
        for ep,length in [('success',180),('failure',60)]:
            for i in range(length):
                store.append_online(row(ep,i,terminal=i==length-1,
                                        reward=1. if ep=='success' and i==length-1 else 0.))
    r=Replay(c,sampling=terminal_sampling(terminal_tail_fraction=1.,latest_collection_fraction=0.))
    rows=r.sample_indices(20000,np.random.default_rng(42))
    db=r.connect()
    pos=dict(db.execute("SELECT id,step_index FROM transitions WHERE buffer='online'"));db.close()
    assert .48<sum(ep=='failure' for _,ep,_ in rows)/len(rows)<.52
    assert all(pos[idx]>=(150 if ep=='success' else 30) for idx,ep,_ in rows)
    assert .025<sum(pos[idx]==(179 if ep=='success' else 59) for idx,ep,_ in rows)/len(rows)<.045
    # Each of the 30 tail offsets, including terminal, has the same probability.
    counts=np.bincount([(179 if ep=='success' else 59)-pos[idx] for idx,ep,_ in rows],minlength=30)
    assert np.all((counts>550)&(counts<800))
    # Tail oversampling changes neither rewards nor the bootstrap mask.
    r.rewards=dict(schema_version=1,success_terminal=10.,failure_terminal=-.01,time_reward=-.01)
    for s in r.sample(100,np.random.default_rng(3)):
        assert s['rewards'][0]==pytest.approx(10. if s['episode_succeed'] and s['dones'] else -.01)
        assert bool(s['masks'])==(not bool(s['dones']))


def test_failure_only_terminal_priority_preserves_corrections_and_discard(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        for i in range(90):store.append_online(row('failure',i,terminal=i==89,intervention=i<10))
        store.append_online(row('discard',0,terminal=True));store.discard_episode('discard')
        store.append_online(row('incomplete',0))
        store.append_online(row('truncated',0,truncated=True))
        store.append_online(row('foreign',0,terminal=True,contract='foreign'))
        r=Replay(c,sampling=terminal_sampling(terminal_tail_fraction=1.))
        rng=np.random.default_rng(1)
        selected_tail=[]
        for _ in range(20):
            batch=r.sample_indices(2,rng)
            assert batch[0][2] and batch[0][1]=='failure'
            assert batch[1][1]=='failure' and not batch[1][2]
            assert batch[1] in r.index[-30:]
            selected_tail.append(batch[1][0])
        assert len(set(selected_tail))>5
        store.discard_episode('failure')
        with pytest.raises(ValueError,match='No completed'):r.sample_indices(2,rng)


def test_terminal_schema_keeps_latest_behavior_priority_and_short_episodes(tmp_path):
    c=config(tmp_path);c['correction_fraction']=0
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(row('old-success',0,terminal=True,reward=1.))
        store.append_online(row('new-failure',0,terminal=True))
    r=Replay(c,sampling=terminal_sampling(terminal_tail_fraction=1.,latest_collection_fraction=1.))
    r.refresh()
    # Inject identities only to isolate sampler behaviour from proposal decoding.
    r.closed['old-success']['collection_policy_id']='acob:old'
    r.closed['new-failure']['collection_policy_id']='acob:new'
    assert r.priority_summary()['mode']=='latest_collection_policy'
    assert r.priority_episode_ids()==['new-failure']
    assert {ep for _,ep,_ in r.sample_indices(50,np.random.default_rng(1))}=={'new-failure'}


@pytest.mark.parametrize('key,value',[('terminal_tail_fraction',1.1),('terminal_tail_seconds',0),
                                     ('terminal_fraction_within_tail',float('nan')),('latest_collection_fraction',-1)])
def test_invalid_terminal_sampling(key,value,tmp_path):
    with pytest.raises(ValueError):Replay(config(tmp_path),sampling=terminal_sampling(**{key:value}))


def test_default_sampling_is_all_outcomes_uniform_five_second_tail():
    from vla_precision.pipette_rl.sampling import load_sampling,tail_settings
    settings=load_sampling()
    assert settings['schema_version']==4
    assert tail_settings(settings)==(.5,5.,False)
    assert 'terminal_fraction_within_tail' not in settings


def test_success_tail_sampling_preserves_corrections_and_full_replay(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as s:
        for i in range(180):
            s.append_online(row('success',i,terminal=i==179,reward=1. if i==179 else 0.,intervention=i<20))
        for i in range(120):
            s.append_online(row('failure',i,terminal=i==119))
    replay=Replay(c,sampling=tail_sampling());replay.refresh()
    selected=replay.sample_indices(40000,np.random.default_rng(42))
    db=replay.connect()
    positions={r[0]:r[1] for r in db.execute("SELECT id,step_index FROM transitions WHERE buffer='online'")};db.close()
    assert all(item[2] for item in selected[:20000])
    terminal=sum(ep=='success' and positions[idx]==179 for idx,ep,_ in selected)
    near=sum(ep=='success' and positions[idx]>=150 for idx,ep,_ in selected)
    assert .04<terminal/len(selected)<.065
    assert .25<near/len(selected)<.30
    assert any(ep=='failure' for _,ep,_ in selected)
    assert any(ep=='success' and 20<=positions[idx]<150 for idx,ep,_ in selected)
    # Fractional tail allocation must also work in actual two-row minibatches.
    rng=np.random.default_rng(42)
    tiny=[replay.sample_indices(2,rng) for _ in range(100)]
    assert all(batch[0][2] for batch in tiny)
    assert any(ep=='success' and positions[idx]==179 for batch in tiny for idx,ep,_ in batch)
    # Sampling must not manufacture positive rewards on preceding tail frames.
    batch=replay.sample(100,np.random.default_rng(1))
    for item in batch:
        assert (item['rewards'][0]>0)==(item['episode_succeed'] and item['dones'])


def test_tail_rejects_discard_foreign_and_incomplete_and_handles_short_episode(tmp_path):
    c=config(tmp_path)
    settings=tail_sampling();settings['success_tail_fraction']=1.;settings['terminal_fraction_within_tail']=0.
    with DualReplayStore(c['replay_db']) as s:
        s.append_online(row('one',0,terminal=True,reward=1.))
        s.append_online(row('discard',0,terminal=True,reward=1.));s.discard_episode('discard')
        s.append_online(row('foreign',0,terminal=True,reward=1.,contract='foreign'))
        s.append_online(row('incomplete',0))
        s.append_online(row('failure',0,terminal=True))
        r=Replay(c,sampling=settings)
        assert {x[1] for x in r.sample_indices(30,np.random.default_rng(0))}=={'one'}
        s.discard_episode('one')
        assert {x[1] for x in r.sample_indices(30,np.random.default_rng(0))}=={'failure'}


def test_no_success_falls_back_to_exact_legacy_sampler(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as s:
        for i in range(10):s.append_online(row('failure',i,terminal=i==9,intervention=i<3))
    legacy=Replay(c);new=Replay(c,sampling=tail_sampling())
    assert legacy.sample_indices(100,np.random.default_rng(1))==new.sample_indices(100,np.random.default_rng(1))


def recent_sampling(**overrides):
    settings=tail_sampling()
    settings.update(schema_version=2,recent_episode_fraction=.5,recent_episode_count=2)
    settings.update(overrides)
    return settings


def test_recent_mixture_in_correction_full_and_success_tail_pools(tmp_path):
    c=config(tmp_path)
    # Insert reverse chronological order to ensure recency uses collection time,
    # not episode names, DB insertion order, or the order of refresh queries.
    with DualReplayStore(c['replay_db']) as store:
        for episode in reversed(range(10)):
            for step in range(6):
                t=row(str(episode),step,terminal=step==5,reward=float(step==5),intervention=step<2)
                t=replace(t,timestamp_ns=(episode+1)*100+step)
                store.append_online(t)
    replay=Replay(c,sampling=recent_sampling())
    selected=replay.sample_indices(20000,np.random.default_rng(13))
    assert replay.recent_episode_ids()==['9','8']
    # P(recent) = .5 + .5 * 2/10 = .6 in both halves of the batch.
    for pool in (selected[:10000],selected[10000:]):
        rate=sum(ep in {'9','8'} for _,ep,_ in pool)/len(pool)
        assert .57<rate<.63
    assert all(item[2] for item in selected[:10000])
    assert any(ep=='0' for _,ep,_ in selected)
    db=replay.connect()
    positions=dict(db.execute("SELECT id,step_index FROM transitions WHERE buffer='online'"))
    db.close()
    # Half of non-correction slots are tails (20% terminal), half are full
    # replay (1/6 terminal). Recency does not alter these quotas.
    terminal=sum(positions[idx]==5 for idx,_,_ in selected[10000:])/10000
    assert .16<terminal<.21


def test_recent_refresh_excludes_unlabelled_and_discarded(tmp_path):
    c=config(tmp_path)
    settings=recent_sampling(recent_episode_count=1,recent_episode_fraction=1.)
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(row('old',0,terminal=True,intervention=True))
        replay=Replay(c,sampling=settings)
        replay.refresh()
        store.append_online(row('new',0))
        assert {ep for _,ep,_ in replay.sample_indices(20,np.random.default_rng(0))}=={'old'}
        store.finalize_episode('new',reward=0.)
        samples=replay.sample_indices(20,np.random.default_rng(0))
        assert replay.recent_episode_ids()==['new']
        # Newest episode has no corrections: retain old correction samples.
        assert all(ep=='old' and human for _,ep,human in samples[:10])
        assert all(ep=='new' for _,ep,_ in samples[10:])
        store.discard_episode('new')
        assert {ep for _,ep,_ in replay.sample_indices(20,np.random.default_rng(0))}=={'old'}
        assert replay.recent_episode_ids()==['old']


def test_zero_recency_is_exact_tail_sampler(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        for ep in range(3):
            for step in range(3):
                store.append_online(row(str(ep),step,terminal=step==2,reward=float(step==2)))
    old=Replay(c,sampling=tail_sampling())
    new=Replay(c,sampling=recent_sampling(recent_episode_fraction=0.))
    assert old.sample_indices(200,np.random.default_rng(2))==new.sample_indices(200,np.random.default_rng(2))


@pytest.mark.parametrize('key,value',[('recent_episode_fraction',1.1),('recent_episode_fraction',float('nan')),
                                     ('recent_episode_count',0),('recent_episode_count',1.5),
                                     ('recent_episode_count',True)])
def test_recent_configuration_is_validated(tmp_path,key,value):
    with pytest.raises(ValueError):Replay(config(tmp_path),sampling=recent_sampling(**{key:value}))


def collection_sampling(**overrides):
    settings=tail_sampling()
    settings.update(schema_version=3,latest_collection_fraction=.5)
    settings.update(overrides)
    return settings


def add_collection_episode(store, ep, time, digest, *, kind='online', foreign=False,
                           matching_key=True, adapter_path='learner/step-00001000'):
    from vla_policy_bridge.hil.policy_proposals import connect_writer, write_proposal
    for step in range(3):
        t=row(ep,step,terminal=step==2,reward=float(step==2),intervention=step==0)
        t=replace(t,timestamp_ns=time+step)
        t.info['policy_observation_key']=f'online:{ep}-{step}'
        store.append_online(t)
    if digest=='missing':
        return
    model=dict(source='live_inference',mode='hil_policy',checkpoint='baseline/88612',
               acob_state_sha256=digest,acob_checkpoint=adapter_path if digest else None)
    db=connect_writer(store.path)
    write_proposal(db,key=f'online:{ep}-0' if matching_key else f'online:wrong-{ep}',
                   episode=ep,step=0,contract='foreign' if foreign else 'test',kind=kind,
                   action=np.zeros(3,np.float32),observation=b'index-test-only',model=model)
    db.close()


def test_collection_priority_groups_all_matching_weights_across_sessions(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        # The two RL checkpoints have the same step number but different hashes.
        # The latest hash is also present in an earlier collection session.
        for i in range(12):
            digest='a'*64 if i<6 else 'b'*64
            add_collection_episode(store,str(i),100*i+1,digest,
                                   adapter_path=f'/session-{i}/step-00001000')
        add_collection_episode(store,'earlier-b',10,'b'*64)
    replay=Replay(c,sampling=collection_sampling())
    samples=replay.sample_indices(12000,np.random.default_rng(7))
    expected={str(i) for i in range(6,12)}|{'earlier-b'}
    assert set(replay.priority_episode_ids())==expected  # Seven, not latest five.
    assert replay.latest_collection_policy_id()=='acob:'+'b'*64
    # 50% priority plus 50% full replay, where the collection is 7/13.
    for pool in (samples[:6000],samples[6000:]):
        rate=sum(ep in expected for _,ep,_ in pool)/len(pool)
        assert .74<rate<.80
    assert all(human for _,_,human in samples[:6000])
    assert any(ep=='0' for _,ep,_ in samples)


@pytest.mark.parametrize('kwargs',[
    {'digest':'missing'}, {'digest':'b'*64,'kind':'offline'},
    {'digest':'b'*64,'foreign':True}, {'digest':'b'*64,'matching_key':False},
])
def test_unknown_latest_collection_does_not_select_an_older_policy(tmp_path,kwargs):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        add_collection_episode(store,'old',1,'a'*64)
        add_collection_episode(store,'new',100,**kwargs)
    replay=Replay(c,sampling=collection_sampling())
    samples=replay.sample_indices(2000,np.random.default_rng(2))
    assert replay.latest_collection_policy_id() is None
    assert replay.priority_episode_ids()==[]
    assert .45<sum(ep=='new' for _,ep,_ in samples)/len(samples)<.55


def test_collection_changes_on_new_policy_and_discard(tmp_path):
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        add_collection_episode(store,'bc',1,None)
        replay=Replay(c,sampling=collection_sampling())
        replay.refresh()
        assert replay.latest_collection_policy_id()=='bc:test'
        add_collection_episode(store,'rl',100,'a'*64)
        replay.refresh()
        assert replay.priority_episode_ids()==['rl']
        store.discard_episode('rl')
        replay.refresh()
        assert replay.priority_episode_ids()==['bc']


def test_collection_late_proposal_and_mixed_policy(tmp_path):
    from vla_policy_bridge.hil.policy_proposals import connect_writer,write_proposal
    c=config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        add_collection_episode(store,'new',1,'missing')
        replay=Replay(c,sampling=collection_sampling())
        replay.refresh()
        assert replay.priority_episode_ids()==[]
        db=connect_writer(store.path)
        for step,digest in [(0,'a'*64),(1,'b'*64)]:
            write_proposal(db,key=f'online:new-{step}',episode='new',step=step,contract='test',kind='online',
                           action=np.zeros(3,np.float32),observation=b'index-test-only',
                           model=dict(source='live_inference',mode='hil_policy',acob_state_sha256=digest))
            replay.refresh()
            assert replay.priority_episode_ids()==(['new'] if step==0 else [])
        db.close()


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),True])
def test_collection_fraction_validation(tmp_path,value):
    with pytest.raises(ValueError):Replay(config(tmp_path),sampling=collection_sampling(latest_collection_fraction=value))


@pytest.mark.parametrize('key,value',[('success_tail_fraction',1.1),('success_tail_seconds',0),
                                     ('terminal_fraction_within_tail',-1),('success_tail_seconds',float('nan'))])
def test_tail_configuration_is_validated(tmp_path,key,value):
    settings=tail_sampling();settings[key]=value
    with pytest.raises(ValueError):Replay(config(tmp_path),sampling=settings)


def test_completed_labels_and_actual_command(tmp_path):
    c = config(tmp_path)
    with DualReplayStore(c["replay_db"]) as store:
        store.append_online(row("success", 0, intervention=True))
        # Same route as Success after GUI PAUSE: no pending row is needed.
        assert store.finalize_episode("success", reward=1.0)
        assert not store.finalize_episode("success", reward=0.0)
        store.append_online(row("failure", 0, terminal=True))
        store.append_online(row("open", 0))
        store.append_online(row("foreign", 0, terminal=True, contract="absolute-run"))
        store.append_online(row("abort", 0, truncated=True))
        store.append_online(row("gap", 2, terminal=True))
        assert store.counts() == {"demo": 1, "online": 6}
    replay = Replay(c)
    replay.refresh()
    replay.refresh()
    assert replay.summary()["episodes"] == 2 and len(replay.index) == 2
    assert len(replay.rejected) == 3
    rows = replay.sample(100, np.random.default_rng(42))
    assert any(not r["episode_succeed"] for r in rows)
    for r in rows:
        np.testing.assert_allclose(r["actions"], [[0, 0, -0.0003]])
        assert r["masks"] == 0 and r["dones"]
        assert r["rewards"][0] == pytest.approx(0.999 if r["episode_succeed"] else -0.001)
        assert r["observations"]["state"].shape == (1, 32)
        assert r["observations"]["base_0_rgb"].shape == (1, 224, 224, 3)


def test_no_feedback_without_execution(tmp_path):
    with DualReplayStore(tmp_path / "r.db") as s:
        assert not s.finalize_episode("empty", reward=1.0)


def test_release_hold_keeps_terminal_td_but_is_not_demo_or_success_bc(tmp_path):
    from vla_precision.pipette_rl.model import make_batch
    import jax, jax.numpy as jnp
    from vla_precision.acob.agent import _eligible_bc_loss
    c = config(tmp_path)
    t = row('success', 0, terminal=True, reward=1, intervention=False)
    t.info.update(training_bc_eligible=False, release_debounce_active=True,
                  raw_intervention=False, control_intervention=True, commanded_action=[0, 0, 0])
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(t)
        assert store.counts() == {'online': 1, 'demo': 0}
    sample = Replay(c).sample(1, np.random.default_rng(0))[0]
    assert sample['episode_succeed'] and sample['dones']
    assert sample['rewards'][0] == pytest.approx(.999)
    assert not sample['intervened'] and not sample['bc_intervened'] and not sample['bc_eligible']
    class Cache:
        def get(self, observation):
            return np.zeros((1, 2), np.float32), np.ones(1, bool), np.zeros(1, np.int32)
    batch = make_batch([sample], Cache())
    assert not bool(batch['bc_eligible'][0])
    # Even an eventual success must supply no BC gradient for release HOLD.
    objective = lambda losses: _eligible_bc_loss(losses, jnp.ones(2), jnp.array([False, True]))[0]
    assert float(objective(jnp.array([1000., 2.]))) == 2.
    np.testing.assert_array_equal(jax.grad(objective)(jnp.array([1000., 2.])), [0., 1.])
    empty = _eligible_bc_loss(jnp.array([1000.]), jnp.ones(1), batch['bc_eligible'])[0]
    assert float(empty) == 0


def test_real_relative_contract():
    c = load_config()
    assert c["credit_horizon"] == 1 and c["model_horizon"] == 10
    assert "absxyz" not in c["baseline"]


def test_human_without_policy_is_bc_but_not_preference(tmp_path):
    c = config(tmp_path)
    transition = row("human", 0, terminal=True, intervention=True)
    transition.info["policy_valid"] = False
    with DualReplayStore(c["replay_db"]) as store:
        store.append_online(transition)
    replay = Replay(c)
    replay.refresh()
    sample = replay.sample(1, np.random.default_rng(0))[0]
    assert sample["bc_intervened"] and not sample["intervened"]
    assert not sample["episode_succeed"]


@pytest.mark.parametrize('arrives_first', [False, True])
def test_late_proposal_enables_preference_without_rewriting_execution(tmp_path, arrives_first):
    from vla_policy_bridge.hil.policy_proposals import connect_writer, write_proposal, offline_key
    from vla_policy_bridge.hil.replay import _observation_blob
    c = config(tmp_path)
    transition = row('human', 0, terminal=True, intervention=True)
    transition.info['policy_valid'] = False
    with DualReplayStore(c['replay_db']) as store:
        db = connect_writer(c['replay_db'])
        blob = _observation_blob(transition.observation)

        def annotate():
            write_proposal(db, key=offline_key('test', 'human', 0, blob), episode='human', step=0,
                           contract='test', kind='offline', action=[-.001, 0, 0], observation=blob,
                           model={'source': 'offline_baseline_counterfactual'})

        if arrives_first:
            annotate()
        store.append_online(transition)
        replay = Replay(c)
        replay.refresh()
        original = db.execute('SELECT * FROM transitions').fetchall()
        if not arrives_first:
            assert not replay.sample(1, np.random.default_rng(0))[0]['intervened']
            annotate()
        sample = replay.sample(1, np.random.default_rng(0))[0]
        assert sample['bc_intervened'] and sample['intervened']
        np.testing.assert_allclose(sample['intervention_bad_actions'], [[-.001, 0, 0]])
        np.testing.assert_allclose(sample['actions'], [[0, 0, -.0003]])
        assert original == db.execute('SELECT * FROM transitions').fetchall()
        store.discard_episode('human')
        with pytest.raises(ValueError, match='No completed'):
            replay.sample(1, np.random.default_rng(0))
        db.close()


def test_direct_proposal_takes_precedence_over_offline_counterfactual(tmp_path):
    from vla_policy_bridge.hil.policy_proposals import connect_writer, write_proposal, offline_key
    c = config(tmp_path)
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(row('human', 0, terminal=True, intervention=True))
    db = connect_writer(c['replay_db'])
    blob = db.execute('SELECT payload FROM observations WHERE step_index=0').fetchone()[0]
    write_proposal(db, key=offline_key('test', 'human', 0, blob), episode='human', step=0,
                   contract='test', kind='offline', action=[-.001, 0, 0], observation=blob, model={})
    db.close()
    sample = Replay(c).sample(1, np.random.default_rng(0))[0]
    np.testing.assert_allclose(sample['intervention_bad_actions'], [[.001, 0, 0]])


def test_online_pair_uses_inference_state_even_after_legacy_observation_dedup(tmp_path):
    import cv2, msgpack
    from vla_policy_bridge.hil.policy_proposals import connect_writer, write_proposal, online_key
    c = config(tmp_path)
    # A previous held next-observation can occupy the legacy (episode,step) row.
    # Pairing must condition on the later observation actually sent for inference.
    source = obs()
    source['eef_position'][0] = .27
    packet = dict(actor_owner='actor', policy_owner='policy', episode_id='human', observation_seq=17,
                  step_index=0, action_hz=30, action_frame='pelvis',
                  state={k: source[k].tolist() for k in ('joint_position', 'eef_position')},
                  images_jpeg={v: cv2.imencode('.jpg', source[v])[1].tobytes()
                               for v in ('rgb', 'wrist_left', 'wrist_right')})
    t = row('human', 0, terminal=True, intervention=True)
    t.info.update(policy_valid=False, policy_observation_key=online_key(packet))
    with DualReplayStore(c['replay_db']) as store:
        store.append_online(t)
    db = connect_writer(c['replay_db'])
    original = db.execute('SELECT payload FROM observations WHERE step_index=0').fetchone()[0]
    write_proposal(db, key=online_key(packet), episode='human', step=0, contract='test', kind='online',
                   action=[-.001, 0, 0], observation=msgpack.packb(packet, use_bin_type=True), model={})
    sample = Replay(c).sample(1, np.random.default_rng(0))[0]
    assert sample['intervened']
    assert sample['observations']['state'][0, 29] == pytest.approx(.27)
    assert original == db.execute('SELECT payload FROM observations WHERE step_index=0').fetchone()[0]
    db.close()


def test_discard_revokes_cached_episode_and_both_buffers(tmp_path):
    c = config(tmp_path)
    with DualReplayStore(c["replay_db"]) as store:
        store.append_online(row("bad", 0, terminal=True, intervention=True))
        store.append_online(row("good", 0, terminal=True))
        replay = Replay(c)
        replay.refresh()
        assert set(replay.closed) == {"good", "bad"}
        cached_ids = store._ids("demo")
        store.discard_episode("bad")
        store.discard_episode("bad")  # repeated disposition is harmless
        assert not store.finalize_episode("bad", reward=1)
        assert store.counts() == {"demo": 1, "online": 2}  # audit retained
        assert store.behavior_index("demo") == []
        with pytest.raises(KeyError):
            store.load_behavior(cached_ids)
        with pytest.raises(KeyError):
            store._load(cached_ids)
        samples = replay.sample(4, np.random.default_rng(0))
        assert {r["episode_id"] for r in samples} == {"good"}
        assert "bad" not in replay.closed and "bad" in replay.discarded


def test_discard_before_pending_flush_excludes_late_rows(tmp_path):
    c = config(tmp_path)
    with DualReplayStore(c["replay_db"]) as store:
        store.discard_episode("bad")
        store.append_online(row("bad", 0, terminal=True, intervention=True))
        assert store._ids("demo") == store._ids("online") == []
    replay = Replay(c)
    replay.refresh()
    assert not replay.index
    with pytest.raises(ValueError, match="No completed"):
        replay.sample(1, np.random.default_rng(0))


def test_remote_path_mapping_preserves_identity_but_rejects_semantic_changes():
    from vla_precision.pipette_rl.config import bind_source_contract
    source = load_config()
    remote = dict(source, baseline="/cluster/model", output="/cluster/output", replay_db="/cluster/replay")
    mapped = bind_source_contract(remote, source)
    assert mapped["baseline"] == "/cluster/model"
    assert mapped["checkpoint_baseline"] == source["baseline"]
    assert mapped["contract_sha256"] == source["contract_sha256"]
    remote = dict(source, discount=0.5)
    with pytest.raises(ValueError, match="beyond paths"):
        bind_source_contract(remote, source)
    with pytest.raises(ValueError, match="checksum"):
        bind_source_contract(dict(source), dict(source, discount=0.5))


def test_local_training_refuses_empty_replay_without_waiting_or_creating_output(tmp_path):
    import os
    import subprocess
    import yaml
    from vla_precision.pipette_rl.config import DEFAULT_CONFIG

    c = yaml.safe_load(DEFAULT_CONFIG.read_text())
    c.update(replay_db=str(tmp_path / "empty.sqlite3"), output=str(tmp_path / "output"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    result = subprocess.run(
        [sys.executable, str(WORKSPACE / "VLA-Precision/scripts/pipette_rl.py"),
         "train", "--config", str(path), "--require-ready"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "found 0. Collect and label Success/Failure first" in result.stderr
    assert not (tmp_path / "output").exists()
