"""Direct actor editor: CPU-only; no VLM, cameras, ports or robot commands."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import EDITOR, FOCUS, load_config, representation_sha, sha256, validate_replay
from vla_precision.gr00t_rlt.model import EditorHeads, Learner, spatial_features, token_model
from vla_precision.gr00t_rlt.replay import Replay
from vla_precision.gr00t_rlt.serving import LivePolicy
from vla_precision.gr00t_rlt.act_context import CONTEXT_DIM
from test_gr00t_rlt_live import record


def recipe(tmp_path):
    c = load_config(EDITOR, workspace=tmp_path)
    c['action_active'] = [i not in c['held_action_indices'] for i in range(18)]
    c['learner']['batch_size'] = 2
    return c


def test_contract_has_independent_identity_and_same_reward(tmp_path):
    for phase in ('p1', 'p2', 'p3', 'p4', 'p5'):
        c, rlt = load_config(EDITOR, phase, tmp_path), load_config(FOCUS, phase, tmp_path)
        assert c['contract_sha256'] != rlt['contract_sha256']
        assert c['output'] != rlt['output']
        assert c['reward'] == rlt['reward']
        assert c['held_action_indices'] == rlt['held_action_indices']
        assert representation_sha(c).startswith('editor_pool:')
        with pytest.raises(ValueError, match='not a learned token'):
            token_model(c)


def test_spatial_pool_keeps_views_quadrants_and_ignores_padding():
    prefix = torch.full((1, 800, 2048), 999., requires_grad=True)
    mask = torch.zeros(1, 800, dtype=torch.bool)
    mask[:, :512] = True
    with torch.no_grad():
        grid = prefix[:, :512].reshape(1, 2, 16, 16, 2048)
        for view in range(2):
            for y in range(2):
                for x in range(2):
                    grid[:, view, y*8:(y+1)*8, x*8:(x+1)*8] = view*4 + y*2 + x
    result = spatial_features(prefix, mask)
    assert result.shape == (1, 16384) and not result.requires_grad
    torch.testing.assert_close(result.reshape(8, 2048)[:, 0], torch.arange(8.).float())
    assert (result < 999).all()
    mask[:, 512] = True
    with pytest.raises(ValueError, match='masked third'):
        spatial_features(prefix, mask)
    mask[:, 512] = False
    mask[:, 256] = False
    with pytest.raises(ValueError, match='two real views'):
        spatial_features(prefix, mask)


def test_zero_parity_bounds_smoothing_and_actor_gradient(tmp_path):
    c = recipe(tmp_path)
    model = EditorHeads(c)
    z, s = torch.randn(2, 16384), torch.randn(2, 59)
    ref = torch.randn(2, 30, 32)*7  # outside old absolute actor clamp
    assert torch.equal(model.act(z, s, ref), ref[..., :18])
    loss = (model.act(z, s, ref) - ref[..., :18] - .1).square().mean()
    loss.backward()
    assert model.actor[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.actor[-1].bias.fill_(99.)
        action = model.act(z, s, ref)
        delta = action-ref[..., :18]
        assert torch.all(delta.abs() < .20001)
        assert torch.equal(action[..., ~model.active], ref[..., :18][..., ~model.active])
        smooth = model.smooth_action(action, ref, torch.ones_like(action)*100)
        assert (smooth-ref[..., :18]).abs().max() < .20001
        assert torch.equal(smooth[..., ~model.active], ref[..., :18][..., ~model.active])
    with pytest.raises(ValueError, match='drop'):
        model.act(z, s, ref, dropout=.5)


def editor_record(c):
    row = record(c)
    row.update(token=np.ones(16384, np.float32), next_token=np.ones(16384, np.float32))
    row.update(scheduler_context=np.zeros(CONTEXT_DIM,np.float32), next_scheduler_context=np.zeros(CONTEXT_DIM,np.float32),
               proposal_accepted=np.bool_(True), autonomy=np.bool_(True), next_autonomy=np.bool_(True))
    return row


def test_human_bc_updates_only_actor_and_reports_all_losses(tmp_path):
    c = recipe(tmp_path)
    row = editor_record(c)
    row['human'][:] = row['valid']
    row['td_eligible'] = np.bool_(False)
    row['autonomy'] = np.bool_(False)
    row['action'][row['valid']] = .1
    batch = {k: torch.as_tensor(np.stack([v, v])) for k,v in row.items() if k not in ('start_ns', 'end_ns')}
    batch.update(reward=torch.ones(2), discount=torch.zeros(2))
    learner = Learner(c)
    before = {k:v.clone() for k,v in learner.model.critics.state_dict().items()}
    learner.update(batch)
    m = learner.update(batch)
    assert m['td_samples'] == m['critic_updates'] == 0
    assert m['bc_loss'] > 0 and m['weighted_rl_loss'] == m['ref_loss'] == 0
    assert 'editor_delta_mean_abs' in m and 'editor_saturation_fraction' in m
    assert all(torch.equal(v, learner.model.critics.state_dict()[k]) for k,v in before.items())
    assert learner.model.actor[-1].weight.abs().sum() > 0


def test_critic_uses_current_and_next_context_not_actor(tmp_path):
    c = recipe(tmp_path)
    learner = Learner(c)
    row = editor_record(c)
    batch = {k:torch.as_tensor(np.stack([v,v])) for k,v in row.items() if k not in ('start_ns','end_ns')}
    batch.update(reward=torch.ones(2),discount=torch.ones(2)*.99)
    args = [batch[k] for k in ('token','state','reference','action','valid')]
    with pytest.raises(ValueError,match='scheduler context'):
        learner.model.q(*args)
    q0 = learner.model.q(*args,batch['scheduler_context'])
    q1 = learner.model.q(*args,torch.randn(2,CONTEXT_DIM))
    assert not torch.equal(q0,q1)
    before = learner.model.critics[0].context_encoder[1].bias.detach().clone()
    learner.update(batch)
    assert not torch.equal(before,learner.model.critics[0].context_encoder[1].bias)
    # Frozen-reference zero actor needs no scheduler context at serving time.
    assert torch.equal(learner.model.act(batch['token'],batch['state'],batch['reference']),batch['reference'][...,:18])


def bare_policy(c):
    p = LivePolicy.__new__(LivePolicy)
    p.c, p.device, p.token_sha = c, 'cpu', representation_sha(c)
    return p


def test_replay_train_resume_publication_and_discard_without_token(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    c = recipe(tmp_path)
    cli_path = Path(__file__).resolve().parents[1] / 'scripts/gr00t_rlt.py'
    spec = importlib.util.spec_from_file_location('editor_test_cli', cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'check_code', lambda c: None)
    monkeypatch.setattr(cli, 'validate_assets', lambda c: None)
    policy = bare_policy(c)
    policy._load_actor()
    assert policy.actor_kind == 'editor_zero'
    db = Replay(tmp_path/'replay')
    eid = db.begin(dict(phase='p1', contract_sha256=c['contract_sha256'], token_sha256=representation_sha(c),
                        behavior_sha256=policy.behavior))
    db.append(eid, editor_record(c))
    db.finish(eid, 'success', 1_000_000_001)
    snapshot = tmp_path/'snapshot.npz'
    assert db.export(c, representation_sha(c), snapshot)['transitions'] == 1
    metadata=json.loads(snapshot.with_suffix('.json').read_text())
    with np.load(snapshot) as f:
        data={k:f[k].copy() for k in f.files}
    assert metadata['collector']=='bridge_async_act_v1'
    with pytest.raises(ValueError,match='proposal-level'):
        validate_replay(data,{**metadata,'collector':'bridge_sequential_v1'},c,token_sha256=representation_sha(c))
    with pytest.raises(ValueError,match='Invalid replay array'):
        validate_replay({**data,'scheduler_context':np.zeros((1,256))},metadata,c,token_sha256=representation_sha(c))
    with pytest.raises(ValueError,match='Cancelled/intervened'):
        validate_replay({**data,'proposal_accepted':np.zeros(1,bool)},metadata,c,token_sha256=representation_sha(c))
    with pytest.raises(ValueError, match='No complete'):
        db.export(c, 'wrong-token', tmp_path/'wrong.npz')
    args = SimpleNamespace(device='cpu', replay=snapshot, resume=None, updates=2)
    cli.fit_heads(c, args)
    output = Path(c['output'])
    assert not (output/'token').exists() and not (output/'bc').exists()
    entry = json.loads((output/'learner/latest.json').read_text())
    assert entry['sha256'] == sha256(entry['checkpoint'])
    policy._load_actor()
    assert policy.actor_kind == 'editor_rl'
    args.resume = Path(entry['checkpoint'])
    cli.fit_heads(c, args)
    latest = json.loads((output/'learner/latest.json').read_text())
    saved = torch.load(latest['checkpoint'], weights_only=True)
    assert saved['learner']['step'] == 4
    assert saved['learner']['critic_updates'] == 4
    db.discard(eid)
    with pytest.raises(ValueError, match='No complete'):
        db.export(c, representation_sha(c), tmp_path/'discard.npz')
    db.close()
    (output/'learner/latest.json').unlink()
    with pytest.raises(ValueError, match='publication missing'):
        policy._load_actor()
    with pytest.raises(ValueError, match='publication missing'):
        bare_policy(c)._load_actor()


def test_operator_selects_served_round_base_or_latest(tmp_path, monkeypatch):
    """GUI 'Serve model': <run>/actor-select.json picks a published round or the
    zero-residual base; absent/latest follows latest.json. The learner's
    resume point (latest.json) is untouched by the choice."""
    from vla_precision.gr00t_rlt.model import make_heads
    from vla_precision.gr00t_rlt.serving import SELECT_FILE
    torch.set_num_threads(1)
    c = recipe(tmp_path)
    cli_path = Path(__file__).resolve().parents[1] / 'scripts/gr00t_rlt.py'
    spec = importlib.util.spec_from_file_location('editor_select_cli', cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'check_code', lambda c: None)
    monkeypatch.setattr(cli, 'validate_assets', lambda c: None)
    db = Replay(tmp_path/'replay')
    eid = db.begin(dict(phase='p1', contract_sha256=c['contract_sha256'], token_sha256=representation_sha(c),
                        behavior_sha256='editor_zero:'+c['contract_sha256']))
    db.append(eid, editor_record(c))
    db.finish(eid, 'success', 1_000_000_001)
    snapshot = tmp_path/'snapshot.npz'
    db.export(c, representation_sha(c), snapshot)
    db.close()
    learner = Path(c['output'])/'learner'
    args = SimpleNamespace(device='cpu', replay=snapshot, resume=None, updates=2)
    cli.fit_heads(c, args)                                   # round 1 -> step 2
    args.resume = learner/'step-00000002.pt'
    cli.fit_heads(c, args)                                   # round 2 -> step 4
    round1 = json.loads((learner/'step-00000002.json').read_text())
    assert (round1['step'], round1['start_step'], round1['updates']) == (2, 0, 2)
    assert (round1['episodes'], round1['transitions']) == (1, 1)
    assert round1['sha256'] == sha256(learner/'step-00000002.pt')
    assert json.loads((learner/'step-00000004.json').read_text())['start_step'] == 2

    run = tmp_path/'run'
    run.mkdir()
    policy = bare_policy(c)
    policy.run = run
    policy._load_actor()                                     # no choice file: newest round
    assert (Path(policy.actor_path).name, policy.selection) == ('step-00000004.pt', 'latest')
    (run/SELECT_FILE).write_text(json.dumps(dict(select='checkpoint', checkpoint=str(learner/'step-00000002.pt'),
                                                 sha256=round1['sha256'])))
    policy._load_actor('rollout-a')
    assert (Path(policy.actor_path).name, policy.actor_kind, policy.selection) == ('step-00000002.pt', 'editor_rl', 'checkpoint')
    assert policy.behavior == round1['sha256']
    assert json.loads((run/'live-actor.json').read_text())['selection'] == 'checkpoint'
    (run/SELECT_FILE).write_text(json.dumps(dict(select='base')))
    policy._load_actor('rollout-b')
    assert (policy.actor_kind, policy.behavior) == ('editor_zero', 'editor_zero:'+c['contract_sha256'])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(c['seed'])
        fresh = make_heads(c).state_dict()
    assert all(torch.equal(v, fresh[k]) for k, v in policy.heads.state_dict().items())   # RL weights dropped
    (run/SELECT_FILE).write_text(json.dumps(dict(select='latest')))
    policy._load_actor('rollout-c')
    assert Path(policy.actor_path).name == 'step-00000004.pt'
    assert Path(json.loads((learner/'latest.json').read_text())['checkpoint']).name == 'step-00000004.pt'
    (run/SELECT_FILE).write_text(json.dumps(dict(select='checkpoint', checkpoint=str(learner/'step-00000002.pt'),
                                                 sha256='wrong')))
    with pytest.raises(ValueError, match='Invalid selected editor checkpoint'):
        policy._load_actor('rollout-d')
    for bad in ('{broken', json.dumps(dict(select='newest')), json.dumps(dict(select='checkpoint'))):
        (run/SELECT_FILE).write_text(bad)
        with pytest.raises(ValueError):
            policy._load_actor('rollout-e')


def test_corrupt_publication_never_silently_initializes_zero(tmp_path):
    c = recipe(tmp_path)
    root = Path(c['output'])/'learner'
    root.mkdir(parents=True)
    (root/'latest.json').write_text('{broken')
    with pytest.raises(ValueError):
        bare_policy(c)._load_actor()
    path = root/'bad.pt'
    path.write_bytes(b'not a checkpoint')
    (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(path),sha256='wrong')))
    with pytest.raises(ValueError,match='Invalid published editor'):
        bare_policy(c)._load_actor()


def test_real_serving_method_uses_fixed_features_and_preserves_joint_seeds(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'VLAPolicyBridge'))
    from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS
    c = recipe(tmp_path)
    policy = bare_policy(c)
    policy._load_actor()
    policy.episode, policy.capture_root = None, None
    policy.stats = dict(q01=[-1]*32, q99=[1]*32)
    ref = np.arange(30*32, dtype=np.float32).reshape(30,32)/100
    mask = np.zeros(800, bool)
    mask[:512] = True
    feature = dict(prefix=np.ones((800,2048),np.float16),mask=mask,state=np.zeros(59,np.float32),reference=ref)
    policy.features = SimpleNamespace(extract=lambda sample:feature,
        physical_actions=lambda action,reference:np.concatenate([action,reference[:,18:]],axis=-1))
    obs = synthetic_observation()
    obs['language']['annotation.human.task_description'] = [[INSTRUCTIONS[0]]]
    result = policy.get_action(obs, {'episode':'test-first'})
    meta = result['info']['rlt']
    assert meta['actor_kind'] == 'editor_zero'
    assert meta['token'].shape == (16384,)
    np.testing.assert_array_equal(meta['action'],ref[:,:18])
    np.testing.assert_array_equal(result['left_arm_abs'][0],ref[:,18:25])
    np.testing.assert_array_equal(result['right_arm_abs'][0],ref[:,25:32])
    assert not hasattr(policy,'token')
    with pytest.raises(ValueError,match='explicit episode'):
        policy.get_action(obs)
    obs['language']['annotation.human.task_description'] = [[INSTRUCTIONS[1]]]
    with pytest.raises(ValueError,match='not p2'):
        policy.get_action(obs,{'episode':'wrong-phase'})
