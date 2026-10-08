import json
import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import load_config, validate_replay
from vla_precision.gr00t_rlt.replay import Replay
from vla_precision.gr00t_rlt.model import Learner


def test_committed_registry_overlay_preserves_recipe_pin(monkeypatch):
    from vla_precision.gr00t_rlt import core
    c=load_config()
    def revision(cmd, **kwargs):
        return core.REGISTRY_COMMIT if cmd[2]==c['starvla_root'] else c['rlinf_commit']
    monkeypatch.setattr(core.subprocess,'check_output',revision)
    core.check_code(c)
    monkeypatch.setattr(core.subprocess,'check_output',lambda *a,**kw:'unreviewed-revision')
    with pytest.raises(ValueError,match='commit differs'):
        core.check_code(c)


def record(c):
    r = {k: np.zeros(59,np.float32) for k in ("state","next_state")}
    r.update({k: np.zeros(256,np.float32) for k in ("token","next_token")})
    r.update({k: np.zeros((30,32),np.float32) for k in ("reference","next_reference")})
    r.update(action=np.zeros((30,18),np.float32), valid=np.zeros((30,18),bool),
             human=np.zeros((30,18),bool),bc_eligible=np.zeros((30,18),bool),
             td_eligible=np.bool_(True),start_ns=np.int64(1),end_ns=np.int64(500000001))
    r["valid"][:3] = c.get("action_active",[True]*18)
    r["bc_eligible"][:] = r["valid"]
    return r


def test_replay_finish_discard_reopen_crash(tmp_path):
    c=load_config()
    db=Replay(tmp_path)
    meta=dict(phase="p1",contract_sha256=c["contract_sha256"],token_sha256="t",behavior_sha256="b")
    eid=db.begin(meta)
    db.append(eid,record(c))
    with pytest.raises(ValueError,match="No complete"):
        db.export(c,"t",tmp_path/"incomplete.npz")
    db.finish(eid,"success",1000000001)
    dest=tmp_path/"complete.npz"
    assert db.export(c,"t",dest)["transitions"]==1
    with np.load(dest) as data:
        reward,discount=validate_replay(data,json.loads(dest.with_suffix('.json').read_text()),c,token_sha256="t")
        assert reward[0]>9 and discount[0]==0
    db.discard(eid)
    with pytest.raises(ValueError,match="No complete"):
        db.export(c,"t",tmp_path/"discarded.npz")
    assert db.db.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]==1
    db.begin(meta)
    db.close()
    db=Replay(tmp_path)
    db.recover()
    assert db.counts()=={"aborted":1,"discarded":1}
    db.close()


def test_all_human_batch_has_bc_but_no_critic_update():
    torch.set_num_threads(2)
    c=load_config()
    r=record(c)
    r['td_eligible']=np.bool_(False)
    r['human'][:]=r['valid']
    r['action'][:]=.5
    b={k:torch.as_tensor(np.stack([v,v])) for k,v in r.items() if k not in ('start_ns','end_ns')}
    b.update(reward=torch.zeros(2),discount=torch.ones(2)*.99)
    learner=Learner(c)
    before={k:v.clone() for k,v in learner.model.critics.state_dict().items()}
    learner.update(b)
    m=learner.update(b)
    assert m['td_samples']==0 and m['critic_loss']==0 and m['weighted_rl_loss']==0
    assert m['human_bc_loss']>0 and m['ref_loss']==0
    assert m['actor_loss']==pytest.approx(m['weighted_bc_loss']+m['weighted_ref_loss']+m['weighted_rl_loss'])
    assert all(torch.equal(v,learner.model.critics.state_dict()[k]) for k,v in before.items())
    assert 'rehearsal_bc_loss' not in m
    assert m['critic_updates']==0


def test_actor_envelope_preserves_frozen_channels():
    from vla_precision.gr00t_rlt.model import Heads
    heads=Heads(load_config())
    with torch.no_grad():
        for p in heads.actor.parameters():
            p.zero_()
        heads.actor[-1].bias.fill_(9.)
        reference=torch.full((1,30,32),7.)
        action=heads.act(torch.zeros(1,256),torch.zeros(1,59),reference)
    assert torch.all(action[...,heads.active]==2.2)
    assert torch.all(action[...,~heads.active]==7.)
