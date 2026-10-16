"""Shared actor/critic and full-task reward tests, CPU only in bwrap."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from vla_precision.gr00t_rlt.core import (CONTINUOUS, EDITOR, load_config, phase_active,
    representation_sha, reward_and_discount, validate_replay)
from vla_precision.gr00t_rlt.model import Learner, make_heads
from vla_precision.gr00t_rlt.replay import Replay
from test_direct_editor import editor_record, bare_policy


def config(tmp_path):
    c = load_config(CONTINUOUS, workspace=tmp_path)
    c["action_active"] = [True]*18
    c["learner"]["batch_size"] = 5
    return c


def test_shared_contract_and_output_but_different_masks(tmp_path):
    recipes = [load_config(CONTINUOUS, f"p{i}", tmp_path) for i in range(1,6)]
    assert len({c["contract_sha256"] for c in recipes}) == 1
    assert len({c["output"] for c in recipes}) == 1
    assert recipes[0]["output"].endswith("/shared")
    assert recipes[0]["contract_sha256"] != load_config(EDITOR, workspace=tmp_path)["contract_sha256"]
    assert phase_active(recipes[0], "p1") != phase_active(recipes[0], "p2")


def test_reward_at_event_and_final_discount():
    r = config(Path("/tmp"))["reward"]
    # At unit discount/zero time penalty, the five event rewards are exactly
    # 1,1,1,1,11. Intermediate phases are NOT Bellman terminals.
    r = {**r, "gamma":1., "step_penalty":0.}
    assert reward_and_discount(1, 1_000_000_001, "continuing", r, phase_reward_ns=500_000_001) == (1.,1.)
    assert reward_and_discount(1, 1_000_000_001, "success", r) == (11.,0.)
    assert reward_and_discount(1, 1_000_000_001, "failure", r) == (0.,0.)
    with pytest.raises(ValueError, match="timestamp"):
        reward_and_discount(1, 1_000_000_001, "continuing", r, phase_reward_ns=2_000_000_000)


def test_phase_conditioned_actor_critics_and_batched_hand_masks(tmp_path):
    torch.set_num_threads(1)
    c = config(tmp_path)
    model = make_heads(c)
    z, state, ref = torch.zeros(5,16384), torch.zeros(5,59), torch.zeros(5,30,32)
    phase = torch.arange(5)
    assert model.actor[0].in_features == 16384+59+30*32+5
    assert model.critics[0].net[0].in_features == 16384+59+30*(32+36)+5+64
    assert torch.equal(model.act(z,state,ref,phase=phase), ref[...,:18])
    with torch.no_grad():
        model.actor[-1].bias.fill_(1.)
    active = model.active_for(phase).expand(5,30,18)
    edited = model.act(z,state,ref,phase=phase)
    assert (edited[~active] == 0).all() and (edited[active] > 0).all()
    for i in range(5):
        assert model.active_for(phase)[i,0].tolist() == phase_active(c, f"p{i+1}")
    smooth = model.smooth_action(edited, ref, torch.ones_like(edited), phase=phase)
    assert (smooth[~active] == 0).all() and smooth.max() <= .2
    with pytest.raises(ValueError, match="phase labels"):
        model.act(z,state,ref)


def add_trajectory(db, c, outcome="success", start_phase=0):
    meta = dict(phase=f"p{start_phase+1}", contract_sha256=c["contract_sha256"], token_sha256=representation_sha(c), behavior_sha256="test")
    eid = db.begin(meta)
    for p in range(start_phase, 5):
        row = editor_record(c)
        start, end = (p+1)*1_000_000_000, (p+2)*1_000_000_000
        row.update(phase=np.int64(p), next_phase=np.int64(min(p+1,4)),
                   phase_reward_ns=np.int64(end-1 if p<4 else 0), start_ns=np.int64(start), end_ns=np.int64(end))
        row["valid"][:] = phase_active(c, f"p{p+1}")
        row["bc_eligible"][:] = row["valid"]
        db.append(eid,row)
        if p < 4 or outcome == "success":
            db.phase_success(eid, f"p{p+1}", end-1)
    db.finish(eid,outcome,6_000_000_001)
    return eid


@pytest.mark.parametrize("start", range(5))
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_suffix_replay_preserves_start_and_rewards(tmp_path, start, outcome):
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db, c, outcome=outcome, start_phase=start)
    path = tmp_path/"suffix.npz"
    db.export(c, representation_sha(c), path)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["episode_start_phases"] == {eid:f"p{start+1}"}
    assert meta["fulltask_success_episodes"] == ([eid] if start == 0 and outcome == "success" else [])
    with np.load(path) as f:
        data = {k:f[k].copy() for k in f.files}
    assert data["phase"].tolist() == list(range(start, 5))
    validate_replay(data, meta, c, token_sha256=representation_sha(c))
    db.close()


def test_suffix_replay_rejects_wrong_start_or_missing_milestone(tmp_path):
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db, c, start_phase=2)
    with db.db:
        db.db.execute("UPDATE episodes SET phase='p1' WHERE id=?", (eid,))
    with pytest.raises(ValueError, match="start phase"):
        db.export(c, representation_sha(c), tmp_path/"bad.npz")
    with db.db:
        db.db.execute("UPDATE episodes SET phase='p3' WHERE id=?", (eid,))
        db.db.execute("DELETE FROM phase_events WHERE episode=? AND phase='p3'", (eid,))
    with pytest.raises(ValueError, match="phase reward evidence"):
        db.export(c, representation_sha(c), tmp_path/"bad.npz")
    db.close()


def test_shared_replay_train_resume_and_discard(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    c = config(tmp_path)
    db = Replay(tmp_path/"replay")
    eid = add_trajectory(db,c)
    snapshot = tmp_path/"shared.npz"
    db.export(c,representation_sha(c),snapshot)
    with np.load(snapshot) as f:
        data = {k:f[k].copy() for k in f.files}
    meta = json.loads(snapshot.with_suffix(".json").read_text())
    reward, discount = validate_replay(data,meta,c,token_sha256=representation_sha(c))
    assert (discount[:4]>0).all() and discount[-1]==0
    spec = importlib.util.spec_from_file_location("shared_train_cli",Path(__file__).resolve().parents[1]/"scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,"check_code",lambda c:None)
    monkeypatch.setattr(cli,"validate_assets",lambda c:None)
    args = SimpleNamespace(device="cpu",replay=snapshot,resume=None,updates=2)
    cli.fit_heads(c,args)
    out = Path(c["output"])/"learner"
    metrics = [json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines()]
    assert sum(metrics[-1]["phase_samples"].values()) == 5
    assert all(k in metrics[-1] for k in ("critic_loss","bc_loss","ref_loss","rl_loss"))
    first = json.loads((out/"latest.json").read_text())
    args.resume = Path(first["checkpoint"])
    cli.fit_heads(c,args)
    saved = torch.load(json.loads((out/"latest.json").read_text())["checkpoint"],weights_only=True)
    assert saved["learner"]["step"] == 4 and saved["learner"]["critic_updates"] == 4
    db.discard(eid)
    with pytest.raises(ValueError,match="No complete"):
        db.export(c,representation_sha(c),tmp_path/"discard.npz")
    db.close()


def test_serving_keeps_shared_weights_across_phase_and_pause(tmp_path, monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"VLAPolicyBridge"))
    from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS
    c = config(tmp_path)
    p = bare_policy(c)
    p._load_actor()
    p.episode, p.capture_root = None, None
    p.stats = dict(q01=[-1]*32,q99=[1]*32)
    ref = np.zeros((30,32),np.float32)
    mask = np.zeros(800,bool)
    mask[:512] = True
    feature = dict(prefix=np.ones((800,2048),np.float16),mask=mask,state=np.zeros(59,np.float32),reference=ref)
    reset, loaded = [], []
    p.features = SimpleNamespace(extract=lambda s:feature,reset=lambda:reset.append(1),
        physical_actions=lambda a,r:np.concatenate([a,r[:,18:]],axis=-1))
    monkeypatch.setattr(p,"_load_actor",lambda *rollout:loaded.append(1))
    obs = synthetic_observation()
    with pytest.raises(ValueError, match="rollout identity"):
        p.get_action(obs, {"episode":"missing-stable-trajectory-id"})
    for i in range(5):
        obs["language"]["annotation.human.task_description"] = [[INSTRUCTIONS[i]]]
        result = p.get_action(obs, {"episode":f"request-{i}","rollout":"same-physical-trial"})
        assert result["info"]["rlt"]["phase"] == f"p{i+1}"
        assert result["info"]["rlt"]["active"] == phase_active(c,f"p{i+1}")
    assert loaded == reset == [1]
    p.get_action(obs,{"episode":"resume-request","rollout":"same-physical-trial"})
    assert loaded == [1]
    p.get_action(obs,{"episode":"new-request","rollout":"new-trial"})
    assert loaded == [1,1]


def test_replay_requires_all_phase_events_and_new_contract(tmp_path):
    c=config(tmp_path)
    db=Replay(tmp_path/"replay")
    eid=add_trajectory(db,c)
    with db.db:
        db.db.execute("DELETE FROM phase_events WHERE episode=? AND phase='p5'",(eid,))
    with pytest.raises(ValueError,match="P5 milestone"):
        db.export(c,representation_sha(c),tmp_path/"bad.npz")
    old=load_config(EDITOR,workspace=tmp_path)
    with pytest.raises(ValueError,match="No complete"):
        db.export(old,representation_sha(old),tmp_path/"old-contract.npz")
    db.close()


def test_half_speed_recipe_is_its_own_contract(tmp_path):
    half = Path(str(CONTINUOUS).replace("continuous.yaml", "continuous_30hz.yaml"))
    c30, c60 = load_config(half, workspace=tmp_path), load_config(CONTINUOUS, workspace=tmp_path)
    assert c30["scheduler"]["action_hz"] == 30 and c60["scheduler"]["action_hz"] == 60
    assert c30["contract_sha256"] != c60["contract_sha256"]
    assert c30["output"] != c60["output"]
    assert c30["reward"] == c60["reward"]   # elapsed-time discount clock is rate-independent


def test_unsupported_playback_rate_is_rejected(tmp_path):
    import yaml
    raw = yaml.safe_load(Path(CONTINUOUS).read_text())
    raw["scheduler"]["action_hz"] = 45
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="action_hz"):
        load_config(path, workspace=tmp_path)
