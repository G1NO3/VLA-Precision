"""RLT experiment settings, actual replay chunks, and human demonstration masks."""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / 'VLA-Precision/configs/pipette_rl/rlt_hil293.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path=DEFAULT):
    c = json.loads(Path(path).read_text())
    assert c['algorithm'] == 'pipette_rlt'
    assert (c['state_dim'], c['action_dim'], c['action_hz'], c['model_horizon']) == (32, 3, 30, 10)
    assert 1 <= c['credit_horizon'] <= c['model_horizon']
    assert c['action_representation'] == 'delta_xyz'
    c['contract_sha256'] = digest(c)
    for k in ['baseline', 'source_replay', 'replay_db', 'output']:
        c[k] = str((ROOT / c[k]).resolve())
    stats = list((Path(c['baseline'])/'assets').rglob('norm_stats.json'))
    if len(stats) != 1 or hashlib.sha256(stats[0].read_bytes()).hexdigest() != c['norm_sha256']:
        raise ValueError('RLT baseline normalization mismatch')
    c['norm_stats'] = str(stats[0])
    return c


def bc_targets(reference, actions, human, eligible, minimum_mm=.01, intentional_hold=None):
    """Return reference/human targets and weights; waiting remains in TD only."""
    human = np.asarray(human, bool)
    active = np.linalg.norm(actions, axis=-1)*1000 >= minimum_mm
    if intentional_hold is not None:
        active |= np.asarray(intentional_hold, bool)
    weights = np.asarray(eligible, bool) & (~human | active)
    targets = np.where(human[..., None], actions, reference)
    return targets, weights


def chunk_return(rewards, gamma):
    return float(np.dot(np.asarray(rewards), gamma**np.arange(len(rewards))))


def open_replay(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


def replay_chunks(c, *, source=True):
    """Nonoverlapping actual action sequences; never cross takeover or gaps.

    Partial chunks are retained only when they contain an explicit terminal.
    Their padded actions are masked in critic/BC inputs and the bootstrap is zero.
    Discarded, partial, foreign and unlabelled episodes never enter learning.
    """
    path = c['source_replay'] if source else c['replay_db']
    if not Path(path).is_file():
        return [], {}
    contract = c['source_contract'] if source else c['contract_sha256']
    chunks, outcomes = [], {}
    with open_replay(path) as db:
        collection={}
        if db.execute("select 1 from sqlite_master where name='policy_proposals_v1'").fetchone():
            for ep,raw_model in db.execute("""select distinct p.episode_id,p.model_json
                from policy_proposals_v1 p join transitions t
                on t.buffer='online' and t.episode_id=p.episode_id and t.step_index=p.step_index
                and json_extract(t.info_json,'$.policy_observation_key')=p.key
                where p.kind='online' and p.contract=?""",(contract,)):
                model=json.loads(raw_model)
                identity=(model.get('sha256',{}).get('heads.pt') if model.get('format')=='pipette.rlt.v1'
                          else model.get('acob_state_sha256'))
                if not identity and model.get('source')=='live_inference' and not model.get('acob_checkpoint') and not model.get('rlt_checkpoint'):
                    identity='bc:'+contract
                collection.setdefault(ep,set()).add(identity)
        discard = set()
        if db.execute("select 1 from sqlite_master where name='discarded_episodes'").fetchone():
            discard = {r[0] for r in db.execute('select episode_id from discarded_episodes')}
        eps = db.execute("select distinct episode_id from transitions where buffer='online' and (terminated=1 or truncated=1)").fetchall()
        for (ep,) in eps:
            if ep in discard:
                continue
            raw = db.execute("select id,step_index,observation_id,next_observation_id,intervention,reward,terminated,truncated,timestamp_ns,info_json from transitions where buffer='online' and episode_id=? order by step_index", (ep,)).fetchall()
            if not raw or not raw[-1][6] or raw[-1][7] or [r[1] for r in raw] != list(range(len(raw))):
                continue
            if any(r[6] or r[7] for r in raw[:-1]):
                continue
            info = [json.loads(r[9]) for r in raw]
            if any(i.get('rl_contract') != contract or i.get('action_hz') != c['action_hz'] for i in info):
                continue
            outcomes[ep] = bool(raw[-1][5] > 0)
            groups, current = [], []
            for r, i in zip(raw, info):
                if current and (bool(r[4]) != bool(current[-1][0][4]) or
                                i.get('source') != current[-1][1].get('source') or
                                not 0 < (r[8]-current[-1][0][8])/1e9 <= c['max_recording_gap_s']):
                    groups.append(current);current = []
                current.append((r, i))
            if current:
                groups.append(current)
            h = c['credit_horizon']
            for group in groups:
                for start in range(0, len(group), h):
                    window = group[start:start+h]
                    if len(window) < h and not window[-1][0][6]:
                        continue
                    last = window[-1][0]
                    actions = np.zeros((h, 3), np.float32)
                    human = np.zeros(h, bool); eligible = np.zeros(h, bool); hold = np.zeros(h, bool)
                    valid = np.zeros(h, bool); rewards = []
                    for k, (r, i) in enumerate(window):
                        a = np.asarray(i['commanded_action'], np.float32)
                        if a.shape != (3,) or not np.isfinite(a).all():
                            raise ValueError('Invalid replay action')
                        actions[k] = a; human[k] = bool(r[4]); valid[k] = True
                        eligible[k] = bool(i.get('training_bc_eligible', True))
                        hold[k] = bool(i.get('intentional_hold', False))
                        rewards.append((c['success_reward'] if outcomes[ep] else c['failure_reward']) if r[6] else c['time_reward'])
                    chunks.append(dict(episode=ep,step=window[0][0][1],db=str(path),
                        collection_policy=(next(iter(collection[ep])) if len(collection.get(ep,()))==1 else None),
                        observation_id=window[0][0][2],next_observation_id=last[3],
                        actions_m=actions,human=human,eligible=eligible,hold=hold,valid=valid,
                        reward=chunk_return(rewards,c['gamma']),discount=0. if last[6] else c['gamma']**len(window),
                        terminal=bool(last[6]),success=outcomes[ep],steps_to_terminal=len(raw)-1-last[1],
                        timestamp_ns=window[0][0][8],length=len(window)))
    return chunks, outcomes


def sample_indices(chunks, n, rng, c):
    """Equal tail/full branches for either outcome, no human-zero oversampling."""
    all_ids = np.arange(len(chunks))
    tail = all_ids[np.array([x['steps_to_terminal'] < c['tail_seconds']*c['action_hz'] for x in chunks])]
    # Priority follows the behaviour weights, not merely recency or DB location.
    newest=max(chunks,key=lambda x:x.get('timestamp_ns',0))
    identity=newest.get('collection_policy')
    latest={i for i,x in enumerate(chunks) if identity and x.get('collection_policy')==identity}
    selected = []
    for _ in range(n):
        pool = tail if len(tail) and rng.random() < c['tail_fraction'] else all_ids
        recent = [i for i in pool if i in latest]
        if recent and rng.random() < c['latest_collection_fraction']:
            pool = recent
        selected.append(int(rng.choice(pool)))
    return selected
