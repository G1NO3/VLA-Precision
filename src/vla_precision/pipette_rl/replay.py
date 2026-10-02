"""Read completed, contract-tagged HIL episodes without writing the actor DB."""

from pathlib import Path
import json, sqlite3, sys
import numpy as np
from .config import IMAGE_MAP, WORKSPACE
from .sampling import validate_sampling, tail_settings
from .rewards import resolve_training_rewards

sys.path.insert(0, str(WORKSPACE / "VLAPolicyBridge"))
from vla_policy_bridge.hil.replay import _decode_observation
from vla_policy_bridge.hil.policy_proposals import lookup as lookup_proposal


def observation(raw):
    from openpi_client.image_tools import resize_with_pad

    state = np.concatenate([np.asarray(raw["joint_position"])[:29], np.asarray(raw["eef_position"])])
    if state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError("Invalid 32-D measured state")
    return {
        "state": state.astype(np.float32)[None],
        **{dst: resize_with_pad(np.asarray(raw[src]), 224, 224)[None] for dst, src in IMAGE_MAP.items()},
    }


class Replay:
    def __init__(self, config, *, sampling=None, rewards=None):
        self.config = config
        self.sampling = validate_sampling(sampling)
        self.rewards = resolve_training_rewards(config,rewards)
        self.index = []
        self.closed = {}
        self.rejected = {}
        self.discarded = set()
        self._collection_signature = None

    @staticmethod
    def discarded_ids(db):
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='discarded_episodes' AND type='table'").fetchone():
            return {r[0] for r in db.execute("SELECT episode_id FROM discarded_episodes")}
        return set()  # Older recordings remain readable.

    def connect(self):
        p = Path(self.config["replay_db"])
        if not p.is_file():
            return None
        db = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        db.execute("PRAGMA query_only=ON")
        return db

    def refresh(self):
        db = self.connect()
        if db is None:
            return
        try:
            self.discarded.update(self.discarded_ids(db))
            for ep in self.discarded:
                self.closed.pop(ep, None)
                self.rejected[ep] = "operator_discard"
            self.index = [r for r in self.index if r[1] not in self.discarded]
            eps = db.execute(
                "SELECT DISTINCT episode_id FROM transitions WHERE buffer='online' AND (terminated=1 OR truncated=1)"
            ).fetchall()
            for (ep,) in eps:
                if ep in self.closed or ep in self.rejected:
                    continue
                rows = db.execute(
                    "SELECT id,step_index,reward,terminated,truncated,intervention,info_json,timestamp_ns FROM transitions WHERE buffer='online' AND episode_id=? ORDER BY step_index",
                    (ep,),
                ).fetchall()
                if not rows or rows[-1][4] or not rows[-1][3]:
                    self.rejected[ep] = "truncated/not explicit outcome"
                    continue
                if [r[1] for r in rows] != list(range(len(rows))):
                    self.rejected[ep] = "noncontiguous episode"
                    continue
                if any(
                    json.loads(r[6]).get("rl_contract") != self.config["contract_sha256"]
                    or json.loads(r[6]).get("action_hz") != 30
                    for r in rows
                ):
                    self.rejected[ep] = "foreign model/rate/contract"
                    continue
                if any(r[3] or r[4] for r in rows[:-1]):
                    self.rejected[ep] = "early terminal"
                    continue
                success = rows[-1][2] > 0
                self.closed[ep] = {"success": success, "steps": len(rows),
                                   "last_timestamp_ns": max(r[7] for r in rows),
                                   "last_row_id": max(r[0] for r in rows)}
                self.index.extend((r[0], ep, bool(r[5])) for r in rows)
            if self.sampling and self.sampling['schema_version'] in (3,4):
                self._refresh_collection_policies(db)
        finally:
            db.close()

    @staticmethod
    def collection_policy_id(model, contract):
        """Identify behaviour weights, never offline counterfactual weights."""
        if model.get('source') != 'live_inference' or model.get('mode') != 'hil_policy':
            return None
        digest = model.get('acob_state_sha256')
        if isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
            return 'acob:' + digest
        # The collector contract binds the BC baseline. Require an explicit
        # no-adapter declaration; absent provenance does not imply BC.
        if ('acob_state_sha256' in model and digest is None
                and not model.get('acob_checkpoint') and model.get('checkpoint')):
            return 'bc:' + contract
        return None

    def _refresh_collection_policies(self, db):
        """Recover legacy episode lineage from matching live proposal records.

        Online proposals exist during human takeover too. Match the actor's
        observation key, episode, step and contract; step-only/offline joins
        could falsely attribute an episode to another policy. Cache unchanged
        append-only proposal tables, while allowing late online records.
        """
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='policy_proposals_v1'").fetchone():
            return
        signature = (db.execute('SELECT MAX(rowid) FROM policy_proposals_v1').fetchone()[0],
                     frozenset(self.closed))
        if signature == self._collection_signature:
            return
        policies = {ep: set() for ep in self.closed}
        for ep, raw in db.execute("""
            SELECT DISTINCT p.episode_id,p.model_json FROM policy_proposals_v1 p
            JOIN transitions t ON t.buffer='online' AND t.episode_id=p.episode_id
              AND t.step_index=p.step_index
              AND json_extract(t.info_json,'$.policy_observation_key')=p.key
            WHERE p.kind='online' AND p.contract=?
        """, (self.config['contract_sha256'],)):
            if ep not in policies:
                continue
            try:
                identity = self.collection_policy_id(json.loads(raw), self.config['contract_sha256'])
            except (ValueError, AttributeError, TypeError):
                identity = None
            policies[ep].add(identity)
        for ep, identities in policies.items():
            self.closed[ep]['collection_policy_id'] = (
                next(iter(identities)) if len(identities) == 1 else None)
        self._collection_signature = signature

    def latest_collection_policy_id(self):
        if not self.closed:
            return None
        newest = max(self.closed, key=lambda ep: (
            self.closed[ep]['last_timestamp_ns'], self.closed[ep]['last_row_id']))
        # Do not silently call an older known policy the latest collection if
        # the newest valid episode has unknown/mixed provenance.
        return self.closed[newest].get('collection_policy_id')

    def priority_episode_ids(self):
        if not self.sampling or self.sampling['schema_version'] not in (3,4):
            return self.recent_episode_ids()
        identity = self.latest_collection_policy_id()
        return sorted(ep for ep, meta in self.closed.items()
                      if identity and meta.get('collection_policy_id') == identity)

    def priority_summary(self):
        return {
            'mode': ('latest_collection_policy' if self.sampling and self.sampling['schema_version'] in (3,4)
                     else 'recent_episodes' if self.sampling and self.sampling['schema_version'] == 2 else 'none'),
            'collection_policy_id': self.latest_collection_policy_id(),
            'episode_ids': self.priority_episode_ids(),
        }

    def recent_episode_ids(self):
        count = self.sampling.get('recent_episode_count', 0) if self.sampling else 0
        if not count:
            return []
        return sorted(self.closed, key=lambda ep: (
            self.closed[ep]['last_timestamp_ns'], self.closed[ep]['last_row_id']),
            reverse=True)[:count]

    def summary(self):
        return {
            "transitions": len(self.index),
            "episodes": len(self.closed),
            "successes": sum(v["success"] for v in self.closed.values()),
            "failures": sum(not v["success"] for v in self.closed.values()),
            "corrections": sum(x[2] for x in self.index),
            "rejected": self.rejected,
            "discarded_episodes": len(self.discarded),
            "sampling": self.sampling,
            "training_rewards": self.rewards,
            "recent_episode_ids": self.recent_episode_ids(),
            "priority": self.priority_summary(),
        }

    def sample_indices(self, n, rng):
        """Select existing rows only; no image decode or reward relabeling.

        Keep the correction quota. Each remaining slot independently selects a
        episode tail or the full replay, so fractional quotas work for batch=2.
        Within each pool, optionally mix the latest collection with the full pool.
        Schema 4 includes successful and failed terminals with no outcome bias;
        older schemas retain success-only tails. Missing priority entries fall
        back to the full pool; missing eligible tails to ordinary replay.
        """
        if not isinstance(n,int) or isinstance(n,bool) or n<1:
            raise ValueError('Batch size must be a positive integer')
        self.refresh()  # Revoke entries even when a caller cached the old index.
        if not self.index:
            raise ValueError("No completed contract-tagged episodes")
        corrections = [x for x in self.index if x[2]]
        priority = set(self.priority_episode_ids())
        fraction = (self.sampling.get('latest_collection_fraction', self.sampling.get('recent_episode_fraction', 0.))
                    if self.sampling else 0.)

        def draw(items, count, episode_key=lambda item: item[1]):
            if not count:
                return []
            probabilities = None
            if fraction > 0 and priority:
                mask = np.array([episode_key(item) in priority for item in items])
                if mask.any() and not mask.all():
                    probabilities = np.full(len(items), (1-fraction)/len(items))
                    probabilities += mask * (fraction / mask.sum())
            indices = (rng.integers(len(items), size=count) if probabilities is None
                       else rng.choice(len(items), size=count, p=probabilities))
            return [items[int(i)] for i in indices]

        nc = int(n * self.config["correction_fraction"]) if corrections else 0
        selected = draw(corrections, nc)
        tails={}
        tail_fraction, tail_seconds, successes_only = tail_settings(self.sampling)
        if tail_fraction>0:
            # Accepted episodes enter index in contiguous step order.
            for item in self.index:
                if not successes_only or self.closed[item[1]]['success']:
                    tails.setdefault(item[1],[]).append(item)
            window=max(1,int(np.ceil(tail_seconds*self.config['action_hz'])))
            tails={ep:items[-window:] for ep,items in tails.items()}
        if not tails:
            selected += draw(self.index, n-nc)
        else:
            episodes=list(tails)
            for _ in range(n-nc):
                if rng.random()<tail_fraction:
                    tail=tails[draw(episodes, 1, episode_key=lambda ep: ep)[0]]
                    if self.sampling['schema_version']==4:
                        # Uniform over the entire tail, including its last frame.
                        # No special terminal quota and no success/failure filter.
                        selected.append(tail[int(rng.integers(len(tail)))])
                    elif len(tail)==1 or rng.random()<self.sampling['terminal_fraction_within_tail']:
                        selected.append(tail[-1])
                    else:
                        selected.append(tail[int(rng.integers(len(tail)-1))])
                else:
                    selected.extend(draw(self.index, 1))
        return selected

    def sample(self, n, rng):
        selected=self.sample_indices(n,rng)
        db = self.connect()
        result = []
        try:
            db.execute("BEGIN")
            if {ep for _, ep, _ in selected} & self.discarded_ids(db):
                raise ValueError("Episode discarded during sampling; refresh replay")
            for idx, ep, _ in selected:
                t = db.execute(
                    """SELECT o.payload,n.payload,t.policy_action,t.intervention,t.reward,t.terminated,t.info_json,t.step_index
                  FROM transitions t JOIN observations o ON o.id=t.observation_id
                  JOIN observations n ON n.id=t.next_observation_id WHERE t.id=?""",
                    (idx,),
                ).fetchone()
                info = json.loads(t[6])
                action = np.asarray(info["commanded_action"], np.float32)
                proposal = np.frombuffer(t[2], np.float32).copy()
                raw_observation = None
                proposal_valid = bool(info.get("policy_valid"))
                # Late proposals are annotations, not retrospectively executed
                # robot actions. Prefer their exact inference source observation.
                if t[3] and not proposal_valid:
                    paired = lookup_proposal(db, info=info, contract=self.config["contract_sha256"],
                                             episode=ep, step=t[7], replay_blob=t[0])
                    if paired is not None:
                        proposal, raw_observation, _ = paired
                        proposal_valid = True
                if action.shape != (3,) or not np.isfinite(action).all():
                    raise ValueError("Invalid executed command")
                # Rank only genuine human overrides of valid, different proposals.
                corrected = bool(t[3] and proposal_valid and np.linalg.norm(action - proposal) > 1e-7)
                reward = self.rewards["time_reward"]
                if t[5]:
                    reward = self.rewards['success_terminal' if self.closed[ep]['success'] else 'failure_terminal']
                result.append(
                    {
                        "episode_id": ep,
                        "step_index": int(t[7]),
                        "steps_to_terminal": self.closed[ep]['steps']-1-int(t[7]),
                        "observations": observation(raw_observation if raw_observation is not None else _decode_observation(t[0])),
                        "next_observations": observation(_decode_observation(t[1])),
                        "actions": action[None],
                        "intervention_bad_actions": proposal[None],
                        "rewards": np.array([reward], np.float32),
                        "masks": np.float32(not t[5]),
                        "dones": bool(t[5]),
                        "intervened": corrected,
                        "bc_intervened": bool(t[3]),
                        "bc_eligible": bool(info.get("training_bc_eligible", True)),
                        "episode_succeed": self.closed[ep]["success"],
                    }
                )
        finally:
            db.close()
        return result
