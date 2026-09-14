"""Read completed, contract-tagged HIL episodes without writing the actor DB."""

from pathlib import Path
import json, sqlite3, sys
import numpy as np
from .config import IMAGE_MAP, WORKSPACE
from .sampling import validate_sampling

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
    def __init__(self, config, *, sampling=None):
        self.config = config
        self.sampling = validate_sampling(sampling)
        self.index = []
        self.closed = {}
        self.rejected = {}
        self.discarded = set()

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
                    "SELECT id,step_index,reward,terminated,truncated,intervention,info_json FROM transitions WHERE buffer='online' AND episode_id=? ORDER BY step_index",
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
                self.closed[ep] = {"success": success, "steps": len(rows)}
                self.index.extend((r[0], ep, bool(r[5])) for r in rows)
        finally:
            db.close()

    def summary(self):
        return {
            "transitions": len(self.index),
            "episodes": len(self.closed),
            "successes": sum(v["success"] for v in self.closed.values()),
            "corrections": sum(x[2] for x in self.index),
            "rejected": self.rejected,
            "discarded_episodes": len(self.discarded),
            "sampling": self.sampling,
        }

    def sample_indices(self, n, rng):
        """Select existing rows only; no image decode or reward relabeling.

        Keep the correction quota. Each remaining slot independently selects a
        success tail or the full replay, so fractional quotas work for batch=2.
        Success episodes are chosen uniformly, then a terminal or preceding tail
        row is drawn. Missing success data falls back to ordinary replay.
        """
        if not isinstance(n,int) or isinstance(n,bool) or n<1:
            raise ValueError('Batch size must be a positive integer')
        self.refresh()  # Revoke entries even when a caller cached the old index.
        if not self.index:
            raise ValueError("No completed contract-tagged episodes")
        corrections = [x for x in self.index if x[2]]
        nc = int(n * self.config["correction_fraction"]) if corrections else 0
        selected = [corrections[int(i)] for i in rng.integers(len(corrections), size=nc)] if nc else []
        tails={}
        if self.sampling and self.sampling['success_tail_fraction']>0:
            # Accepted episodes enter index in contiguous step order.
            for item in self.index:
                if self.closed[item[1]]['success']:
                    tails.setdefault(item[1],[]).append(item)
            window=max(1,int(np.ceil(self.sampling['success_tail_seconds']*self.config['action_hz'])))
            tails={ep:items[-window:] for ep,items in tails.items()}
        if not tails:
            selected += [self.index[int(i)] for i in rng.integers(len(self.index), size=n-nc)]
        else:
            episodes=list(tails)
            for _ in range(n-nc):
                if rng.random()<self.sampling['success_tail_fraction']:
                    tail=tails[episodes[int(rng.integers(len(episodes)))]]
                    if len(tail)==1 or rng.random()<self.sampling['terminal_fraction_within_tail']:
                        selected.append(tail[-1])
                    else:
                        selected.append(tail[int(rng.integers(len(tail)-1))])
                else:
                    selected.append(self.index[int(rng.integers(len(self.index)))])
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
                reward = self.config["time_reward"]
                if t[5]:
                    reward += (
                        self.config["success_reward"] if self.closed[ep]["success"] else self.config["failure_reward"]
                    )
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
