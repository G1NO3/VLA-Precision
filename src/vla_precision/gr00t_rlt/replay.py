"""Crash-safe episode ledger. Discard is reversible metadata, never file deletion.

Only complete success/failure episodes export. The collector stores proposal
actions plus a per-row publication audit, NOT claims of motor execution.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np


class Replay:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "replay.sqlite3", timeout=10, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, phase TEXT, contract TEXT,
            token TEXT, behavior TEXT, outcome TEXT, created REAL, terminal_ns INTEGER);
          CREATE TABLE IF NOT EXISTS transitions(episode TEXT, seq INTEGER, payload BLOB,
            PRIMARY KEY(episode,seq));
          CREATE TABLE IF NOT EXISTS events(created REAL, episode TEXT, kind TEXT, detail TEXT);
          CREATE TABLE IF NOT EXISTS transition_summary(episode TEXT, seq INTEGER,
            human INTEGER, td INTEGER, bc INTEGER, PRIMARY KEY(episode,seq));
          CREATE TABLE IF NOT EXISTS phase_events(episode TEXT, phase TEXT,
            reward REAL, stamp_ns INTEGER, PRIMARY KEY(episode,phase));
        """)

    def recover(self):
        with self.db:
            self.db.execute("UPDATE episodes SET outcome='aborted' WHERE outcome='running'")

    def begin(self, metadata, *, eid=None):
        eid = eid or uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,'running',?,NULL)",
                (eid, metadata["phase"], metadata["contract_sha256"], metadata["token_sha256"],
                 metadata["behavior_sha256"], time.time()))
        return eid

    def event(self, eid, kind, detail):
        with self.db:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?)",
                            (time.time(), eid, kind, json.dumps(detail)))

    def append(self, eid, record):
        blob = io.BytesIO()
        np.savez_compressed(blob, **record)
        with self.db:
            outcome = self.db.execute("SELECT outcome FROM episodes WHERE id=?", (eid,)).fetchone()
            if outcome != ("running",):
                raise ValueError("Cannot append to a closed episode")
            seq = self.db.execute("SELECT COUNT(*) FROM transitions WHERE episode=?", (eid,)).fetchone()[0]
            self.db.execute("INSERT INTO transitions VALUES(?,?,?)", (eid, seq, blob.getvalue()))
            self.db.execute("INSERT INTO transition_summary VALUES(?,?,?,?,?)",
                (eid, seq, int(np.any(record.get("human", False))),
                 int(np.any(record.get("td_eligible", False))),
                 int(np.any(record.get("bc_eligible", False)))))

    def finish(self, eid, outcome, terminal_ns):
        if outcome not in ("success", "failure", "discarded", "aborted"):
            raise ValueError(outcome)
        with self.db:
            self.db.execute("UPDATE episodes SET outcome=?,terminal_ns=? WHERE id=? AND outcome='running'",
                            (outcome, terminal_ns, eid))

    def phase_success(self, eid, phase, stamp_ns):
        """One credited milestone per phase/trajectory, retained after discard."""
        if phase not in ("p1", "p2", "p3", "p4", "p5") or stamp_ns <= 0:
            raise ValueError("Invalid phase milestone")
        with self.db:
            if self.db.execute("SELECT outcome FROM episodes WHERE id=?", (eid,)).fetchone() != ("running",):
                raise ValueError("Phase success requires a running trajectory")
            self.db.execute("INSERT INTO phase_events VALUES(?,?,?,?)", (eid, phase, 1.0, stamp_ns))

    def discard(self, eid):
        with self.db:
            self.db.execute("UPDATE episodes SET outcome='discarded' WHERE id=?", (eid,))
            self.db.execute("INSERT INTO events VALUES(?,?,'discard',?)", (time.time(), eid, "{}"))

    def counts(self):
        return dict(self.db.execute("SELECT outcome,COUNT(*) FROM episodes GROUP BY outcome"))

    def export(self, c, token_sha256, dest, progress=None, *, full_chunk_human=True, behaviors=None):
        """``progress(**counts)``, optional: called per transition read (the
        caller throttles) and once before the snapshot is written.

        ``full_chunk_human`` (async editor): relabel every takeover window's
        whole 30-row chunk from its own AND the following windows' published
        human commands (vla_policy_bridge.editor_labels), instead of only the
        ~5 rows the recorder could see. The recorder's labels must come out
        identical, or the export refuses.

        ``behaviors``: optional behavior identities (the served actor's sha256)
        to export only the episodes those actors collected."""
        from .core import validate_replay, is_async_editor, is_shared
        report = progress or (lambda **counts: None)
        full_chunk_human = bool(full_chunk_human and is_async_editor(c))
        if full_chunk_human:
            relabel = dict(recorded=0, labelled=0, windows=0)
        if token_sha256.startswith("bootstrap:"):
            raise ValueError("Bootstrap replay needs a trained token and encode-bootstrap before RL export")
        rows, ids = [], []
        start_phases, fulltask_successes = {}, []
        capture_checked = {}
        # One read transaction: a discard must not produce a half-episode snapshot.
        with self.db:
            self.db.execute("BEGIN")
            scope = "contract=?" if is_shared(c) else "phase=? AND contract=?"
            args = [c["contract_sha256"], token_sha256] if is_shared(c) else [c["phase"], c["contract_sha256"], token_sha256]
            if behaviors is not None:
                behaviors = sorted(set(behaviors))
                scope += " AND behavior IN (" + ",".join("?"*len(behaviors)) + ")"
                args[-1:-1] = behaviors
            episodes = self.db.execute("SELECT id,outcome,terminal_ns,phase FROM episodes WHERE " + scope +
                " AND token=? AND outcome IN ('success','failure') ORDER BY created", args).fetchall()
            for n, (eid, outcome, stamp, start_phase) in enumerate(episodes):
                payloads = self.db.execute("SELECT seq,payload FROM transitions WHERE episode=? ORDER BY seq", (eid,)).fetchall()
                if not payloads or [p[0] for p in payloads] != list(range(len(payloads))):
                    continue
                ids.append(eid)
                episode_rows = []
                for i, (_, blob) in enumerate(payloads):
                    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
                        row = {k: f[k].copy() for k in f.files}
                    # Async capture is not durable when the action is returned.
                    # Refuse training until BOTH exact observations are committed
                    # and hash-verified. Cache small alignment fields, not images.
                    for prefix in ("", "next_"):
                        if prefix + "observation_receipt" in row:
                            from .bootstrap import load_observation
                            rel = str(row[prefix + "observation_file"].item())
                            receipt = str(row[prefix + "observation_receipt"].item())
                            key = (rel, receipt)
                            if key not in capture_checked:
                                captured = load_observation(self.root, row, c, prefix)
                                capture_checked[key] = (captured["state"], captured["reference"])
                            for name, expected in zip(("state", "reference"), capture_checked[key]):
                                if not np.array_equal(row[prefix + name], expected):
                                    raise ValueError("Observation/replay alignment mismatch: " + prefix + name)
                    row.update(outcome=outcome if i == len(payloads)-1 else "continuing",
                               episode_outcome=outcome, terminal_ns=stamp if i == len(payloads)-1 else 0)
                    episode_rows.append(row)
                    report(step="read", episodes_done=n, episodes_total=len(episodes),
                           transitions=len(rows)+len(episode_rows),
                           observations_checked=len(capture_checked))
                if full_chunk_human and any(np.any(r.get("publication_human", False)) for r in episode_rows):
                    self._full_chunk_human(episode_rows, c, relabel)
                if is_shared(c):
                    # An episode may begin at any phase, but cannot splice
                    # independently reset phases or earn skipped milestones.
                    start = int(episode_rows[0]["phase"])
                    if start_phase != f"p{start+1}" or not 0 <= start < 5:
                        raise ValueError("Episode start phase disagrees with first transition")
                    start_phases[eid] = start_phase
                    milestones = []
                    for i, row in enumerate(episode_rows):
                        if i and int(episode_rows[i-1]["next_phase"]) != int(row["phase"]):
                            raise ValueError("Disconnected phase transitions")
                        if int(row["phase_reward_ns"]):
                            milestones.append((f"p{int(row['phase'])+1}", int(row["phase_reward_ns"])))
                    events = self.db.execute("SELECT phase,stamp_ns FROM phase_events WHERE episode=? AND phase!='p5' ORDER BY phase", (eid,)).fetchall()
                    if milestones != events or len({p for p, _ in milestones}) != len(milestones):
                        raise ValueError("Missing or duplicate phase reward evidence")
                    if outcome == "success" and (
                            [p for p, _ in milestones] != [f"p{i+1}" for i in range(start, 4)]
                            or int(episode_rows[-1]["phase"]) != 4):
                        raise ValueError("Success requires all phases from episode start through P5")
                    final = self.db.execute("SELECT stamp_ns FROM phase_events WHERE episode=? AND phase='p5'", (eid,)).fetchone()
                    if outcome == "success" and (not final or not int(episode_rows[-1]["start_ns"]) <= final[0] <= stamp):
                        raise ValueError("Final success lacks operator P5 milestone evidence")
                    if outcome == "failure" and final:
                        raise ValueError("Failure episode cannot also have final success")
                    if outcome == "success" and start == 0:
                        fulltask_successes.append(eid)
                rows.extend(episode_rows)
                report(step="read", episodes_done=n+1, episodes_total=len(episodes),
                       transitions=len(rows), observations_checked=len(capture_checked))
        if not rows:
            raise ValueError("No complete compatible episodes; collect and label first")
        report(step="write", episodes_done=len(episodes), episodes_total=len(episodes),
               transitions=len(rows), observations_checked=len(capture_checked))
        # Audit arrays are stored in SQLite; training interchange uses only these fields.
        keys = ("token", "next_token", "state", "next_state", "reference", "next_reference",
                "action", "valid", "human", "bc_eligible", "td_eligible", "start_ns", "end_ns",
                "terminal_ns", "outcome", "episode_outcome")
        if is_async_editor(c):
            keys += ("scheduler_context", "next_scheduler_context", "proposal_accepted", "autonomy", "next_autonomy")
            if all("observation_file" in r and "next_observation_file" in r for r in rows):
                keys += ("observation_file", "next_observation_file")   # PI05 head training re-reads the prefix
        if is_shared(c):
            keys += ("phase", "next_phase", "phase_reward_ns")
        data = {k: np.stack([r[k] for r in rows]) for k in keys}
        meta = dict(schema="gr00t_rlt_replay.v1", phase="fulltask" if is_shared(c) else c["phase"], contract_sha256=c["contract_sha256"],
                    token_sha256=token_sha256, execution_contract=c["execution_contract"],
                    completed_episodes_only=True, episodes=ids, collector="bridge_sequential_v1",
                    human_td="excluded_controller_mismatch", action_evidence="bridge_published_not_motor_ack")
        if is_async_editor(c):
            meta.update(collector="bridge_async_act_v1", action_evidence="requested_chunk_not_act_blend",
                        human_labels="full_chunk_following_windows_v1" if full_chunk_human else "recorder_own_window_v1")
            if full_chunk_human:
                meta["human_relabel"] = relabel
        if behaviors is not None:
            meta["behavior_filter"] = behaviors
        if is_shared(c):
            meta.update(episode_start_phases=start_phases,
                        fulltask_success_episodes=fulltask_successes,
                        success_scope="ordered_start_phase_through_p5")
        validate_replay(data, meta, c, token_sha256=token_sha256)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.with_suffix(".json").exists():
            raise FileExistsError("Never overwrite a replay snapshot")
        with dest.open("xb") as f:
            np.savez_compressed(f, **data)
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
        return dict(episodes=len(ids), transitions=len(rows), path=str(dest),
                    **({"human_relabel": relabel} if full_chunk_human else {}))

    def _full_chunk_human(self, episode_rows, c, counts):
        """Relabel the takeover windows of one episode in place (see export)."""
        from vla_policy_bridge.editor_labels import episode_labels
        if not hasattr(self, "_label_kin"):
            from vla_policy_bridge.pico_intervention import UrdfKinematics, URDF, ARM_SIDES
            self._label_kin = {s: UrdfKinematics(URDF, target_link=link, arm_slice=sl) for s, (link, sl) in ARM_SIDES.items()}
        kin = self._label_kin
        windows = []
        for row in episode_rows:
            with np.load(self.root / str(row["observation_file"].item()), allow_pickle=False) as f:
                raw = f["physical_state"].copy()
            windows.append(dict(start_ns=int(row["start_ns"]), raw=raw, active=row["valid"][0],
                                ns=row["publication_ns"], q=row["published_q29"], hands=row["publication_hands"],
                                human=row["publication_human"], eligible=row["publication_eligible"]))
        labels = episode_labels(windows, kin, c["action_q01"], c["action_q99"], c["scheduler"]["action_hz"])
        for row, (values, mask) in zip(episode_rows, labels):
            if not row["publication_human"].any():
                continue   # autonomous windows keep the proposal (TD / reference rows)
            old = row["human"]
            if (old & ~mask).any() or (old.any() and np.abs(values[old] - row["action"][old]).max() > 1e-4):
                raise ValueError("Full-chunk human labels disagree with the recorder's labels")
            row["action"] = np.where(mask, values, row["action"]).astype(row["action"].dtype)
            row["human"], row["bc_eligible"] = mask.copy(), mask.copy()   # as the recorder: no reference rows
            counts["recorded"] += int(old.sum()); counts["labelled"] += int(mask.sum()); counts["windows"] += 1

    def close(self):
        self.db.close()
