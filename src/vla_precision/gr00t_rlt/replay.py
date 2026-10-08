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
        """)

    def recover(self):
        with self.db:
            self.db.execute("UPDATE episodes SET outcome='aborted' WHERE outcome='running'")

    def begin(self, metadata):
        eid = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,'running',?,NULL)",
                (eid, metadata["phase"], metadata["contract_sha256"], metadata["token_sha256"],
                 metadata["behavior_sha256"], time.time()))
        return eid

    def append(self, eid, record):
        blob = io.BytesIO()
        np.savez_compressed(blob, **record)
        with self.db:
            outcome = self.db.execute("SELECT outcome FROM episodes WHERE id=?", (eid,)).fetchone()
            if outcome != ("running",):
                raise ValueError("Cannot append to a closed episode")
            seq = self.db.execute("SELECT COUNT(*) FROM transitions WHERE episode=?", (eid,)).fetchone()[0]
            self.db.execute("INSERT INTO transitions VALUES(?,?,?)", (eid, seq, blob.getvalue()))

    def finish(self, eid, outcome, terminal_ns):
        if outcome not in ("success", "failure", "discarded", "aborted"):
            raise ValueError(outcome)
        with self.db:
            self.db.execute("UPDATE episodes SET outcome=?,terminal_ns=? WHERE id=? AND outcome='running'",
                            (outcome, terminal_ns, eid))

    def discard(self, eid):
        with self.db:
            self.db.execute("UPDATE episodes SET outcome='discarded' WHERE id=?", (eid,))
            self.db.execute("INSERT INTO events VALUES(?,?,'discard',?)", (time.time(), eid, "{}"))

    def counts(self):
        return dict(self.db.execute("SELECT outcome,COUNT(*) FROM episodes GROUP BY outcome"))

    def export(self, c, token_sha256, dest):
        from .core import validate_replay
        rows, ids = [], []
        # One read transaction: a discard must not produce a half-episode snapshot.
        with self.db:
            self.db.execute("BEGIN")
            episodes = self.db.execute("SELECT id,outcome,terminal_ns FROM episodes WHERE phase=? "
                "AND contract=? AND token=? AND outcome IN ('success','failure') ORDER BY created",
                (c["phase"], c["contract_sha256"], token_sha256)).fetchall()
            for eid, outcome, stamp in episodes:
                payloads = self.db.execute("SELECT seq,payload FROM transitions WHERE episode=? ORDER BY seq", (eid,)).fetchall()
                if not payloads or [p[0] for p in payloads] != list(range(len(payloads))):
                    continue
                ids.append(eid)
                for i, (_, blob) in enumerate(payloads):
                    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
                        row = {k: f[k].copy() for k in f.files}
                    row.update(outcome=outcome if i == len(payloads)-1 else "continuing",
                               episode_outcome=outcome, terminal_ns=stamp if i == len(payloads)-1 else 0)
                    rows.append(row)
        if not rows:
            raise ValueError("No complete compatible episodes; collect and label first")
        # Audit arrays are stored in SQLite; training interchange uses only these fields.
        keys = ("token", "next_token", "state", "next_state", "reference", "next_reference",
                "action", "valid", "human", "bc_eligible", "td_eligible", "start_ns", "end_ns",
                "terminal_ns", "outcome", "episode_outcome")
        data = {k: np.stack([r[k] for r in rows]) for k in keys}
        meta = dict(schema="gr00t_rlt_replay.v1", phase=c["phase"], contract_sha256=c["contract_sha256"],
                    token_sha256=token_sha256, execution_contract=c["execution_contract"],
                    completed_episodes_only=True, episodes=ids, collector="bridge_sequential_v1",
                    human_td="excluded_controller_mismatch", action_evidence="bridge_published_not_motor_ack")
        validate_replay(data, meta, c, token_sha256=token_sha256)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.with_suffix(".json").exists():
            raise FileExistsError("Never overwrite a replay snapshot")
        with dest.open("xb") as f:
            np.savez_compressed(f, **data)
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
        return dict(episodes=len(ids), transitions=len(rows), path=str(dest))

    def close(self):
        self.db.close()
