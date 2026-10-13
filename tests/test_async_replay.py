import io
import threading
import sqlite3

import numpy as np
import pytest

from vla_precision.gr00t_rlt.async_replay import AsyncReplay
from vla_precision.gr00t_rlt.replay import Replay


META = dict(phase="p1", contract_sha256="c", token_sha256="t", behavior_sha256="b")


def test_fifo_snapshot_terminal_counts_and_discard(tmp_path):
    replay = Replay(tmp_path)
    writer = AsyncReplay(replay)
    eid = writer.begin(META)
    writer.flush()
    entered, release = threading.Event(), threading.Event()
    append = replay.append
    def blocked(*a):
        entered.set()
        assert release.wait(3)
        append(*a)
    replay.append = blocked
    try:
        row = dict(action=np.ones((30, 18)), td_eligible=np.bool_(True))
        writer.append(eid, row)
        assert entered.wait(1)
        row["action"][:] = 7
        writer.phase_success(eid, "p1", 10)
        writer.event(eid, "terminal_evidence", {"event_id": "test"})
        writer.finish(eid, "success", 20)
        assert writer.status()["pending"] >= 4
        assert writer.counts() == {"running": 1}  # not falsely successful
        release.set()
        writer.flush()
        assert writer.counts() == {"success": 1}
        with sqlite3.connect(tmp_path/"replay.sqlite3") as db:
            blob = db.execute("SELECT payload FROM transitions").fetchone()[0]
            with np.load(io.BytesIO(blob)) as f:
                np.testing.assert_array_equal(f["action"], 1.)
            assert db.execute("SELECT phase FROM phase_events").fetchone() == ("p1",)
            assert db.execute("SELECT kind FROM events").fetchone() == ("terminal_evidence",)
        writer.discard(eid)
        writer.flush()
        assert writer.counts() == {"discarded": 1}
    finally:
        release.set()
        writer.close()


def test_worker_error_latches_and_cannot_commit_queued_success(tmp_path):
    replay = Replay(tmp_path)
    writer = AsyncReplay(replay)
    eid = writer.begin(META)
    writer.flush()
    entered, release = threading.Event(), threading.Event()
    def fail(*a):
        entered.set()
        assert release.wait(3)
        raise OSError("disk failed")
    replay.append = fail
    try:
        writer.append(eid, {})
        assert entered.wait(1)
        writer.finish(eid, "success", 1)
        release.set()
        with pytest.raises(RuntimeError, match="disk failed"):
            writer.flush()
        assert writer.counts() == {"aborted": 1}
        with pytest.raises(RuntimeError, match="disk failed"):
            writer.begin(META)
    finally:
        release.set()
        writer.close()


def test_full_queue_nonblocking_and_close_drains(tmp_path):
    replay = Replay(tmp_path)
    writer = AsyncReplay(replay, capacity=1)
    eid = writer.begin(META)
    writer.flush()
    entered, release = threading.Event(), threading.Event()
    append = replay.append
    def block(*a):
        entered.set()
        assert release.wait(3)
        append(*a)
    replay.append = block
    writer.append(eid, dict(x=np.array([1])))
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError, match="queue full"):
            writer.finish(eid, "success", 1)
        assert writer.status()["error"]
    finally:
        release.set()
        writer.close()
    reader = Replay(tmp_path)
    assert reader.counts() == {"aborted": 1}
    reader.close()
