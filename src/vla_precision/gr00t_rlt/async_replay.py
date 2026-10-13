"""Bounded FIFO replay writer. Control threads never compress or touch SQLite.

Counts are COMMITTED counts. A queued finish is not durable success. The
collector must retain its training lock until pending==0 and error is None.
One worker owns every begin/append/milestone/event/finish/discard operation.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time
import uuid

from .replay import Replay


class AsyncReplay:
    def __init__(self, replay, capacity=32):
        self.root = replay.root
        self._replay = replay  # ownership transfers after startup recovery
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(capacity)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-writer")
        self._pending = 0
        self._counts = replay.counts()
        self._error = None
        self._closed = False
        self._last_ms = self._last_wait_ms = 0.
        self._last_committed = None
        self._aborted_on_error = False

    def check(self):
        with self._lock:
            if self._error:
                raise RuntimeError("Replay writer failed: " + self._error)
            if self._closed:
                raise RuntimeError("Replay writer closed")

    def _submit(self, method, *args, **kwargs):
        with self._lock:
            self.check()
            if not self._slots.acquire(blocking=False):
                self._error = "bounded queue full; collection must stop"
                raise RuntimeError("Replay writer " + self._error)
            try:
                # Own the snapshot: callers may clear/reuse audit arrays as
                # soon as enqueue returns. No GPU tensors belong in replay.
                args, kwargs = copy.deepcopy((args, kwargs))
                queued = time.perf_counter()
                self._pending += 1
                try:
                    self._pool.submit(self._write, method, args, kwargs, queued)
                except BaseException:
                    self._pending -= 1
                    raise
            except BaseException:
                self._slots.release()
                raise

    def _write(self, method, args, kwargs, queued):
        start = time.perf_counter()
        try:
            with self._lock:
                failed = self._error is not None
            if not failed:
                getattr(self._replay, method)(*args, **kwargs)
            with self._lock:
                recover = self._error is not None and not self._aborted_on_error
            if recover:
                self._replay.recover()  # running episodes can never become success
                self._aborted_on_error = True
            counts = self._replay.counts()
            with self._lock:
                self._counts = counts
                if not failed:
                    self._last_committed = method
        except Exception as exc:
            with self._lock:
                self._error = self._error or repr(exc)
            try:
                self._replay.recover()
                counts = self._replay.counts()
                with self._lock:
                    self._counts = counts
                    self._aborted_on_error = True
            except Exception:
                pass  # process restart recovery still rejects running episodes
        finally:
            with self._lock:
                self._last_ms = (time.perf_counter()-start)*1000
                self._last_wait_ms = (start-queued)*1000
                self._pending -= 1
                self._slots.release()

    def begin(self, metadata):
        eid = uuid.uuid4().hex
        self._submit("begin", metadata, eid=eid)
        return eid

    def append(self, *args): self._submit("append", *args)
    def phase_success(self, *args): self._submit("phase_success", *args)
    def finish(self, *args): self._submit("finish", *args)
    def discard(self, *args): self._submit("discard", *args)
    def event(self, *args): self._submit("event", *args)

    def counts(self):
        with self._lock:
            return dict(self._counts)

    def status(self):
        with self._lock:
            return dict(pending=self._pending, error=self._error, closed=self._closed,
                        last_write_ms=self._last_ms, last_queue_wait_ms=self._last_wait_ms,
                        last_committed=self._last_committed, counts_are_committed=True)

    def flush(self):
        """Offline/test barrier ONLY. Never call from a control callback."""
        self._pool.submit(lambda: None).result()
        self.check()

    def export(self, *args, **kwargs):
        self.flush()  # offline API; the real learner opens its own Replay
        reader = Replay(self.root)
        try:
            return reader.export(*args, **kwargs)
        finally:
            reader.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)  # drain in FIFO order before closing SQLite
        self._replay.close()
