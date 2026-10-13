"""Bounded CPU-only observation persistence, outside the action response path.

An action response names a pending observation and its future durable receipt.
Only a successfully fsynced NPZ + receipt is usable as training evidence. A
crash leaves missing evidence, never a fabricated successful capture.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .bootstrap import save_observation


class CaptureWriter:
    def __init__(self, root, max_pending=8):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.root = Path(root)
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._pending = 0
        self._error = None
        self._closed = False
        self._last_ms = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="focus-capture")

    def check(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Observation writer failed; stop collection") from self._error
            if self._closed:
                raise RuntimeError("Observation writer is closed")

    def status(self):
        with self._lock:
            return dict(pending=self._pending, last_write_ms=self._last_ms,
                        error=None if self._error is None else str(self._error))

    def submit(self, features, config, episode):
        self.check()
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("Observation writer queue full; stop collection")
        try:
            # Own the CPU buffers: neither a camera refresh nor the next model
            # request may mutate a queued sample. Never retain CUDA tensors.
            arrays = {k: np.array(v, copy=True) for k, v in features.items()
                      if not k.startswith("_")}
            key = uuid.uuid4().hex
            with self._lock:
                self._pending += 1
            try:
                self._pool.submit(self._write, arrays, dict(config), episode, key)
            except BaseException:
                with self._lock:
                    self._pending -= 1
                raise
        except BaseException:
            self._slots.release()
            raise
        return dict(observation_file=f"observations/{key}.npz",
                    observation_receipt=f"observations/{key}.receipt.json")

    def _write(self, arrays, config, episode, key):
        start = time.perf_counter()
        try:
            # DEFLATE measured ~235 ms for a 6.9-MB captured prefix+images;
            # merely moving it to one thread would fill the queue at 5 Hz.
            # ZIP_STORED preserves exact arrays and np.load compatibility.
            result = save_observation(self.root, arrays, config, episode,
                                      key=key, compressed=False)
            path = self.root / f"observations/{key}.receipt.json"
            tmp = path.with_suffix(".pending")
            receipt = dict(schema="focus_capture_receipt.v1", **result,
                           compression="stored",
                           contract_sha256=config["contract_sha256"],
                           request_episode=episode)
            with tmp.open("x") as f:
                json.dump(receipt, f)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            print(f"[focus capture] FAILED: {exc}", flush=True)
        finally:
            with self._lock:
                self._pending -= 1
                self._last_ms = (time.perf_counter()-start)*1000
            self._slots.release()

    def close(self):
        # Drain on graceful shutdown; SIGKILL instead leaves pending evidence
        # untrainable. Do not silently discard queued observations.
        self._pool.shutdown(wait=True)
        self.check()
        with self._lock:
            self._closed = True
