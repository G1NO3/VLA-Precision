"""Finite-learner progress for the web console: stage + data counts.

The learner (the only writer) keeps ``<run>/learner-progress.json`` current;
the console polls it with stdlib JSON only (no Torch, no replay reads). Each
write replaces the file atomically, so a poll never sees a torn file. Count
updates are throttled to MIN_INTERVAL_S; stage changes and the final state
always write. Progress is advisory: a failed write never stops training.
"""
import json
import os
from pathlib import Path
import time

MIN_INTERVAL_S = 0.5
FILE = "learner-progress.json"
#: stage names in order, per learner command
STAGES = {"train": ("export", "load", "train", "save"),
          "initialize": ("cache", "token", "bc", "encode")}


class Progress:
    def __init__(self, run, command, clock=time.time):
        self.path = Path(run) / FILE
        self.clock = clock
        self._last = float("-inf")
        now = clock()
        self.state = dict(schema="rlt_learner_progress.v1", pid=os.getpid(), command=command,
                          status="running", error=None, stage=None, started=now, updated=now,
                          stages=[dict(name=s, started=None, ended=None)
                                  for s in STAGES.get(command, ())],
                          counts={})
        self._write()

    def _entry(self, name):
        for s in self.state["stages"]:
            if s["name"] == name:
                return s
        s = dict(name=name, started=None, ended=None)
        self.state["stages"].append(s)
        return s

    def _close_open(self, now):
        for s in self.state["stages"]:
            if s["started"] is not None and s["ended"] is None:
                s["ended"] = now

    def stage(self, name, **counts):
        """Enter ``name``; the previous stage ends now. Counts accumulate across
        stages (the export's episode totals stay visible while training)."""
        now = self.clock()
        self._close_open(now)
        self._entry(name)["started"] = now
        self.state["stage"] = name
        self.state["counts"].update(counts)
        self._write()

    def update(self, force=False, **counts):
        self.state["counts"].update(counts)
        if force or self.clock() - self._last >= MIN_INTERVAL_S:
            self._write()

    def finish(self, error=None, **counts):
        self._close_open(self.clock())
        self.state["counts"].update(counts)
        self.state.update(status="failed" if error else "done", error=error,
                          failed_stage=self.state["stage"] if error else None)
        self._write()

    def _write(self):
        now = self.clock()
        self.state["updated"] = now
        self._last = now
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.state, allow_nan=False))
            os.replace(tmp, self.path)
        except (OSError, ValueError):
            pass  # advisory only: the console shows a stale file, training goes on
