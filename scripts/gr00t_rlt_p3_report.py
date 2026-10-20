#!/usr/bin/env python3
"""Real-robot P3 (insertion) report per served editor, read-only.

Groups every recorded episode (success, failure and discarded) by the actor
that served it (episode behavior = the checkpoint's sha256, matched against
step-*.pt / variant-*.pt in the learner dir) and prints, for the P3 windows:
episodes that entered / finished P3, median P3 duration, share of windows the
operator took over, takeovers per episode, the editor's own edit of the right
wrist (proposal - reference) and the operator's correction of the right wrist
(human label - reference, i.e. how far off the policy was), both in mm.

  PYTHONPATH=src python scripts/gr00t_rlt_p3_report.py
"""
import argparse
import collections
import glob
import io
import os
import sqlite3
from pathlib import Path

import numpy as np

from vla_precision.gr00t_rlt.core import load_config, sha256

ROOT = Path(__file__).resolve().parents[1]
P3 = 2
RIGHT_XYZ = slice(6, 9)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=ROOT / "configs/pipette_rl/pi05_fulltask_v6d_focusvel_continuous_30hz.yaml")
    p.add_argument("--run", type=Path, default=Path("/home/jwang3617/pipette/runs/pi05_v6d_focusvel_editor_continuous_30hz"))
    a = p.parse_args()
    c = load_config(a.config, "p1")
    mm = np.asarray(c["action_half_range"])[RIGHT_XYZ] * 1000
    names = {sha256(Path(f)): Path(f).stem for f in glob.glob(str(Path(c["output"]) / "learner/*.pt"))
             if Path(f).name.startswith(("step-", "variant-"))}
    db = sqlite3.connect(f"file:{a.run / 'replay.sqlite3'}?mode=ro", uri=True)
    finished = collections.defaultdict(set)
    for e, phase in db.execute("SELECT episode, phase FROM phase_events"):
        finished[e].add(phase)
    rows = collections.OrderedDict()
    for e, behavior, outcome in db.execute("SELECT id, behavior, outcome FROM episodes ORDER BY created").fetchall():
        name = "zero editor" if behavior.startswith("editor_zero") else names.get(behavior, behavior[:12])
        r = rows.setdefault(name, dict(episodes=0, success=0, entered=0, done=0, windows=0, human=0,
                                       duration=[], takeovers=[], edit=[], correction=[]))
        r["episodes"] += 1
        r["success"] += outcome == "success"
        windows = human = takeovers = 0
        start = end = None
        before = False
        for (blob,) in db.execute("SELECT payload FROM transitions WHERE episode=? ORDER BY seq", (e,)):
            z = np.load(io.BytesIO(blob))
            if int(z["phase"]) != P3:
                continue
            windows += 1
            start = int(z["start_ns"]) if start is None else start
            end = int(z["end_ns"])
            took = z["human"][:, RIGHT_XYZ].all(1)
            if took.any():
                human += 1
                moving = took & z["bc_eligible"][:, RIGHT_XYZ].all(1)
                if moving.any():
                    r["correction"].append(np.linalg.norm((z["action"][moving, RIGHT_XYZ] - z["reference"][moving, RIGHT_XYZ]) * mm, axis=1).mean())
            elif bool(z["proposal_accepted"]):
                r["edit"].append(np.linalg.norm((z["proposal_action"][:, RIGHT_XYZ] - z["reference"][:, RIGHT_XYZ]) * mm, axis=1).mean())
            takeovers += bool(took.any()) and not before
            before = bool(took.any())
        if windows:
            r["entered"] += 1
            r["done"] += "p3" in finished[e]
            r["windows"] += windows
            r["human"] += human
            r["duration"].append((end - start) / 1e9)
            r["takeovers"].append(takeovers)
    db.close()
    mean = lambda v: f"{np.mean(v):5.2f}" if v else "    -"
    print(f"{'actor':34s} {'success':>8s} {'P3 in/done':>10s} {'P3 s (med)':>10s} {'human win':>9s} "
          f"{'takeover/ep':>11s} {'edit mm':>7s} {'human corr mm (n)':>17s}")
    for name, r in rows.items():
        if not r["entered"]:
            print(f"{name:34s} {r['success']:3d}/{r['episodes']:<4d} {'0':>10s}")
            continue
        print(f"{name:34s} {r['success']:3d}/{r['episodes']:<4d} {r['entered']:4d}/{r['done']:<5d} "
              f"{np.median(r['duration']):10.1f} {r['human']/r['windows']:9.1%} {np.mean(r['takeovers']):11.1f} "
              f"{mean(r['edit']):>7s} {mean(r['correction']):>10s} ({len(r['correction'])})")


if __name__ == "__main__":
    main()
