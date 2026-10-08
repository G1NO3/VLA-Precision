#!/usr/bin/env python3
"""RLToken server/export/finite learner. Robot collection is a separate Bridge flag."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

from vla_precision.gr00t_rlt.core import DEFAULT, load_config, sha256
from vla_precision.gr00t_rlt.replay import Replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("serve", "export", "train"))
    p.add_argument("--config", type=Path, default=DEFAULT)
    p.add_argument("--phase", choices=["p1","p2","p3","p4","p5"], default="p1")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--checkpoint", type=Path, help="Optional GUI assertion; must match the pinned recipe")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--smoke-test", action="store_true", help="Load and infer with synthetic observation; no socket or robot")
    a = p.parse_args()
    if a.updates < 1:
        p.error("--updates must be positive")
    c = load_config(a.config, a.phase)
    if a.checkpoint is not None and a.checkpoint.resolve() != Path(c["checkpoint"]).resolve():
        p.error("--checkpoint differs from the pinned RL recipe")
    sys.path.insert(0, c["starvla_root"])
    sys.path.insert(0, str(Path(c["starvla_root"]).parent / "VLAPolicyBridge"))
    if a.command == "serve":
        from vla_precision.gr00t_rlt.serving import LivePolicy
        from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer
        from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS
        policy = LivePolicy(c, a.device)
        obs = synthetic_observation()
        obs["language"]["annotation.human.task_description"] = [[INSTRUCTIONS[int(a.phase[1])-1]]]
        result = policy.get_action(obs, {"episode": "offline_warmup"})
        import numpy as np
        if not all(np.isfinite(v).all() for k,v in result.items() if k != "info"):
            raise ValueError("Non-finite warmup actions")
        if a.smoke_test:
            print(json.dumps(dict(smoke_test="passed", phase=a.phase, actor=policy.actor_path,
                                  behavior_sha256=policy.behavior, token_sha256=policy.token_sha)))
            return
        server = ZmqGr00tPolicyServer(policy, host="127.0.0.1", port=a.port)
        try:
            server.run()
        finally:
            server.close()
    else:
        import time
        # Lock the shared phase output, not only the replay root: two different
        # collectors must not publish competing actor/optimizer checkpoints.
        learner_lock = None
        if a.command == "train":
            Path(c["output"]).mkdir(parents=True, exist_ok=True)
            learner_lock = (Path(c["output"])/"learner.lock").open("a+")
            fcntl.flock(learner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = Replay(a.run)
        path = a.run / "snapshots" / f"{a.phase}-{time.time_ns()}.npz"
        try:
            print(json.dumps(db.export(c, sha256(Path(c["output"])/"token/best.pt"), path)))
        finally:
            db.close()
        if a.command == "train":
            import importlib.util
            from types import SimpleNamespace
            spec = importlib.util.spec_from_file_location("rlt_train_cli", Path(__file__).with_name("gr00t_rlt.py"))
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            latest = Path(c["output"])/"learner/latest.json"
            resume = Path(json.loads(latest.read_text())["checkpoint"]) if latest.is_file() else None
            expected = json.loads(path.with_suffix(".json").read_text())["episodes"]
            audit = Replay(a.run)
            def replay_check():
                # Discard revokes the current update run as well as future sampling.
                # Already-published weights cannot be retroactively untrained.
                rows = dict(audit.db.execute("SELECT id,outcome FROM episodes"))
                if any(rows.get(e) not in ("success","failure") for e in expected):
                    raise ValueError("Replay episode revoked/discarded: update not published")
                status = Path("/tmp/policy_bridge/rlt_status.json")
                if status.is_file():
                    s = json.loads(status.read_text())
                    if (Path(s["root"]).resolve() == a.run.resolve() and time.time()-s["time"] < 2
                            and not s["paused"]):
                        raise ValueError("Pause collection before a finite learner run")
            try:
                replay_check()
                cli.fit_heads(c, SimpleNamespace(device=a.device, updates=a.updates, replay=path,
                                                resume=resume, replay_check=replay_check))
            finally:
                audit.close()


if __name__ == "__main__":
    main()
