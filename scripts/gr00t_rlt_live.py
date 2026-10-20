#!/usr/bin/env python3
"""RLToken server/export/finite learner. Robot collection is a separate Bridge flag."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

from vla_precision.gr00t_rlt.core import DEFAULT, load_config, sha256, is_editor, representation_sha, is_wrist_velocity
from vla_precision.gr00t_rlt.replay import Replay

#: the running train/initialize command's progress file (see gr00t_rlt.progress)
PROGRESS = None


def main():
    global PROGRESS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("serve", "export", "train", "initialize", "check-live", "check-bootstrap"))
    p.add_argument("--bootstrap", action="store_true", help="Explicit frozen FOCUS reference collection, no token/actor")
    p.add_argument("--config", type=Path, default=DEFAULT)
    p.add_argument("--phase", choices=["p1","p2","p3","p4","p5"], default="p1")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--focus-camera", help="FOCUS ego stream host:port, same endpoint as Bridge RGB")
    p.add_argument("--checkpoint", type=Path, help="Optional GUI assertion; must match the pinned recipe")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--reference-weight", type=float, default=None,
                   help="train only: override learner.reference_weight for this run (the recipe's "
                        "contract hash, hence replay and serving compatibility, is unchanged; "
                        "the checkpoint records the options used)")
    p.add_argument("--keep-static-human", action="store_true",
                   help="train only: keep held (non-moving) human takeover rows in human BC; by default "
                        "only rows where the operator's command moves are trained (contract unchanged)")
    p.add_argument("--human-batch-fraction", type=float, default=None,
                   help="train only: share of each batch drawn from windows with human BC rows "
                        "(default 0.5; 0 = old phase-stratified sampler; contract unchanged)")
    p.add_argument("--q-weight", type=float, default=None,
                   help="train only: override learner.q_weight for this run; 0 = HG-DAgger-style (human BC + "
                        "reference only, the critic still trains but never moves the actor; contract unchanged)")
    p.add_argument("--editor-activation", choices=("tanh", "linear"), default=None,
                   help="train only: 'linear' removes the editor's tanh bound (residual_scale becomes the output "
                        "gain; edited actions are clamped to +/-2.2). Default: tanh, or the resumed checkpoint's")
    p.add_argument("--editor-phases", nargs="+", choices=("p1", "p2", "p3", "p4", "p5"), default=None,
                   help="train only: the editor edits only in these phases and serves the reference elsewhere "
                        "(e.g. --editor-phases p3; BC/reference losses only there; contract unchanged). Default: all, "
                        "or the resumed checkpoint's")
    p.add_argument("--reference-anchor", type=Path, default=None,
                   help="train only: an editor checkpoint whose output replaces the frozen pi0.5 reference as the "
                        "reference-loss target (a trust region around e.g. the best round; contract unchanged; "
                        "a resumed anchored run keeps its anchor)")
    p.add_argument("--collected-by", type=Path, action="append", default=None,
                   help="export/train: only episodes served by this actor checkpoint (sha256 match); repeatable")
    p.add_argument("--recorder-human-labels", action="store_true",
                   help="export/train: keep only the ~5 human label rows the recorder saw per window, instead of "
                        "labelling the whole chunk from the following windows' human commands (default)")
    p.add_argument("--smoke-test", action="store_true", help="Load and infer with synthetic observation; no socket or robot")
    a = p.parse_args()
    if a.editor_activation is not None and a.command != "train":
        p.error("--editor-activation applies to train only")
    if a.editor_phases is not None and a.command != "train":
        p.error("--editor-phases applies to train only")
    if a.human_batch_fraction is not None and (a.command != "train" or not 0 <= a.human_batch_fraction < 1):
        p.error("--human-batch-fraction applies to train only, in [0, 1)")
    if a.reference_anchor is not None and (a.command != "train" or not a.reference_anchor.is_file()):
        p.error("--reference-anchor applies to train only and must be an existing checkpoint")
    if a.keep_static_human and a.command != "train":
        p.error("--keep-static-human applies to train only")
    if a.bootstrap and a.command != "serve":
        p.error("--bootstrap is only valid for serve")
    if a.updates < 1:
        p.error("--updates must be positive")
    c = load_config(a.config, a.phase)
    if a.reference_weight is not None:
        if a.command != "train":
            p.error("--reference-weight applies to train only")
        if not 0 <= a.reference_weight <= 10:
            p.error("--reference-weight must be in [0, 10]")
        recipe = c["learner"]["reference_weight"]
        # After load_config hashed the recipe: contract_sha256 is unchanged.
        c["learner"]["reference_weight"] = float(a.reference_weight)
        print(f"[learner] reference_weight {a.reference_weight:g} for this run "
              f"(recipe {recipe:g}); contract unchanged", flush=True)
    if a.q_weight is not None:
        if a.command != "train" or not 0 <= a.q_weight <= 10:
            p.error("--q-weight applies to train only, in [0, 10]")
        print(f"[learner] q_weight {a.q_weight:g} for this run (recipe {c['learner']['q_weight']:g}); contract unchanged", flush=True)
        c["learner"]["q_weight"] = float(a.q_weight)   # after load_config hashed the recipe
    if is_editor(c) and (a.bootstrap or a.command in ("initialize", "check-bootstrap")):
        p.error("Direct editor starts from a zero residual; no token/BC/bootstrap initialization")
    if a.checkpoint is not None and a.checkpoint.resolve() != Path(c["checkpoint"]).resolve():
        p.error("--checkpoint differs from the pinned RL recipe")
    sys.path.insert(0, c["starvla_root"])
    sys.path.insert(0, str(Path(c["starvla_root"]).parent / "VLAPolicyBridge"))
    if a.command == "check-bootstrap":
        from vla_precision.gr00t_rlt.bootstrap import check_bootstrap
        print(json.dumps(check_bootstrap(c)))
        return
    if a.command == "check-live":
        from vla_precision.gr00t_rlt.serving import check_live_weights
        print(json.dumps(check_live_weights(c, a.run)))
        return
    if a.command == "serve":
        if c.get("focus", {}).get("enabled"):
            if not a.focus_camera and not a.smoke_test:
                p.error("FOCUS serve requires --focus-camera (same ego stream as Bridge)")
            c["focus_camera"] = None if a.smoke_test else a.focus_camera
            c["offline_smoke_test"] = a.smoke_test
        from vla_precision.gr00t_rlt.serving import LivePolicy
        from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer
        from vla_policy_bridge.pi05_fulltask import synthetic_observation, INSTRUCTIONS
        policy = LivePolicy(c, a.device, bootstrap=a.bootstrap, run=a.run)
        server = None
        try:
            obs = synthetic_observation()
            if is_wrist_velocity(c):
                import numpy as np
                obs["state"]["wrist_vel"] = np.zeros((1, 1, 6), np.float32)
            if c.get("focus", {}).get("enabled"):
                import numpy as np
                obs["video"] = {"rgb": np.zeros((1, 1, 720, 1280, 3), np.uint8)}
            obs["language"]["annotation.human.task_description"] = [[INSTRUCTIONS[int(a.phase[1])-1]]]
            capture_root, policy.capture_root = policy.capture_root, None
            try:
                result = policy.get_action(obs, {"episode": "offline_warmup", "rollout": "offline_warmup"})
            finally:
                policy.capture_root = capture_root  # never save synthetic warmup as robot data
            import numpy as np
            if not all(np.isfinite(v).all() for k,v in result.items() if k != "info"):
                raise ValueError("Non-finite warmup actions")
            if a.smoke_test:
                print(json.dumps(dict(smoke_test="passed", phase=a.phase, actor=policy.actor_path,
                                      behavior_sha256=policy.behavior, token_sha256=policy.token_sha)))
                return
            server = ZmqGr00tPolicyServer(policy, host="127.0.0.1", port=a.port)
            server.run()
        finally:
            if server is not None:
                server.close()
            policy.close()
    else:
        import time
        # Lock the shared phase output, not only the replay root: two different
        # collectors must not publish competing actor/optimizer checkpoints.
        learner_lock = None
        if a.command in ("train", "initialize"):
            if not (a.run / "replay.sqlite3").is_file():
                raise FileNotFoundError("No replay ledger; collect and label real episodes first")
            collection_lock = (a.run / "training.lock").open("a+")
            fcntl.flock(collection_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            Path(c["output"]).mkdir(parents=True, exist_ok=True)
            learner_lock = (Path(c["output"])/"learner.lock").open("a+")
            fcntl.flock(learner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from vla_precision.gr00t_rlt.progress import Progress
            PROGRESS = Progress(a.run, a.command)
        if a.command == "initialize":
            from vla_precision.gr00t_rlt.bootstrap import cache_bootstrap, encode_bootstrap, check_bootstrap, source_check
            import importlib.util
            from types import SimpleNamespace
            check_bootstrap(c)
            spec = importlib.util.spec_from_file_location("rlt_init_cli", Path(__file__).with_name("gr00t_rlt.py"))
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            root = Path(c["output"])
            if (root / "learner/latest.json").exists():
                raise ValueError("Initialization cannot overwrite an RL run")
            manifest_path = root / "features/manifest.json"
            PROGRESS.stage("cache")
            if not manifest_path.exists():
                print(json.dumps(cache_bootstrap(c, a.run)), flush=True)
            manifest = json.loads(manifest_path.read_text())
            if (manifest["contract_sha256"] != c["contract_sha256"]
                    or Path(manifest["bootstrap_source"]["root"]).resolve() != a.run.resolve()):
                raise ValueError("Initialization cache belongs to a different collection")
            source_check(manifest)
            # Existing complete stages can be reused; never overwrite partial outputs.
            # fit_bc checks token provenance against this exact feature manifest.
            options = SimpleNamespace(device=a.device, updates=None)
            for stage in ("token", "bc"):
                dest = root / stage
                if dest.exists() and not (dest / "metadata.json").is_file():
                    raise ValueError(f"Incomplete {stage} initialization under {dest}; preserve it and use a new output or explicitly repair offline. No silent partial-weight reuse.")
            PROGRESS.stage("token")
            if not (root / "token/best.pt").exists():
                print("[initialize] training token", flush=True)
                cli.fit_token(c, options)
            PROGRESS.stage("bc")
            if not (root / "bc/best.pt").exists():
                print("[initialize] training BC actor", flush=True)
                cli.fit_bc(c, options)
            import torch
            token = torch.load(root / "token/best.pt", map_location="cpu", weights_only=True)
            if token.get("feature_manifest_sha256") != sha256(manifest_path):
                raise ValueError("Initialization token belongs to another feature manifest")
            for stage in ("token", "bc"):
                meta = json.loads((root / stage / "metadata.json").read_text())
                if meta["checkpoint_sha256"] != sha256(root / stage / "best.pt"):
                    raise ValueError("Initialization completion marker/checkpoint mismatch")
            source_check(manifest)
            from vla_precision.gr00t_rlt.serving import check_live_weights
            check_live_weights(c)
            PROGRESS.stage("encode")
            print(json.dumps(encode_bootstrap(c, a.run, a.device)), flush=True)
            print("[initialize] complete; select FOCUS v6 RLToken after safely ending the old stack", flush=True)
            PROGRESS.finish()
            return
        db = Replay(a.run)
        path = a.run / "snapshots" / f"{a.phase}-{time.time_ns()}.npz"
        if PROGRESS:
            PROGRESS.stage("export", snapshot=str(path))
        try:
            exported = db.export(c, representation_sha(c), path,
                                 progress=PROGRESS.update if PROGRESS else None,
                                 full_chunk_human=not a.recorder_human_labels,
                                 behaviors=None if a.collected_by is None else [sha256(x) for x in a.collected_by])
            print(json.dumps(exported))
        finally:
            db.close()
        if a.command == "train":
            import importlib.util
            from types import SimpleNamespace
            spec = importlib.util.spec_from_file_location("rlt_train_cli", Path(__file__).with_name("gr00t_rlt.py"))
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            latest = Path(c["output"])/"learner/latest.json"
            if is_editor(c):
                from vla_precision.gr00t_rlt.serving import check_live_weights
                check_live_weights(c)  # validate publication hash before resume; no silent fallback
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
                            and (not s["paused"] or s.get("busy"))):
                        raise ValueError("Pause collection before a finite learner run")
            try:
                replay_check()
                cli.fit_heads(c, SimpleNamespace(device=a.device, updates=a.updates, replay=path,
                                                resume=resume, replay_check=replay_check,
                                                progress=PROGRESS, keep_static_human=a.keep_static_human,
                                                human_batch_fraction=a.human_batch_fraction,
                                                editor_activation=a.editor_activation,
                                                editor_phases=a.editor_phases,
                                                reference_anchor=a.reference_anchor))
            finally:
                audit.close()
            PROGRESS.finish()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if PROGRESS is not None and PROGRESS.state["status"] == "running":
            PROGRESS.finish(error="stopped (SIGINT)" if isinstance(exc, KeyboardInterrupt)
                            else f"{type(exc).__name__}: {exc}")
        raise
