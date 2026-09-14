#!/usr/bin/env python3
"""CPU-only, explicit replay export / verification / checkpoint publication.

No SSH, Redis, process restarts, or motion. Transfer immutable directories with
rsync; run verify before training, and publish on the collector after download.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from vla_precision.pipette_rl.config import DEFAULT_CONFIG, load_config
from vla_precision.pipette_rl.replay import Replay


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export_replay(config, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(config["replay_db"])
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".export-") as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir()
        # SQLite backup includes committed WAL pages; never cp a live main DB.
        src = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
        dst = sqlite3.connect(root / "replay.sqlite3")
        try:
            src.backup(dst, pages=256, sleep=0.05)
            dst.execute("PRAGMA journal_mode=DELETE")
            # Select complete, explicitly labelled episodes in the consistent
            # backup. Never expose an in-flight/paused/unlabelled episode.
            discarded = Replay.discarded_ids(dst)
            terminal = {r[0] for r in dst.execute(
                "SELECT t.episode_id FROM transitions t WHERE t.buffer='online' "
                "AND t.terminated=1 AND t.truncated=0 AND NOT EXISTS "
                "(SELECT 1 FROM transitions later WHERE later.buffer='online' "
                "AND later.episode_id=t.episode_id AND later.step_index>t.step_index)")}
            eligible = discarded | terminal
            dst.execute("CREATE TEMP TABLE export_episodes (episode_id TEXT PRIMARY KEY)")
            dst.executemany("INSERT INTO export_episodes VALUES (?)", [(ep,) for ep in sorted(eligible)])
            with dst:
                dst.execute("DELETE FROM transitions WHERE episode_id NOT IN (SELECT episode_id FROM export_episodes)")
                dst.execute("DELETE FROM observations WHERE id NOT IN "
                            "(SELECT observation_id FROM transitions UNION SELECT next_observation_id FROM transitions)")
                # Late proposal annotations carry images too. Preserve the
                # same labelled-episode boundary as the raw replay export.
                if dst.execute("SELECT 1 FROM sqlite_master WHERE name='policy_proposals_v1' AND type='table'").fetchone():
                    dst.execute("DELETE FROM policy_proposals_v1 WHERE episode_id NOT IN (SELECT episode_id FROM export_episodes)")
            # Remove deleted payloads from free pages as well before upload.
            dst.execute("VACUUM")
            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Replay integrity check failed")
        finally:
            dst.close()
            src.close()
        (root / "source-contract.json").write_text(json.dumps(config, indent=2))
        manifest = {
            "format": "pipette.replay-transfer.v1",
            "created_ns": time.time_ns(),
            "files": {name: sha256(root / name) for name in ("replay.sqlite3", "source-contract.json")},
            "baseline_metadata_sha256": sha256(Path(config["baseline"]) / "params/_METADATA"),
            "m_pi": 0.0,
            "episode_selection": "explicit_success_failure_or_discard_only",
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (root / "READY").write_text("complete\n")
        root.rename(destination)
    return destination


def verify_bundle(bundle, config_path=DEFAULT_CONFIG):
    root = Path(bundle).resolve()
    if not (root / "READY").is_file():
        raise ValueError("Incomplete replay bundle")
    m = json.loads((root / "manifest.json").read_text())
    if (m.get("format") != "pipette.replay-transfer.v1" or m.get("m_pi") != 0.0
            or set(m["files"]) != {"replay.sqlite3", "source-contract.json"}):
        raise ValueError("Unknown replay bundle format")
    for name, expected in m["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Transfer checksum mismatch: {name}")
    c = load_config(config_path, source_contract=root / "source-contract.json", replay_db=root / "replay.sqlite3")
    if sha256(Path(c["baseline"]) / "params/_METADATA") != m["baseline_metadata_sha256"]:
        raise ValueError("Remote baseline metadata differs from collector baseline")
    replay = Replay(c)
    replay.refresh()
    return {"contract_sha256": c["contract_sha256"], "m_pi": 0.0,
            "runtime_baseline": c["baseline"], "checkpoint_baseline": c["checkpoint_baseline"],
            "replay": replay.summary(), "robot_output": False}


def publish_checkpoint(source, config):
    from vla_precision.pipette_rl.model import read_checkpoint

    source = Path(source).resolve()
    meta, _ = read_checkpoint(source, config)
    if meta.get("smoke_only") or not meta.get("training_episode_ids"):
        raise ValueError("Only real RL checkpoints with episode lineage may be published")
    replay = Replay(config)
    replay.refresh()
    unknown = set(meta["training_episode_ids"]) - replay.closed.keys()
    if unknown:
        raise ValueError(f"Training episodes discarded or not valid in collector replay: {sorted(unknown)}")
    output = Path(config["output"]) / "learner"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"remote-step-{meta['step']:08d}-{time.time_ns()}"
    with tempfile.TemporaryDirectory(dir=output, prefix=".download-") as tmp:
        copy = Path(tmp) / "checkpoint"
        copy.mkdir()
        for name in ("state.msgpack", "metadata.json", "READY"):
            shutil.copyfile(source / name, copy / name)
        read_checkpoint(copy, config)
        copy.rename(target)
    # Do not reuse the remote latest.json: it contains remote absolute paths.
    fd, tmp = tempfile.mkstemp(dir=output, prefix=".latest-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"checkpoint": str(target), "step": meta["step"]}, f)
    os.replace(tmp, output / "latest.json")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("export", "verify", "publish"))
    p.add_argument("path", type=Path, help="New export directory, received replay bundle, or received checkpoint")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    if args.mode == "verify":
        result = verify_bundle(args.path, args.config)
    elif args.mode == "export":
        result = {"bundle": str(export_replay(load_config(args.config), args.path))}
    else:
        result = {"checkpoint": str(publish_checkpoint(args.path, load_config(args.config))),
                  "robot_output": False, "policy_restarted": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
