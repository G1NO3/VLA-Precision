#!/usr/bin/env python3
"""Pipette ACoB learner; never connects to robot/Redis or starts motion."""

import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from vla_precision.pipette_rl.config import load_config, DEFAULT_CONFIG, WORKSPACE, resolve_training_discount, check_discount_fork
from vla_precision.pipette_rl.sampling import load_sampling, DEFAULT_SAMPLING_CONFIG


def policy_processes():
    # One-GPU workstation: train between trials, without contending with control.
    found = []
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = p.read_bytes().split(b"\0")
            names = [x.rsplit(b'/', 1)[-1] for x in argv[:4]]
            if (b'run_pipette_pi05_policy.py' in names
                    or (b'backfill_pipette_proposals.py' in names and b'--apply' in argv)):
                found.append(int(p.parent.name))
        except (OSError, ValueError):
            pass
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "train", "smoke"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--sampling-config", type=Path, default=DEFAULT_SAMPLING_CONFIG,
                   help="Learner-only success-tail sampling configuration; leaves replay contract unchanged")
    p.add_argument("--uniform-replay", action="store_true",
                   help="Use legacy correction + uniform replay sampling")
    p.add_argument("--discount", type=float,
                   help="Learner discount override; changed discount requires a separate output when resuming")
    p.add_argument("--source-contract", type=Path, help="Collector source-contract.json; permit only runtime path relocation")
    p.add_argument("--replay-db", type=Path, help="Immutable replay snapshot from collector")
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-latest", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-ready", action="store_true",
                   help="Refuse insufficient labelled replay immediately, without waiting or allocating GPU")
    args = p.parse_args()
    c = load_config(args.config, source_contract=args.source_contract, replay_db=args.replay_db)
    discount=resolve_training_discount(c,args.discount)
    from vla_precision.pipette_rl.replay import Replay

    replay = Replay(c, sampling=None if args.uniform_replay else load_sampling(args.sampling_config))
    replay.refresh()
    if args.mode == "check":
        print(json.dumps({"contract": c, "learner_discount":discount, "replay": replay.summary(), "robot_output": False}, indent=2))
        return
    if args.max_updates < 1:
        raise ValueError("--max-updates must be positive")
    if args.mode == "train" and args.require_ready and len(replay.index) < c["min_online"]:
        raise ValueError(f"Need at least {c['min_online']} transitions from completed valid episodes; "
                         f"found {len(replay.index)}. Collect and label Success/Failure first; Discard is excluded.")
    if policy_processes():
        raise RuntimeError("Stop the policy/proposal backfill process before training on the shared GPU")
    import numpy as np

    output = (args.output or Path(c["output"]) / ("smoke" if args.mode == "smoke" else "learner")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and (output / "latest.json").exists():
        if args.resume_latest and args.resume is None:
            args.resume = Path(json.loads((output / "latest.json").read_text())["checkpoint"])
        if args.resume is None:
            raise ValueError("Existing learner output: use --resume-latest or --resume")

    def status(state, **extra):
        d = {"state": state, "time": time.time(), "robot_output": False, **extra}
        tmp = output / "status.json.tmp"
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(output / "status.json")
        print(json.dumps(d), flush=True)

    if args.mode == "train":
        while len(replay.index) < c["min_online"]:
            if policy_processes():
                status("stopped_policy_started", step=0)
                return
            status("waiting_completed_episodes", replay=replay.summary())
            time.sleep(10)
            replay.refresh()
    if policy_processes():
        raise RuntimeError("Policy started while waiting; stop it before allocating the learner")
    if args.resume:
        from vla_precision.pipette_rl.model import read_checkpoint

        metadata, _ = read_checkpoint(args.resume, c)
        previous_discount=resolve_training_discount(c,resumed_metadata=metadata)
        discount=resolve_training_discount(c,args.discount,metadata)
        check_discount_fork(output,args.resume,previous_discount,discount)
        if metadata.get("smoke_only"):
            raise ValueError("A smoke checkpoint cannot initialize a real RL run")
        if "training_episode_ids" not in metadata:
            raise ValueError("Resume checkpoint lacks episode lineage; start a new run for discard-aware training")
        if set(metadata["training_episode_ids"]) & replay.discarded:
            status("stopped_discarded_training_data", step=metadata["step"])
            raise ValueError("Resume checkpoint used discarded episodes; choose a clean checkpoint or restart from BC")
        if set(metadata["training_episode_ids"]) - replay.closed.keys():
            raise ValueError("Replay snapshot lacks valid training history; use a cumulative snapshot")
    status("loading", baseline=c["baseline"], discount=discount, sampling=replay.sampling,
           parent_checkpoint=str(args.resume) if args.resume else None)
    import jax
    from vla_precision.pipette_rl.model import (
        create_agent,
        ContextCache,
        make_batch,
        save_checkpoint,
        restore_checkpoint,
        trainable,
    )

    agent = create_agent(c,discount=discount)
    cache = ContextCache(agent)
    step = 0
    used_episodes = set()
    if args.resume:
        agent, step = restore_checkpoint(agent, c, args.resume)
        used_episodes.update(metadata["training_episode_ids"])
    rng = np.random.default_rng(c["seed"] + step)
    if args.mode == "smoke":
        # Recorded observations, synthetic labels ONLY to exercise real gradients.
        # These records never enter the online DB or a deployable checkpoint.
        run = WORKSPACE / "outputs/pi05-deployment/20260913-002247-c297fb"
        files = sorted(run.glob("*/result.json"))[:2]
        if len(files) < 2:
            raise ValueError("Smoke needs two saved live observations")
        import cv2
        from vla_precision.pipette_rl.config import IMAGE_MAP

        obs = []
        for f in files:
            d = json.loads(f.read_text())
            raw = {"state": np.asarray(d["state"], np.float32)[None]}
            for dst, src in IMAGE_MAP.items():
                im = cv2.cvtColor(cv2.imread(str(f.parent / f"{src}.png")), cv2.COLOR_BGR2RGB)
                from openpi_client.image_tools import resize_with_pad

                raw[dst] = resize_with_pad(im, 224, 224)[None]
            obs.append(raw)
        rows = [
            {
                "observations": obs[i],
                "next_observations": obs[1 - i],
                "actions": np.array([[0, 0, -0.0003]], np.float32),
                "intervention_bad_actions": np.zeros((1, 3), np.float32),
                "rewards": np.array([0.999 if i == 0 else -0.001], np.float32),
                "masks": np.float32(0 if i == 0 else 1),
                "dones": i == 0,
                "intervened": i == 1,
                "episode_succeed": i == 0,
            }
            for i in range(2)
        ]
        status("encoding_smoke_contexts")
        batch = make_batch(rows, cache)
        before = [np.asarray(x).copy() for x in jax.tree.leaves(trainable(agent))]
        for networks in [frozenset({"critic"}), frozenset({"critic", "actor"})]:
            status("smoke_update", networks=sorted(networks))
            start = time.time()
            agent, metrics = agent.update_ql(batch, networks_to_update=networks)
            jax.block_until_ready(agent.state)
            metrics = {k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).ndim == 0}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite metrics")
            status("smoke_update_complete", seconds=time.time() - start, metrics=metrics)
        after = [np.asarray(x) for x in jax.tree.leaves(trainable(agent))]
        changed = any(not np.array_equal(a, b) for a, b in zip(before, after))
        assert changed, "LoRA did not update"
        path = save_checkpoint(agent, c, 1, output / f"checkpoint-{time.time_ns()}")
        # Smoke artefacts are explicitly refused by the deployment loader.
        meta = json.loads((path / "metadata.json").read_text())
        meta["smoke_only"] = True
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))
        restored, _ = restore_checkpoint(agent, c, path)
        for original, reloaded in [
            (trainable(agent), trainable(restored)),
            (agent.state.pi_state.opt_state, restored.state.pi_state.opt_state),
            (agent.state.critic_state, restored.state.critic_state),
        ]:
            for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(reloaded)):
                np.testing.assert_array_equal(a, b)
        status(
            "smoke_passed",
            lora_changed=changed,
            checkpoint_roundtrip=True,
            optimizer_and_critic_roundtrip=True,
            checkpoint=str(path),
        )
        return
    status("training", replay=replay.summary(), sampling=replay.sampling,discount=discount,
           resumed_sampling=metadata.get('replay_sampling') if args.resume else None)
    def check_discarded():
        replay.refresh()
        revoked = used_episodes & replay.discarded
        if revoked:
            status("stopped_discarded_training_data", episodes=sorted(revoked), step=step)
            raise ValueError("Discarded episodes already influenced this run; restart from a clean checkpoint/output")

    target_step = step + args.max_updates
    while step < target_step:
        if policy_processes():
            status("stopped_policy_started", step=step)
            return
        check_discarded()
        rows = replay.sample(c["batch_size"], rng)
        batch = make_batch(rows, cache)
        # Detect revocations during image/VLM encoding before applying updates.
        check_discarded()
        batch_episodes = {r["episode_id"] for r in rows}
        if batch_episodes & replay.discarded:
            continue
        used_episodes.update(batch_episodes)
        start = time.time()
        for _ in range(c["critic_updates_per_step"] - 1):
            agent, _ = agent.update_ql(batch, networks_to_update=frozenset({"critic"}))
        networks = frozenset({"critic"} if step < c["critic_warmup_steps"] else {"critic", "actor"})
        agent, metrics = agent.update_ql(batch, networks_to_update=networks)
        jax.block_until_ready(agent.state)
        step += 1
        scalars = {k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).ndim == 0}
        scalars.update(
            discount=discount,
            sampled_success_terminal_count=sum(bool(r['episode_succeed'] and r['dones']) for r in rows),
            sampled_success_tail_count=sum(bool(r['episode_succeed'] and r['steps_to_terminal']<
                (replay.sampling['success_tail_seconds']*c['action_hz'] if replay.sampling else 0)) for r in rows),
            sampled_steps_to_terminal_mean=float(np.mean([r['steps_to_terminal'] for r in rows])),
        )
        if not all(np.isfinite(v) for v in scalars.values()):
            raise ValueError("Nonfinite training metrics; checkpoint not published")
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "seconds": time.time() - start, **scalars}) + "\n")
        if step % 10 == 0:
            status("training", step=step, replay=replay.summary(), metrics=scalars)
        if step % c["checkpoint_every"] == 0 or step == target_step:
            check_discarded()
            path = save_checkpoint(agent, c, step, output / f"step-{step:08d}",
                                   training_episode_ids=used_episodes, replay_sampling=replay.sampling,
                                   parent_checkpoint=args.resume)
            check_discarded()
            tmp = output / "latest.json.tmp"
            tmp.write_text(json.dumps({"checkpoint": str(path), "step": step}))
            tmp.replace(output / "latest.json")
    status("completed", step=step, discount=discount,sampling=replay.sampling)


if __name__ == "__main__":
    main()
