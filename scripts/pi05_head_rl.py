#!/usr/bin/env python3
"""Full PI05 action-head finetuning with the editor's three losses (BPTT).

  encode  run the frozen PI05 once per replay observation (current and next),
          check it reproduces the recorded reference, and cache the prefix
          K/V (valid tokens) plus the served head weights and initial noise.
  train   TD3-style: twin editor critic (optionally warm-started from an
          editor checkpoint), the action head as the actor; human BC +
          reference/hold + Q, backpropagated through every denoising step.
  eval    offline P1/P3 edit and human-agreement metrics of a trained head.
  install put a trained head into the live learner directory as
          variant-head-<name>.pt (+ .json label), selectable in the console's
          Serve model like any variant; the server swaps PI05's action head
          and serves it with a zero editor (serving.HEAD_KIND).

See vla_precision/gr00t_rlt/pi05_head.py for the formulation. encode/train/eval
write only to --cache and --output; only install writes to the learner
directory (two new files, never an existing one), never to the replay.

  PYTHONPATH=src python scripts/pi05_head_rl.py encode --replay SNAPSHOT.npz
  PYTHONPATH=src python scripts/pi05_head_rl.py train --replay SNAPSHOT.npz --output outputs/pi05-head-rl/run1 \\
      --critic-init outputs/.../learner/step-00000600.pt
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vla_precision.gr00t_rlt.core import (STATIC_HUMAN, drop_static_human, is_shared, load_config,
                                          representation_sha, sha256, validate_replay)

CONFIG = ROOT / "configs/pipette_rl/pi05_fulltask_v6d_focusvel_continuous_30hz.yaml"
RUN = Path("/home/jwang3617/pipette/runs/pi05_v6d_focusvel_editor_continuous_30hz")
HEAD_DEFAULTS = dict(head_lr=5e-6, batch_size=32)


def editor_cli():
    spec = importlib.util.spec_from_file_location("gr00t_rlt_cli", ROOT / "scripts/gr00t_rlt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def default_cache(c):
    return Path(c["output_root"]).parent / "pi05-head-prefix-cache" / Path(c["output_root"]).name


def cache_file(cache, observation_file):
    name = Path(str(observation_file)).name
    if not name.endswith(".npz"):
        raise ValueError(f"Unexpected observation file {observation_file}")
    return Path(cache) / (name[:-4] + ".kv.npz")


def load_snapshot(path, c):
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    if "observation_file" not in data:
        raise ValueError("Snapshot predates observation_file export; export a new one")
    r, d = validate_replay(data, meta, c, token_sha256=representation_sha(c))
    data.update(reward=r, discount=d)
    return data, meta


def paused_guard(run):
    def check():
        st = Path("/tmp/policy_bridge/rlt_status.json")
        if st.is_file():
            s = json.loads(st.read_text())
            if Path(s["root"]).resolve() == Path(run).resolve() and time.time() - s["time"] < 2 and (not s["paused"] or s.get("busy")):
                raise ValueError("Collection resumed: stop the head experiment (no checkpoint written)")
    return check


# --- encode -----------------------------------------------------------------

def encode(a, c):
    import torch
    sys.path.insert(0, c["starvla_root"])
    from vla_precision.gr00t_rlt.features import FrozenPI05Features
    from vla_precision.gr00t_rlt.pi05_head import collate_prefix, compact_prefix, denoise, head_state, reference_noise, save_prefix
    # Only the file columns: the full snapshot (~4 GB) plus PI05's 8 GB load does not fit in RAM.
    meta = json.loads(Path(a.replay).with_suffix(".json").read_text())
    if meta.get("contract_sha256") != c["contract_sha256"]:
        raise ValueError("Snapshot belongs to another recipe")
    with np.load(a.replay, allow_pickle=False) as f:
        if "observation_file" not in f.files:
            raise ValueError("Snapshot predates observation_file export; export a new one")
        files = list(dict.fromkeys([*f["observation_file"].tolist(), *f["next_observation_file"].tolist()]))
    cache = Path(a.cache or default_cache(c))
    cache.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not cache_file(cache, f).exists()][:a.limit or None]
    print(json.dumps(dict(observations=len(files), cached=len(files) - len(todo), encoding=len(todo), cache=str(cache))), flush=True)
    guard = paused_guard(a.run)
    guard()
    feats = FrozenPI05Features(c, a.device)
    model = feats.model
    captured = {}
    forward_prefix, embed_prefix = model.vlm.forward_prefix, model.embed_prefix

    class PrefixOnly(Exception):
        """Stops predict_action once the prefix K/V exists (skips the served denoising)."""

    def capture_forward(*args, **kwargs):
        out = forward_prefix(*args, **kwargs)
        captured["past_key_values"] = out.past_key_values
        if captured.get("prefix_only"):
            raise PrefixOnly
        return out

    def capture_embed(*args, **kwargs):
        out = embed_prefix(*args, **kwargs)
        captured["pad"] = out[1]
        return out

    model.vlm.forward_prefix, model.embed_prefix = capture_forward, capture_embed
    head = model.action_head
    noise = reference_noise(c["reference_seed"], (1, model.action_horizon, model.action_dim), a.device)
    base = cache / "base_head.pt"
    if not base.exists():
        torch.save(dict(state=head_state(head), dims=dict(model.config.framework.action_expert_config),
                        action_dim=model.action_dim, action_horizon=model.action_horizon, max_state_dim=model.max_state_dim,
                        steps=model.num_inference_steps, noise=noise.cpu(), reference_seed=c["reference_seed"],
                        checkpoint=c["checkpoint"], contract_sha256=c["contract_sha256"]), base)
    keys = ("rgb", "focus", "physical_state", "instruction", "reference", "prefix", "contract_sha256")

    def read(rel):
        with np.load(a.run / str(rel), allow_pickle=False) as f:
            return {k: f[k] for k in keys}

    # Every --verify-every'th observation: full served forward + cached sampler, both
    # must reproduce the recorded reference. The rest: VLM prefix only.
    io = concurrent.futures.ThreadPoolExecutor(4)
    reads = [io.submit(read, rel) for rel in todo[:8]]
    writes, errors = [], []
    with (cache / "manifest.jsonl").open("a") as log:
        for n, rel in enumerate(todo):
            if n % 50 == 0:
                guard()
            obs = reads[n].result()
            reads[n] = None
            if n + 8 < len(todo):
                reads.append(io.submit(read, todo[n + 8]))
            if str(obs["contract_sha256"]) != c["contract_sha256"]:
                raise ValueError(f"{rel}: observation recipe mismatch")
            verify = n % a.verify_every == 0
            captured["prefix_only"] = not verify
            sample = {"observation.images.rgb": obs["rgb"], "observation.images.focus": obs["focus"],
                      "observation.state": obs["physical_state"], "task": str(obs["instruction"])}
            try:
                value = feats.extract(sample)
            except PrefixOnly:
                value = None
            prefix = compact_prefix(captured["past_key_values"], captured["pad"])
            row = dict(file=str(rel), tokens=int(prefix[0].shape[3]), verified=verify)
            if verify:
                with torch.inference_mode():
                    batch_kv, valid = collate_prefix([prefix])
                    cached = denoise(head, batch_kv, valid, noise, model.num_inference_steps)[0].float().cpu().numpy()
                row.update(served_reference_error=float(np.abs(value["reference"] - obs["reference"]).max()),
                           cached_reference_error=float(np.abs(cached - obs["reference"]).max()),
                           prefix_error=float(np.abs(value["prefix"].astype(np.float32) - obs["prefix"].astype(np.float32)).max()))
                if row["cached_reference_error"] > a.max_reference_error:
                    raise ValueError(f"Cached sampler does not reproduce the recorded reference: {row}")
                errors.append(row["cached_reference_error"])
            prefix = (prefix[0].cpu(), prefix[1])
            writes.append(io.submit(save_prefix, cache_file(cache, rel), prefix))
            writes = [w for w in writes if not w.done() or w.result()]
            log.write(json.dumps(row) + "\n")
            if n % 100 == 0:
                log.flush()
                print(json.dumps(dict(done=n + 1, of=len(todo), **row)), flush=True)
    for w in writes:
        w.result()
    if errors:
        print(json.dumps(dict(encoded=len(todo), verified=len(errors), cached_reference_error_max=max(errors),
                              cached_reference_error_median=float(np.median(errors)))), flush=True)


# --- shared training/eval plumbing -------------------------------------------

def build_head(base, device):
    import torch
    from starVLA.model.modules.action_model.OpenPI_ActionHead import GemmaDims, OpenPI05ActionHead
    saved = torch.load(base, map_location="cpu", weights_only=True)
    head = OpenPI05ActionHead(GemmaDims(**saved["dims"]), action_dim=saved["action_dim"],
                              action_horizon=saved["action_horizon"], max_state_dim=saved["max_state_dim"])
    missing, unexpected = head.load_state_dict(saved["state"], strict=False)
    if unexpected or any(not k.startswith("gemma_expert.lm_head") for k in missing):
        raise ValueError(f"Base head mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return head, saved


class Loader:
    """Cached prefixes + replay arrays for a batch of windows, one batch ahead."""
    KEYS = ("token", "next_token", "state", "next_state", "reference", "next_reference", "action", "valid",
            "human", "bc_eligible", "reward", "discount", "td_eligible", "scheduler_context",
            "next_scheduler_context", "phase", "next_phase")

    def __init__(self, data, cache, device, workers=8):
        from vla_precision.gr00t_rlt.pi05_head import load_prefix
        self.data, self.cache, self.device, self.load_prefix = data, Path(cache), device, load_prefix
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)

    def _read(self, ids, next_too=True):
        from vla_precision.gr00t_rlt.pi05_head import collate_prefix
        out = dict(ids=np.asarray(ids))
        for key, field in (("kv", "observation_file"), ("next_kv", "next_observation_file")):
            if key == "next_kv" and not next_too:
                continue
            kvs = list(self.pool.map(lambda f: self.load_prefix(cache_file(self.cache, f)), self.data[field][ids]))
            out[key], out[key + "_valid"] = collate_prefix(kvs)
        for k in self.KEYS:
            if k in self.data:
                out[k] = self.data[k][ids]
        return out

    def submit(self, ids, next_too=True):
        return self.pool.submit(self._read, ids, next_too)

    def to_device(self, batch):
        import torch
        out = {}
        for k, v in batch.items():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if t.dtype == torch.float64:
                t = t.float()
            if k in ("phase", "next_phase"):
                t = t.long()
            elif t.dtype not in (torch.bool, torch.bfloat16, torch.long):
                t = t.float()
            out[k] = t.to(self.device, non_blocking=True)
        return out


def check_cache(data, cache):
    missing = [f for f in dict.fromkeys([*data["observation_file"].tolist(), *data["next_observation_file"].tolist()])
               if not cache_file(cache, f).exists()]
    if missing:
        raise ValueError(f"{len(missing)} observations have no cached prefix; run encode first")


# --- train ------------------------------------------------------------------

def train(a, c):
    import torch
    sys.path.insert(0, c["starvla_root"])
    from vla_precision.gr00t_rlt.pi05_head import HeadLearner
    cache = Path(a.cache or default_cache(c))
    out = Path(a.output)
    if out.exists() and any(out.glob("head-step-*.pt")) and not a.resume:
        raise ValueError("Output already has a head run; use --resume or a new --output")
    out.mkdir(parents=True, exist_ok=True)
    data, meta = load_snapshot(a.replay, c)
    check_cache(data, cache)
    options = dict(c["learner"], **HEAD_DEFAULTS)
    for key in ("head_lr", "batch_size", "q_weight", "reference_weight", "critic_lr"):
        if getattr(a, key) is not None:
            options[key] = getattr(a, key)
    static = None if a.keep_static_human else dict(STATIC_HUMAN)
    if static is not None:
        print(json.dumps(dict(human_static_filter=static, **drop_static_human(
            data, c["action_half_range"], c["scheduler"]["action_hz"], **static))), flush=True)
    options.update(human_static_filter=static, human_batch_fraction=a.human_batch_fraction,
                   denoise_checkpointing=a.checkpoint_steps, jacobian_steps=a.jacobian_steps, micro_batch=a.micro_batch)
    c = dict(c, learner=options)
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    head, base = build_head(cache / "base_head.pt", "cpu")
    learner = HeadLearner(c, head, base["noise"], steps=base["steps"], device=a.device,
                          amp=a.device.startswith("cuda"), checkpoint=a.checkpoint_steps, jacobian_steps=a.jacobian_steps,
                          micro_batch=a.micro_batch)
    if a.resume:
        previous = torch.load(a.resume, map_location=a.device, weights_only=True)
        learner.load_state_dict(previous["learner"])
        rng.bit_generator.state = previous["numpy_rng"]
        torch.set_rng_state(previous["torch_rng"].cpu())
    elif a.critic_init:
        editor = torch.load(a.critic_init, map_location=a.device, weights_only=True)
        if editor["contract_sha256"] != c["contract_sha256"]:
            raise ValueError("Critic init belongs to another recipe")
        learner.load_critic(editor["learner"])
    cli = editor_cli()
    phase_rows = [np.flatnonzero(data["phase"] == p) for p in np.unique(data["phase"])] if is_shared(c) else None
    has_human = (data["human"] & data["bc_eligible"] & data["valid"]).any((1, 2))
    human_rows = [r[has_human[r]] for r in phase_rows] if phase_rows else [np.flatnonzero(has_human)]
    human_rows = [r for r in human_rows if len(r)]
    n_human = int(round(a.human_batch_fraction * options["batch_size"])) if human_rows else 0
    loader = Loader(data, cache, a.device)
    guard = paused_guard(a.run)
    # Shuffled: the quota's human windows come last, and per-term gradient stats use the first micro-batch.
    draw = lambda: rng.permutation(cli.sample_batch(rng, len(data["reward"]), options["batch_size"], phase_rows, human_rows, n_human))
    # Numerics check: the fp32-master / bf16-autocast head vs the recorded (served bf16) reference.
    probe = loader._read(draw()[:8], next_too=False)
    probe.pop("ids")
    probe = loader.to_device(probe)
    with torch.no_grad():
        gap = float((learner.act(learner.head, probe["kv"], probe["kv_valid"]) - probe["reference"]).abs().max())
    print(json.dumps(dict(start_step=learner.step, numerics_gap_vs_recorded_reference=gap, options=options,
                          windows=int(len(data["reward"])), human_windows=int(has_human.sum()))), flush=True)
    pending = loader.submit(draw())
    started = time.time()
    with (out / "metrics.jsonl").open("a") as log:
        for done in range(1, a.updates + 1):
            guard()
            batch = pending.result()
            ids = batch.pop("ids")
            batch = loader.to_device(batch)
            if done < a.updates:
                pending = loader.submit(draw())
            actor_step = (learner.step + 1) % options["actor_update_interval"] == 0
            stats = actor_step and a.term_stats_every and (learner.step + 1) // options["actor_update_interval"] % a.term_stats_every == 0
            metrics = learner.update(batch, term_stats=bool(stats))
            metrics.update(human_windows=int(has_human[ids].sum()), seconds=time.time() - started)
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            if done % 10 == 0:
                print(json.dumps(metrics), flush=True)
    guard()
    dest = out / f"head-step-{learner.step:08d}.pt"
    tmp = dest.with_suffix(".tmp")
    torch.save(dict(learner=learner.state_dict(), learner_options=options, contract_sha256=c["contract_sha256"],
                    base_head=str(cache / "base_head.pt"), base_checkpoint=base["checkpoint"],
                    replay=str(a.replay), replay_sha256=sha256(a.replay), episodes=meta.get("episodes"),
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), live_enabled=False), tmp)
    tmp.replace(dest)
    serve = out / f"head-step-{learner.step:08d}.serve.pt"
    torch.save(dict(action_head={k: v.to(torch.bfloat16) for k, v in learner.head.state_dict().items()},
                    base_checkpoint=base["checkpoint"], step=learner.step, learner_options=options), serve)
    print(json.dumps(dict(checkpoint=str(dest), serve=str(serve), sha256=sha256(dest))), flush=True)


# --- install ----------------------------------------------------------------

def install(a, c):
    import re
    import torch
    from vla_precision.gr00t_rlt.serving import HEAD_KIND
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.name or ""):
        raise ValueError("--name: lowercase letters, digits and dashes (it becomes variant-head-<name>.pt)")
    learner = Path(c["output"]) / "learner"
    dest = learner / f"variant-head-{a.name}.pt"
    if dest.exists() or dest.with_suffix(".json").exists():
        raise FileExistsError(f"{dest.name} exists; pick another --name (installed heads are never overwritten)")
    saved = torch.load(a.head, map_location="cpu", weights_only=True)
    if saved.get("contract_sha256") != c["contract_sha256"]:
        raise ValueError("Head checkpoint belongs to another recipe")
    if Path(saved["base_checkpoint"]).resolve() != Path(c["checkpoint"]).resolve():
        raise ValueError("Head was finetuned from another PI05 checkpoint than this recipe serves")
    state = saved["learner"]["head"]      # fp32 master weights; the server casts to its own dtypes
    options = saved["learner_options"]
    tmp = dest.with_suffix(".tmp")
    torch.save(dict(kind=HEAD_KIND, action_head=state, contract_sha256=c["contract_sha256"],
                    base_checkpoint=saved["base_checkpoint"], step=saved["learner"]["step"],
                    learner_options=options, source=str(Path(a.head).resolve()),
                    source_sha256=sha256(a.head), replay=saved.get("replay"), live_enabled=False), tmp)
    tmp.replace(dest)
    digest = sha256(dest)
    label = a.label or (f"FULL HEAD, q_weight {options.get('q_weight')}, batch {options.get('batch_size')}, "
                        f"{saved['learner']['step']} updates, {len(saved.get('episodes') or [])} ep")
    (dest.with_suffix(".json")).write_text(json.dumps(dict(
        label=label, kind="pi05_head", step=saved["learner"]["step"], start_step=0, updates=saved["learner"]["step"],
        episodes=len(saved.get("episodes") or []) or None, snapshot=saved.get("replay"), sha256=digest,
        source=str(Path(a.head).resolve()), created=time.time(),
        note="Serves this action head in place of PI05's with a zero editor; the recorded reference is this head's chunk."),
        indent=2))
    print(json.dumps(dict(installed=str(dest), sha256=digest, label=label)), flush=True)


# --- eval -------------------------------------------------------------------

def evaluate(a, c):
    import torch
    sys.path.insert(0, c["starvla_root"])
    from vla_precision.gr00t_rlt.pi05_head import HeadLearner
    cache = Path(a.cache or default_cache(c))
    data, _ = load_snapshot(a.replay, c)
    head, base = build_head(cache / "base_head.pt", "cpu")
    learner = HeadLearner(dict(c, learner=dict(c["learner"], **HEAD_DEFAULTS)), head, base["noise"], steps=base["steps"],
                          device=a.device, amp=a.device.startswith("cuda"))
    if a.head:
        learner.head.load_state_dict(torch.load(a.head, map_location=a.device, weights_only=True)["learner"]["head"], strict=False)
    learner.head.eval()
    mm = np.asarray(c["action_half_range"])[:12] * np.array([1000] * 3 + [57.3] * 3 + [1000] * 3 + [57.3] * 3)
    rng = np.random.default_rng(0)
    pick = lambda idx, n: np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
    policy = data["proposal_accepted"] & ~data["human"].any((1, 2))
    right = (data["human"][:, :, 6:9].all(2) & data["bc_eligible"][:, :, 6:9].all(2))
    sets = dict(p3_policy=pick(np.flatnonzero((data["phase"] == 2) & policy), a.windows),
                p3_human=pick(np.flatnonzero((data["phase"] == 2) & right.any(1)), a.windows),
                p1_policy=pick(np.flatnonzero((data["phase"] == 0) & policy), a.windows))
    loader = Loader(data, cache, a.device)

    def edits(ids):
        if not len(ids):
            return np.zeros((0, 30, 12))
        out = []
        for chunk in np.array_split(ids, max(1, len(ids) // 16)):
            b = loader._read(chunk, next_too=False)
            b.pop("ids")
            b = loader.to_device(b)
            with torch.no_grad():
                pred = learner._edited(learner.act(learner.head, b["kv"], b["kv_valid"]), b["reference"], b["phase"])
            out.append(((pred - b["reference"][..., :18]).cpu().numpy())[..., :12] * mm)
        return np.concatenate(out)

    e = edits(sets["p3_policy"])
    eh = edits(sets["p3_human"])
    lab = (data["action"][sets["p3_human"]] - data["reference"][sets["p3_human"], :, :18])[..., :12] * mm
    hm = right[sets["p3_human"]]
    cos, ratio, before, after = [], [], [], []
    for i in range(len(eh)):
        m = hm[i]
        if m.sum() < 3:
            continue
        h, d = lab[i][m][:, 6:9].reshape(-1), eh[i][m][:, 6:9].reshape(-1)
        cos.append(h @ d / (np.linalg.norm(h) * np.linalg.norm(d) + 1e-12))
        ratio.append(np.linalg.norm(d) / (np.linalg.norm(h) + 1e-12))
        before.append(np.linalg.norm(lab[i][m][:, 6:9], axis=-1).mean())
        after.append(np.linalg.norm((lab[i][m] - eh[i][m])[:, 6:9], axis=-1).mean())
    left = edits(sets["p1_policy"])[..., 0:3]
    mean = lambda v: float(np.mean(v)) if len(v) else None
    print(json.dumps(dict(head=str(a.head) if a.head else "base",
                          p3_right_edit_mm=mean(np.linalg.norm(e[..., 6:9], axis=-1)),
                          p3_human_cos=float(np.mean(cos)) if cos else None, p3_human_ratio=float(np.median(ratio)) if ratio else None,
                          p3_error_mm_base=float(np.mean(before)) if before else None, p3_error_mm_head=float(np.mean(after)) if after else None,
                          p1_left_edit_mm=mean(np.linalg.norm(left, axis=-1)),
                          p1_left_bias_mm=left.mean((0, 1)).round(2).tolist() if len(left) else None,
                          windows={k: len(v) for k, v in sets.items()})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("encode", "train", "eval", "install"))
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--replay", type=Path, help="encode/train/eval: replay snapshot .npz (with observation_file)")
    p.add_argument("--cache", type=Path, help="prefix K/V cache (default outputs/pi05-head-prefix-cache/<recipe>)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, help="encode: at most this many new observations")
    p.add_argument("--verify-every", type=int, default=20,
                   help="encode: full served forward + cached-sampler parity check on every Nth observation (1 = all)")
    p.add_argument("--max-reference-error", type=float, default=1e-3,
                   help="encode: refuse if the cached sampler differs from the recorded reference by more (normalized)")
    p.add_argument("--output", type=Path, help="train: new run directory")
    p.add_argument("--resume", type=Path)
    p.add_argument("--critic-init", type=Path, help="train: editor checkpoint whose twin critic warm-starts this run")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--head-lr", type=float)
    p.add_argument("--critic-lr", type=float)
    p.add_argument("--q-weight", type=float)
    p.add_argument("--reference-weight", type=float)
    p.add_argument("--keep-static-human", action="store_true")
    p.add_argument("--human-batch-fraction", type=float, default=.5)
    p.add_argument("--checkpoint-steps", action="store_true", help="train: recompute each denoising step in backward (less memory)")
    p.add_argument("--micro-batch", type=int, default=32,
                   help="train: windows per forward/backward chunk; gradients accumulate to the full --batch-size (exact)")
    p.add_argument("--jacobian-steps", type=int, default=None,
                   help="train: backpropagate every denoising step into the weights, but keep the state Jacobian "
                        "dx_{k+1}/dx_k only for the last K steps (identity before). Default: exact BPTT")
    p.add_argument("--term-stats-every", type=int, default=20, help="train: per-term gradient norms every N actor updates (3x backward); 0 = never")
    p.add_argument("--head", type=Path, help="eval: head-step-*.pt (default: the base head); install: the head to install")
    p.add_argument("--windows", type=int, default=400, help="eval: windows per set")
    p.add_argument("--name", help="install: variant-head-<name>.pt")
    p.add_argument("--label", help="install: the label shown in the console's Serve model (default: from the run's options)")
    a = p.parse_args()
    if a.command != "install" and a.replay is None:
        p.error(f"{a.command} needs --replay")
    if a.command == "train" and a.output is None:
        p.error("train needs --output")
    if not 0 <= a.human_batch_fraction < 1:
        p.error("--human-batch-fraction must be in [0, 1)")
    if a.jacobian_steps is not None and a.jacobian_steps < 1:
        p.error("--jacobian-steps must be positive")
    c = load_config(a.config, "p1")
    if a.command == "install" and (a.head is None or a.name is None):
        p.error("install needs --head head-step-*.pt and --name")
    dict(encode=encode, train=train, eval=evaluate, install=install)[a.command](a, c)


if __name__ == "__main__":
    main()
