#!/usr/bin/env python3
"""Offline GR00T RLT: pinned downloads, feature probe/cache, token and AC training.

No command in this entrypoint connects to Redis, cameras, DDS or a robot.
The separate gr00t_rlt_live.py entrypoint serves the sequential Bridge collector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from vla_precision.gr00t_rlt.core import DEFAULT, check_code, digest, load_config, sha256, validate_assets, validate_replay


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def download(c):
    from huggingface_hub import HfApi, snapshot_download
    b = c["baseline"]
    checkpoint = f"gr00t/checkpoints/steps_{b['step']}_pytorch_model.pt"
    names = [checkpoint, "gr00t/config.yaml", "gr00t/dataset_statistics.json"]
    info = HfApi().model_info(b["repo"], revision=b["revision"], files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    root = Path(c["run_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    missing = sum(remote[n].size for n in names if not (root.parent / n).exists())
    if shutil.disk_usage(root.parent).free < missing + (20 << 30):
        raise OSError("Keep at least 20 GiB free; refuse download")
    snapshot_download(b["repo"], revision=b["revision"], local_dir=root.parent,
                      allow_patterns=names + ["gr00t/*.yaml", "README.md"], max_workers=2)
    receipt = {"repo": b["repo"], "revision": b["revision"], "files": {}}
    for name in names:
        path = root.parent / name
        actual = sha256(path)
        expected = remote[name].lfs.sha256 if remote[name].lfs else None
        if path.stat().st_size != remote[name].size or (expected and actual != expected):
            raise ValueError(f"Download size/SHA mismatch: {name}")
        receipt["files"][name.removeprefix("gr00t/")] = dict(bytes=path.stat().st_size, sha256=actual)
    write_json(root / "download_receipt.json", receipt)
    print(json.dumps(validate_assets(c)), flush=True)


def readiness(c):
    """Read-only inventory. Online readiness is NEVER inferred from unit tests."""
    checks = {}
    for key, check in (("code", lambda: check_code(c)), ("model", lambda: validate_assets(c))):
        try:
            check()
            checks[key] = {"ready": True}
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            checks[key] = {"ready": False, "reason": str(exc)}
    root = Path(c["prepared_root"]) / c["phase"]
    checks["prepared_data"] = {"ready": (root / "manifest.json").is_file(), "path": str(root)}
    for key, subpath in (("features", "features/manifest.json"), ("token", "token/best.pt"),
                         ("bc", "bc/best.pt"), ("learner", "learner/latest.json")):
        path = Path(c["output"]) / subpath
        checks[key] = {"present": path.is_file(), "path": str(path)}
    return dict(phase=c["phase"], checkpoint=c["checkpoint"], checks=checks,
                offline_assets_ready=all(checks[k]["ready"] for k in ("code", "model", "prepared_data")),
                online_ready=False, implemented=["RLToken service", "sequential Bridge collector",
                    "SQLite outcome/discard ledger", "finite replay learner", "opt-in GUI lane"], blockers=[
                    "Sequential collector and outcome controls still need physical robot acceptance",
                    "Do not reuse IL async+ACT logs as sequential_chunk_no_ensemble_v1 replay",
                    "PICO hardware axes/tracking-loss and bimanual handback need operator validation",
                    "Only P1 token/BC prepared; other phases require their own trained artifacts"],
                artifact_note="Presence is not checkpoint provenance or physical safety validation")


def cache_features(c, args):
    from vla_precision.data.pipette_fulltask import FullTaskDataset
    from vla_precision.gr00t_rlt.features import FrozenGR00TFeatures
    assets = validate_assets(c)
    check_code(c)
    root = Path(c["prepared_root"]) / c["phase"]
    datasets = {s: FullTaskDataset(root, split=s,
                 left_thumb_relative=c["baseline"]["left_thumb"] == "chunk_relative")
                for s in ("train", "validation")}
    if any(ds.phase != int(c["phase"][1:]) for ds in datasets.values()):
        raise ValueError("Prepared dataset phase differs from GR00T checkpoint")
    if args.command == "probe":
        extractor = FrozenGR00TFeatures(c, args.device)
        data = extractor.extract(datasets["train"][0])
        print(json.dumps({k: list(v.shape) for k, v in data.items()}))
        if not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError("Non-finite probe features")
        return
    if not getattr(args, "allow_video_decode", False):
        for dataset in datasets.values():
            for ep in dataset.episodes:
                for view in c["views"]:
                    if not (dataset.shared / f"episode_{ep:06d}" / f"{view}.npy").is_file():
                        raise FileNotFoundError("Complete --cache-images preparation or explicitly pass --allow-video-decode")
    else:
        for dataset in datasets.values():
            for ep in dataset.episodes:
                for view in c["views"]:
                    if not ((dataset.shared / f"episode_{ep:06d}" / f"{view}.npy").is_file()
                            or (dataset.source / "videos/chunk-000" / f"observation.images.{view}"
                                / f"episode_{ep:06d}.mp4").is_file()):
                        raise FileNotFoundError(f"Missing source video/cache: episode {ep}, view {view}")
    out = Path(c["output"]) / "features"
    out.mkdir(parents=True, exist_ok=False)
    extractor = FrozenGR00TFeatures(c, args.device)
    records = []
    rng = np.random.default_rng(c["seed"])
    for split, dataset in datasets.items():
        count = min(len(dataset), args.max_observations if split == "train" else max(1, args.max_observations // 5))
        for index in sorted(rng.choice(len(dataset), count, replace=False)):
            ep, frame = dataset.samples[index]
            filename = f"{split}_ep{ep:06d}_f{frame:06d}.npz"
            features = extractor.extract(dataset[int(index)])
            np.savez_compressed(out / filename, **features)
            records.append(dict(file=filename, split=split, episode=ep, frame=frame, sha256=sha256(out / filename)))
            print(f"features {len(records)} {filename}", flush=True)
    write_json(out / "manifest.json", {"contract_sha256": c["contract_sha256"], "assets": assets,
               "data_manifest_sha256": sha256(root / "manifest.json"), "records": records})


def fit_token(c, args):
    import torch
    from vla_precision.gr00t_rlt.model import token_model
    check_code(c)
    assets = validate_assets(c)
    root = Path(c["output"]) / "features"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["contract_sha256"] != c["contract_sha256"] or manifest["assets"] != assets:
        raise ValueError("Feature cache belongs to another baseline/contract")
    for row in manifest["records"]:
        if sha256(root / row["file"]) != row["sha256"]:
            raise ValueError("Changed feature cache")
    train = [r for r in manifest["records"] if r["split"] == "train"]
    val = [r for r in manifest["records"] if r["split"] == "validation"]
    if not train or not val or {r["episode"] for r in train} & {r["episode"] for r in val}:
        raise ValueError("Require disjoint train/validation episodes")
    out = Path(c["output"]) / "token"
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    model = token_model(c).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["token"]["lr"])
    def batch(rows):
        values = []
        for row in rows:
            with np.load(root / row["file"], allow_pickle=False) as f:
                values.append((f["prefix"].copy(), f["mask"].copy()))
        length = max(len(p) for p, _ in values)
        p = np.zeros((len(rows), length, c["token"]["input_dim"]), np.float32)
        mask = np.zeros((len(rows), length), bool)
        for i, (prefix, valid) in enumerate(values):
            p[i, :len(prefix)] = prefix
            mask[i, :len(prefix)] = valid
        return torch.as_tensor(p, device=args.device), torch.as_tensor(mask, device=args.device)
    best = float("inf")
    updates = args.updates or c["token"]["updates"]
    for step in range(1, updates + 1):
        model.train()
        rows = [train[i] for i in rng.integers(len(train), size=c["token"]["batch_size"])]
        loss, _ = model.loss(*batch(rows))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite reconstruction loss")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0 or step == updates:
            model.eval()
            with torch.inference_mode():
                errors = [float(model.loss(*batch([row]))[0]) for row in val]
            score = float(np.mean(errors))
            metric = dict(step=step, reconstruction_loss=float(loss.detach()), validation_loss=score)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)
            if score < best:
                best = score
                torch.save(dict(state=model.state_dict(), contract_sha256=c["contract_sha256"],
                                feature_manifest_sha256=sha256(root / "manifest.json"), step=step,
                                validation_loss=best), out / "best.tmp")
                (out / "best.tmp").replace(out / "best.pt")
    write_json(out / "metadata.json", dict(contract_sha256=c["contract_sha256"], best_validation=best,
               checkpoint_sha256=sha256(out / "best.pt"), live_enabled=False))


def fit_heads(c, args):
    import torch
    from vla_precision.gr00t_rlt.model import Learner
    if args.replay is None:
        raise ValueError("--replay must be a validated bimanual sequential replay NPZ, not the old XYZ SQLite")
    check_code(c)
    validate_assets(c)
    token = Path(c["output"]) / "token/best.pt"
    saved_token = torch.load(token, map_location="cpu", weights_only=True)
    if saved_token["contract_sha256"] != c["contract_sha256"]:
        raise ValueError("Token belongs to another phase/experiment")
    meta = json.loads(args.replay.with_suffix(".json").read_text())
    with np.load(args.replay, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    r, d = validate_replay(data, meta, c, token_sha256=sha256(token))
    data.update(reward=r, discount=d)
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    learner = Learner(c, args.device)
    if args.resume:
        previous = torch.load(args.resume, map_location=args.device, weights_only=True)
        if previous["contract_sha256"] != c["contract_sha256"] or previous["token_sha256"] != sha256(token):
            raise ValueError("Resume contract/token mismatch")
        learner.load_state_dict(previous["learner"])
        torch.set_rng_state(previous["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in previous["cuda_rng"]])
        rng.bit_generator.state = previous["numpy_rng"]
    else:
        warm = torch.load(Path(c["output"]) / "bc/best.pt", map_location=args.device, weights_only=True)
        if warm["contract_sha256"] != c["contract_sha256"] or warm["token_sha256"] != sha256(token):
            raise ValueError("BC warm-start contract/token mismatch")
        learner.model.actor.load_state_dict(warm["actor"])
        learner.target.load_state_dict(learner.model.state_dict())
    out = Path(c["output"]) / "learner"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.json").exists() and not args.resume:
        raise ValueError("An existing learner requires explicit --resume; use a new output for a new run")
    updates = args.updates or c["learner"]["updates"]
    dest = out / f"step-{learner.step + updates:08d}.pt"
    if dest.exists():
        raise FileExistsError("Use --resume for an existing run; never overwrite weights")
    keys = ("token", "next_token", "state", "next_state", "reference", "next_reference",
            "action", "valid", "human", "bc_eligible", "reward", "discount")
    if "td_eligible" in data:
        keys += ("td_eligible",)
    for _ in range(updates):
        if getattr(args, "replay_check", None):
            args.replay_check()
        ids = rng.integers(len(r), size=c["learner"]["batch_size"])
        batch = {k: torch.as_tensor(data[k][ids], device=args.device,
                                   dtype=torch.bool if data[k].dtype == np.bool_ else torch.float32) for k in keys}
        metrics = learner.update(batch)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics, allow_nan=False) + "\n")
        if learner.step % 10 == 0:
            print(json.dumps(metrics), flush=True)
    if getattr(args, "replay_check", None):
        args.replay_check()
    tmp = dest.with_suffix(".tmp")
    torch.save(dict(learner=learner.state_dict(), contract_sha256=c["contract_sha256"], token_sha256=sha256(token),
                    replay_sha256=sha256(args.replay), replay_metadata_sha256=sha256(args.replay.with_suffix(".json")),
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                    live_enabled=False), tmp)
    tmp.replace(dest)
    write_json(out / "latest.json", dict(checkpoint=str(dest), sha256=sha256(dest), live_enabled=False))


def fit_bc(c, args):
    """Warm-start the direct actor on demonstrations before allowing Q updates."""
    import torch
    from vla_precision.gr00t_rlt.model import Heads, token_model
    check_code(c)
    validate_assets(c)
    root = Path(c["output"])
    token_path = root / "token/best.pt"
    saved = torch.load(token_path, map_location=args.device, weights_only=True)
    manifest_path = root / "features/manifest.json"
    if saved["contract_sha256"] != c["contract_sha256"] or saved["feature_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Token/data contract mismatch")
    manifest = json.loads(manifest_path.read_text())
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    token = token_model(c).to(args.device).eval().requires_grad_(False)
    token.load_state_dict(saved["state"])
    examples = {"train": [], "validation": []}
    with torch.inference_mode():
        for row in manifest["records"]:
            path = root / "features" / row["file"]
            if sha256(path) != row["sha256"]:
                raise ValueError("Changed feature cache")
            with np.load(path, allow_pickle=False) as f:
                z = token.encode_flat(torch.tensor(f["prefix"], device=args.device).float()[None],
                                      torch.tensor(f["mask"], device=args.device)[None])[0].cpu().numpy()
                examples[row["split"]].append((z, f["state"].copy(), f["reference"].copy(), f["demo_action"][:, :18].copy()))
    del token
    model = Heads(c).to(args.device)
    opt = torch.optim.Adam(model.actor.parameters(), lr=c["learner"]["actor_lr"])
    out = root / "bc"
    out.mkdir(parents=True, exist_ok=False)
    def batch(rows):
        return [torch.tensor(np.stack(x), device=args.device, dtype=torch.float32) for x in zip(*rows)]
    def loss(rows):
        z, s, ref, target = batch(rows)
        error = (model.act(z, s, ref) - target).square()
        return (error * model.active).sum() / (model.active.sum() * len(rows) * c["horizon"])
    best = float("inf")
    updates = args.updates or c["learner"]["bc_updates"]
    for step in range(1, updates + 1):
        rows = [examples["train"][i] for i in rng.integers(len(examples["train"]), size=c["learner"]["batch_size"])]
        value = loss(rows)
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite BC loss")
        opt.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0 or step == updates:
            with torch.inference_mode():
                scores = [float(loss([row])) for row in examples["validation"]]
            score = float(np.mean(scores))
            metrics = dict(step=step, bc_loss=float(value.detach()), validation_mse=score)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if score < best:
                best = score
                torch.save(dict(actor=model.actor.state_dict(), contract_sha256=c["contract_sha256"],
                                token_sha256=sha256(token_path), step=step, validation_mse=score,
                                live_enabled=False), out / "best.tmp")
                (out / "best.tmp").replace(out / "best.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["download", "check", "status", "probe", "cache", "token", "bc", "train"])
    p.add_argument("--config", type=Path, default=DEFAULT)
    p.add_argument("--phase", choices=["p1", "p2", "p3", "p4", "p5"], default="p1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-weights", action="store_true")
    p.add_argument("--max-observations", type=int, default=2048)
    p.add_argument("--allow-video-decode", action="store_true",
                   help="Extract sampled frames from original MP4s instead of a full 66-GiB image cache; slower")
    p.add_argument("--updates", type=int)
    p.add_argument("--replay", type=Path)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.max_observations < 1 or (args.updates is not None and args.updates < 1):
        p.error("Counts must be positive")
    c = load_config(args.config, args.phase)
    if args.command == "download":
        download(c)
    elif args.command == "check":
        check_code(c)
        print(json.dumps(validate_assets(c, verify_weights=args.verify_weights), indent=2))
    elif args.command == "status":
        print(json.dumps(readiness(c), indent=2))
    elif args.command in ("cache", "probe"):
        cache_features(c, args)
    elif args.command == "token":
        fit_token(c, args)
    elif args.command == "bc":
        fit_bc(c, args)
    else:
        fit_heads(c, args)


if __name__ == "__main__":
    main()
