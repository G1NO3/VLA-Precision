"""Frozen-FOCUS collection -> exact cached features -> token-bound replay.

No robot I/O. Images are the raw ego and the actual served crop at each policy
observation, not continuous video. Per-row command audits live in SQLite.
"""
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

import numpy as np

from .core import check_code, is_focus, sha256, validate_assets


def bootstrap_token(c):
    return "bootstrap:" + c["contract_sha256"]


def check_bootstrap(c):
    if not is_focus(c):
        raise ValueError("Bootstrap requires the FOCUS-v6 recipe")
    check_code(c)
    assets = validate_assets(c)
    # Offline existence/import check only; no network, model/GPU or camera.
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2VideoModel, Sam2VideoProcessor
    import cv2
    for repo, processor in (("facebook/sam2.1-hiera-small", Sam2VideoProcessor),
                            ("google/owlv2-base-patch16-ensemble", Owlv2Processor)):
        root = Path(snapshot_download(repo, local_files_only=True))
        processor.from_pretrained(root, local_files_only=True)
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            if not list(f.keys()):
                raise ValueError("Empty detector weights")
    for n in (62, 262):
        path = Path(c["starvla_root"]) / f"deployment/model_server/focus_exemplars/holder_exemplar_{n}.png"
        if cv2.imread(str(path)) is None:
            raise ValueError(f"Missing focus exemplar: {path}")
    return dict(assets=assets, bootstrap_ready=True, hardware_validated=False)


def save_observation(root, features, c, episode, *, key=None, compressed=True):
    root = Path(root) / "observations"
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 << 30:
        raise OSError("FOCUS capture refuses <2 GiB free; pause collection and free storage")
    key = key or uuid.uuid4().hex
    if len(key) != 32 or any(ch not in "0123456789abcdef" for ch in key):
        raise ValueError("Invalid observation identity")
    path = root / (key + ".npz")
    tmp = root / (key + ".pending")
    arrays = {**{k:v for k,v in features.items() if not k.startswith("_")}, "contract_sha256": np.asarray(c["contract_sha256"]),
              "request_episode": np.asarray(episode), "saved_monotonic_ns": np.int64(time.monotonic_ns())}
    with tmp.open("xb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    # Persist the directory entry before a replay row may refer to it.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(observation_file="observations/" + path.name, observation_sha256=sha256(path))


def load_observation(root, row, c, prefix=""):
    rel = str(row[prefix + "observation_file"].item())
    path = (Path(root) / rel).resolve()
    if path.parent != (Path(root) / "observations").resolve() or path.suffix != ".npz":
        raise ValueError("Observation path escapes capture directory")
    if prefix + "observation_receipt" in row:
        receipt_path = (Path(root) / str(row[prefix + "observation_receipt"].item())).resolve()
        if receipt_path != path.with_suffix(".receipt.json"):
            raise ValueError("Invalid observation receipt path")
        if not receipt_path.is_file():
            raise ValueError("Observation capture pending or failed; retry only after writer drains")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "focus_capture_receipt.v1"
                or receipt.get("observation_file") != rel
                or receipt.get("contract_sha256") != c["contract_sha256"]):
            raise ValueError("Observation receipt mismatch")
        expected_hash = receipt["observation_sha256"]
    else:
        expected_hash = str(row[prefix + "observation_sha256"].item())
    if sha256(path) != expected_hash:
        raise ValueError("Observation hash mismatch")
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    if str(data["contract_sha256"].item()) != c["contract_sha256"]:
        raise ValueError("Observation recipe mismatch")
    for key in ("state", "reference"):
        if not np.array_equal(data[key], row[prefix + key]):
            raise ValueError("Observation/replay alignment mismatch: " + prefix + key)
    p, mask = data["prefix"], data["mask"]
    if (p.ndim != 2 or p.shape[1] != c["token"]["input_dim"]
            or len(p) > c["token"]["prefix_seq_len"] or not np.isfinite(p).all()
            or mask.shape != p.shape[:1] or mask.dtype != np.bool_ or not mask.any()):
        raise ValueError("Invalid captured visual features")
    return data


def readonly_db(root):
    path = (Path(root) / "replay.sqlite3").resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def source_check(manifest):
    """A later discard revokes bootstrap training too, not just RL export."""
    source = manifest.get("bootstrap_source")
    if source is None:
        return
    db = readonly_db(source["root"])
    try:
        outcomes = dict(db.execute("SELECT id,outcome FROM episodes"))
        if any(outcomes.get(e) != outcome for e, outcome in source["episodes"].items()):
            raise ValueError("Bootstrap source episode changed/discarded; rebuild the cache in a new output")
    finally:
        db.close()


def cache_bootstrap(c, run):
    """Build initialization data without rerunning SAM or the VLM; split by episode."""
    check_code(c)
    assets = validate_assets(c)
    db = readonly_db(run)
    try:
        db.execute("BEGIN")
        episodes = db.execute("SELECT id,outcome FROM episodes WHERE phase=? AND contract=? AND token=? "
            "AND outcome IN ('success','failure') AND EXISTS (SELECT 1 FROM transitions t WHERE t.episode=episodes.id) "
            "ORDER BY created,id", (c["phase"], c["contract_sha256"], bootstrap_token(c))).fetchall()
        if len(episodes) < 2:
            raise ValueError("Collect at least TWO labelled episodes for disjoint train/validation; more is recommended")
        ids = np.random.default_rng(c["seed"]).permutation(len(episodes))
        validation = {episodes[i][0] for i in ids[:max(1, len(episodes)//5)]}
        out = Path(c["output"]) / "features"
        out.mkdir(parents=True, exist_ok=False)
        records = []
        for eid, outcome in episodes:
            for seq, blob in db.execute("SELECT seq,payload FROM transitions WHERE episode=? ORDER BY seq", (eid,)):
                row = unpack(blob)
                f = load_observation(run, row, c)
                valid = row["bc_eligible"] & row["valid"]
                if not valid.any():
                    continue
                f["demo_action"] = np.concatenate((row["action"], row["reference"][:, 18:]), axis=-1)
                f["demo_valid"] = valid
                name = f"{eid}_{seq:06d}.npz"
                with (out/name).open("xb") as stream:
                    np.savez_compressed(stream, **f)
                records.append(dict(file=name, split="validation" if eid in validation else "train",
                                    episode=eid, frame=seq, sha256=sha256(out/name)))
        if {r["split"] for r in records} != {"train", "validation"}:
            raise ValueError("Both splits need BC-eligible published windows; collect more episodes")
        manifest = dict(contract_sha256=c["contract_sha256"], assets=assets, records=records,
                        bootstrap_source=dict(root=str(Path(run).resolve()), episodes=dict(episodes)),
                        target="published PICO command labels / autonomous reference distillation; masked padding")
        source_check(manifest)
        tmp = out / "manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(out / "manifest.json")
        return dict(episodes=len(episodes), observations=len(records), validation_episodes=len(validation), path=str(out))
    finally:
        db.close()


def encode_bootstrap(c, run, device="cpu"):
    """Atomically replace placeholder tokens IN PLACE, preserving IDs and discards.

    No duplicate replay/ledger or lossy image recomputation. Call under the
    training lock with collection paused. A failed transaction changes nothing.
    """
    import torch
    from .model import token_model
    check_code(c)
    validate_assets(c)
    path = Path(c["output"]) / "token/best.pt"
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["contract_sha256"] != c["contract_sha256"]:
        raise ValueError("Token recipe mismatch")
    model = token_model(c).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["state"])
    token_sha = sha256(path)
    # Existing database only. Never create an empty replay by typo.
    db = sqlite3.connect((Path(run).resolve()/"replay.sqlite3").as_uri()+"?mode=rw", uri=True)
    count = 0
    try:
        db.execute("BEGIN IMMEDIATE")
        episodes = db.execute("SELECT id FROM episodes WHERE phase=? AND contract=? AND token=? "
            "AND outcome IN ('success','failure')", (c["phase"], c["contract_sha256"], bootstrap_token(c))).fetchall()
        for (eid,) in episodes:
            payloads = db.execute("SELECT seq,payload FROM transitions WHERE episode=? ORDER BY seq", (eid,)).fetchall()
            for seq, blob in payloads:
                row = unpack(blob)
                for prefix in ("", "next_"):
                    f = load_observation(run, row, c, prefix)
                    with torch.inference_mode():
                        z = model.encode_flat(torch.as_tensor(f["prefix"], device=device).float()[None],
                                              torch.as_tensor(f["mask"], device=device)[None])[0].cpu().numpy()
                    if z.shape != (c["token"]["embed_dim"],) or not np.isfinite(z).all():
                        raise ValueError("Invalid encoded token")
                    row[prefix+"token"] = z
                out = io.BytesIO()
                np.savez_compressed(out, **row)
                db.execute("UPDATE transitions SET payload=? WHERE episode=? AND seq=?", (out.getvalue(), eid, seq))
                count += 1
            db.execute("UPDATE episodes SET token=? WHERE id=?", (token_sha, eid))
            db.execute("INSERT INTO events VALUES(?,?,'bootstrap_encoded',?)",
                       (time.time(), eid, json.dumps(dict(token_sha256=token_sha))))
        db.commit()
        return dict(episodes=len(episodes), windows=count, token_sha256=token_sha)
    finally:
        db.close()
