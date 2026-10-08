"""Pinned phase contracts, elapsed-time reward and strict offline replay schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

WORKSPACE = Path(__file__).resolve().parents[4]
LEGACY = WORKSPACE / "VLA-Precision/configs/pipette_rl/gr00t_fulltask_rlt.yaml"
DEFAULT = WORKSPACE / "VLA-Precision/configs/pipette_rl/gr00t_fulltask_v2_rlt.yaml"
V2_PHASE_HELD = {"p1": [12, 17], "p2": [13, 14, 15, 16, 17],
                 "p3": [12, 13, 14, 15, 16, 17], "p4": [13, 14, 15, 16, 17],
                 "p5": [12, 17]}

# The P1 artifacts were trained with this registry-only working-tree patch.
# Committing it changes HEAD, not model code or the serialized data contract.
# Accept only the reviewed exact revision, never arbitrary descendants.
REGISTRY_BASE = "217eeb95bd359e31254e1c09ddfb04216711ec26"
REGISTRY_COMMIT = "2c52f5f10fe232bd875bd0bdad8454775154bd58"


def check_code(c):
    for repo in ("starvla", "rlinf"):
        actual = subprocess.check_output(
            ["git", "-C", c[repo + "_root"], "rev-parse", "HEAD"], text=True).strip()
        accepted = {c[repo + "_commit"]}
        if repo == "starvla" and c["starvla_commit"] == REGISTRY_BASE:
            accepted.add(REGISTRY_COMMIT)
        if actual not in accepted:
            raise ValueError(f"{repo} commit differs from validated recipe: {actual}")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path=DEFAULT, phase="p1", workspace=WORKSPACE):
    c = yaml.safe_load(Path(path).read_text())
    if c["schema"] not in ("gr00t_fulltask_rlt.v1", "gr00t_fulltask_rlt.v2") or phase not in c["phases"]:
        raise ValueError("Unsupported GR00T RLT recipe/phase")
    c["phase"] = phase
    phase_config = c.pop("phases")[phase]
    if c["schema"] == "gr00t_fulltask_rlt.v1":
        c["baseline"] = phase_config
        c["held_action_indices"] = []
    else:
        c["held_action_indices"] = phase_config["held_action_indices"]
        if (len(set(c["held_action_indices"])) != len(c["held_action_indices"])
                or any(type(i) is not int or not 0 <= i < 18 for i in c["held_action_indices"])):
            raise ValueError("Invalid held action indices")
        if c["held_action_indices"] != V2_PHASE_HELD[phase]:
            raise ValueError("v2 phase hand mask differs from the deployed bridge contract")
    if (c["state_dim"], c["model_action_dim"], c["actor_action_dim"], c["horizon"]) != (59, 32, 18, 30):
        raise ValueError("Expected state59, full32, actor18, horizon30")
    if c["views"] != ["rgb", "wrist_left"] or c["joint_seed_policy"] != "frozen_reference":
        raise ValueError("Unexpected camera/joint-seed contract")
    if c["live_enabled"] or not c["freeze_gr00t"] or not c["freeze_token_during_rl"]:
        raise ValueError("Require frozen GR00T/token and live_enabled=false; live collection uses an explicit separate entrypoint")
    if c["execution_contract"] != "sequential_chunk_no_ensemble_v1":
        raise ValueError("Asynchronous/ensembled replay requires its own scheduler-aware adapter")
    if c["reward"]["clock"] != "monotonic_elapsed" or not c["reward"]["terminal_replaces_last_tick"]:
        raise ValueError("Unsupported reward clock/terminal convention")
    expected_thumb = ("absolute" if c["schema"] == "gr00t_fulltask_rlt.v1" and phase == "p1"
                      else "chunk_relative")
    if c["baseline"]["left_thumb"] != expected_thumb:
        raise ValueError("Wrong left-thumb semantics for phase")
    c["contract_sha256"] = digest(c)
    for name in ("models_root", "prepared_root", "output_root", "starvla_root", "rlinf_root"):
        c[name] = str((Path(workspace) / c[name]).resolve())
    c["run_root"] = str(Path(c["models_root"]) / c["baseline"]["repo"].split("/")[-1] / "gr00t")
    c["checkpoint"] = str(Path(c["run_root"]) / "checkpoints" / f"steps_{c['baseline']['step']}_pytorch_model.pt")
    c["output"] = str(Path(c["output_root"]) / phase)
    stats_path = Path(c["run_root"]) / "dataset_statistics.json"
    if stats_path.is_file():
        action = json.loads(stats_path.read_text())[c["unnorm_key"]]["action"]
        active = np.asarray(action["q99"])[:18] != np.asarray(action["q01"])[:18]
        active[c["held_action_indices"]] = False
        c["action_active"] = active.tolist()
        if len(c["action_active"]) != 18 or not any(c["action_active"]):
            raise ValueError("Expected nonempty 18-channel physical control mask")
    return c


def reward_and_discount(start_ns, end_ns, outcome, reward, *, terminal_ns=None):
    """Match jm_pipette bc_async reward clock, independent of proposal row count.

    outcome is continuing/success/failure. Discard/abort are not RL terminals.
    Success/failure replaces the final tick penalty; human labelling wait counts.
    """
    if outcome not in ("continuing", "success", "failure"):
        raise ValueError("Discarded, aborted or unlabelled terminal episode is not trainable")
    if not np.isfinite([start_ns, end_ns]).all() or end_ns <= start_ns:
        raise ValueError("Require increasing monotonic timestamps")
    terminal = outcome != "continuing"
    if terminal_ns is not None:
        if not terminal or not np.isfinite(terminal_ns) or terminal_ns < end_ns:
            raise ValueError("Invalid terminal label timestamp")
        end_ns = terminal_ns
    ticks = max(1, round((end_ns - start_ns) * reward["rate_hz"] / 1e9))
    gamma, penalty = reward["gamma"], reward["step_penalty"]
    if not 0 < gamma <= 1 or not np.isfinite(list(reward[k] for k in
            ("gamma", "step_penalty", "success_reward", "failure_reward", "rate_hz"))).all():
        raise ValueError("Invalid reward configuration")
    n = ticks - int(terminal)
    total = penalty * (n if gamma == 1 else -np.expm1(n * np.log(gamma)) / (1 - gamma))
    if terminal:
        total += gamma ** (ticks - 1) * reward[f"{outcome}_reward"]
    return float(total), 0.0 if terminal else float(gamma ** ticks)


def validate_assets(c, *, verify_weights=False):
    """Check metadata without instantiating GR00T or downloading base weights."""
    root = Path(c["run_root"])
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    stats = json.loads((root / "dataset_statistics.json").read_text())[c["unnorm_key"]]
    f = cfg["framework"]
    if f["name"] != "CosmosGR00TN1d7" or f["qwenvl"]["select_layer"] != c["feature_layer"]:
        raise ValueError("Wrong GR00T backbone")
    if any(f["action_model"][k] != v for k, v in
           (("state_dim", 59), ("action_dim", 32), ("action_horizon", 30))):
        raise ValueError("Wrong baseline state/action layout")
    relative = cfg["datasets"]["vla_data"]["action_mode_apply_keys"]
    if ("action.left_hand_rel" in relative) != (c["baseline"]["left_thumb"] == "chunk_relative"):
        raise ValueError("Baseline left-thumb action semantics differ from recipe")
    for key, width in (("state", 59), ("action", 32)):
        for quantile in ("q01", "q99"):
            values = np.asarray(stats[key][quantile])
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError("Invalid normalization statistics")
    checkpoint = Path(c["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    receipt = json.loads((root / "download_receipt.json").read_text())
    if receipt["repo"] != c["baseline"]["repo"] or receipt["revision"] != c["baseline"]["revision"]:
        raise ValueError("Checkpoint revision mismatch")
    weight_entry = f"checkpoints/{checkpoint.name}"
    required = {"config.yaml", "dataset_statistics.json", weight_entry}
    if not required <= set(receipt["files"]):
        raise ValueError("Download receipt is missing required artifacts")
    if (c["baseline"].get("weights_sha256")
            and receipt["files"][weight_entry]["sha256"] != c["baseline"]["weights_sha256"]):
        raise ValueError("Weights do not match pinned v2 checkpoint")
    for name, expected in receipt["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Incomplete artifact: {name}")
        if (verify_weights or name != f"checkpoints/{checkpoint.name}") and sha256(path) != expected["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return {"phase": c["phase"], "checkpoint": str(checkpoint), "contract_sha256": c["contract_sha256"],
            "weights_sha256": receipt["files"][f"checkpoints/{checkpoint.name}"]["sha256"],
            "statistics_sha256": sha256(root / "dataset_statistics.json"),
            "action_active": c["action_active"], "held_action_indices": c["held_action_indices"],
            "reward": c["reward"], "live_enabled": False}


def validate_replay(data, meta, c, *, token_sha256):
    """Reject old XYZ replay, unproven execution, stale tokens and ambiguous masks.

    This is an interchange for future sequential collection, NOT an importer of
    jm_pipette's asynchronous QwenOFT buffer. Executed reference seeds are kept.
    """
    for k, expected in (("schema", "gr00t_rlt_replay.v1"), ("phase", c["phase"]),
                        ("contract_sha256", c["contract_sha256"]),
                        ("execution_contract", c["execution_contract"]), ("token_sha256", token_sha256)):
        if meta.get(k) != expected:
            raise ValueError(f"Replay mismatch: {k}")
    if meta.get("synthetic", False) or not meta.get("completed_episodes_only", False):
        raise ValueError("Only completed real episodes may train a real learner")
    n = len(data["state"])
    if not n:
        raise ValueError("Empty replay")
    shapes = {"state": (n, 59), "next_state": (n, 59),
              "token": (n, c["token"]["embed_dim"]), "next_token": (n, c["token"]["embed_dim"]),
              "reference": (n, 30, 32), "next_reference": (n, 30, 32),
              "action": (n, 30, 18), "valid": (n, 30, 18),
              "human": (n, 30, 18), "bc_eligible": (n, 30, 18)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid replay array: {key}, expected {shape}")
    for key in ("valid", "human", "bc_eligible"):
        if data[key].dtype != np.bool_:
            raise ValueError(f"Mask must be boolean: {key}")
    if not data["valid"].reshape(n, -1).any(1).all():
        raise ValueError("Empty action window")
    active = np.asarray(c.get("action_active", [True] * 18))
    if data["valid"][..., ~active].any():
        raise ValueError("Zero-span or phase-held action channels cannot be RL degrees of freedom")
    if (data["human"] & ~data["valid"]).any() or (data["bc_eligible"] & ~data["valid"]).any():
        raise ValueError("Supervision outside valid action window")
    if "td_eligible" in data:
        td = data["td_eligible"]
        if td.dtype != np.bool_ or td.shape != (n,):
            raise ValueError("td_eligible must be a per-window boolean")
        if meta.get("collector") == "bridge_sequential_v1" and (td & data["human"].any(axis=(1, 2))).any():
            raise ValueError("PICO controller windows are BC-only, not autonomous Bellman samples")
    elif meta.get("collector") == "bridge_sequential_v1":
        raise ValueError("Live replay must state TD eligibility")
    rewards, discounts = [], []
    for key in ("start_ns", "end_ns", "terminal_ns", "outcome", "episode_outcome"):
        if data[key].shape != (n,):
            raise ValueError(f"Invalid per-transition metadata: {key}")
    for i in range(n):
        if data["episode_outcome"][i] not in ("success", "failure"):
            raise ValueError("Discard/abort/incomplete episode must be excluded")
        outcome = str(data["outcome"][i])
        if outcome != "continuing" and outcome != data["episode_outcome"][i]:
            raise ValueError("Terminal outcome disagrees with episode")
        r, d = reward_and_discount(int(data["start_ns"][i]), int(data["end_ns"][i]), outcome,
                                  c["reward"], terminal_ns=int(data["terminal_ns"][i]) if outcome != "continuing" else None)
        rewards.append(r)
        discounts.append(d)
    return np.asarray(rewards, np.float32), np.asarray(discounts, np.float32)
