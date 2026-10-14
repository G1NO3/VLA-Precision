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
PI05 = WORKSPACE / "VLA-Precision/configs/pipette_rl/pi05_fulltask_v5_rlt.yaml"
FOCUS = WORKSPACE / "VLA-Precision/configs/pipette_rl/pi05_fulltask_v6_focus_rlt.yaml"
EDITOR = WORKSPACE / "VLA-Precision/configs/pipette_rl/pi05_fulltask_v6_focus_editor.yaml"
CONTINUOUS = WORKSPACE / "VLA-Precision/configs/pipette_rl/pi05_fulltask_v6_focus_continuous.yaml"


def is_shared(c):
    return c["schema"] == "pi05_fulltask_editor.v3"


def phase_active(c, phase):
    """Physical zero-span mask AND the current phase's held-hand mask."""
    if not is_shared(c):
        return c["action_active"]
    if phase not in V2_PHASE_HELD:
        raise ValueError("Unknown phase")
    active = np.asarray(c.get("action_active", [True] * 18), dtype=bool).copy()
    active[V2_PHASE_HELD[phase]] = False
    return active.tolist()


def is_editor(c):
    return c["schema"] in ("pi05_fulltask_editor.v1", "pi05_fulltask_editor.v2", "pi05_fulltask_editor.v3")


def is_async_editor(c):
    return c["schema"] in ("pi05_fulltask_editor.v2", "pi05_fulltask_editor.v3")


def representation_sha(c):
    # Legacy replay column name is token_sha256; no learned token in editor mode.
    return "editor_pool:" + c["contract_sha256"] if is_editor(c) else sha256(Path(c["output"]) / "token/best.pt")


def is_focus(c):
    return c.get("focus", {}).get("enabled", False)


def is_pi05(c):
    return c["schema"] == "pi05_fulltask_rlt.v1" or is_editor(c)


def bridge_profile(c):
    if is_focus(c):
        return "starvla_pi05_pipette_5taskv6_egofocus_native_inspire"
    return "starvla_pi05_pipette_5taskv5_inspire" if is_pi05(c) else "gr00t_pipette_fulltask_v2"
V2_PHASE_HELD = {"p1": [12, 17], "p2": [13, 14, 15, 16, 17],
                 "p3": [12, 13, 14, 15, 16, 17], "p4": [13, 14, 15, 16, 17],
                 "p5": [12, 17]}

# The P1 artifacts were trained with this registry-only working-tree patch.
# Committing it changes HEAD, not model code or the serialized data contract.
# Accept only the reviewed exact revision, never arbitrary descendants.
REGISTRY_BASE = "217eeb95bd359e31254e1c09ddfb04216711ec26"
REGISTRY_COMMIT = "2c52f5f10fe232bd875bd0bdad8454775154bd58"
# Reviewed starVLA_dev merge: new PI05 registries and (H,W)->PIL(W,H)
# resize correction. The pinned GR00T v2 uses square views, so its input
# geometry, model and action contract are unchanged. No descendant wildcard.
PI05_COMPAT_COMMIT = "11ba155b90f4ffd7f29814b991bc4ad76838f3cd"
# Reviewed FOCUS timing-only commit: no weights, crop geometry, action or replay
# schema changes. Keep the recipe hash stable for already collected episodes.
FOCUS_BASE_COMMIT = "de0a9a3bade4eb57d3ec53a82416aaa198dcb68b"
FOCUS_TIMING_COMMIT = "0bc36a499a677ef2060a57c98c503120c8da5a1d"


def check_code(c):
    for repo in (("starvla",) if is_editor(c) else ("starvla", "rlinf")):
        actual = subprocess.check_output(
            ["git", "-C", c[repo + "_root"], "rev-parse", "HEAD"], text=True).strip()
        accepted = {c[repo + "_commit"]}
        if repo == "starvla" and c["starvla_commit"] == REGISTRY_BASE:
            accepted.update((REGISTRY_COMMIT, PI05_COMPAT_COMMIT))
        if repo == "starvla" and is_focus(c) and c["starvla_commit"] == FOCUS_BASE_COMMIT:
            accepted.add(FOCUS_TIMING_COMMIT)
        if actual not in accepted:
            raise ValueError(f"{repo} commit differs from validated recipe: {actual}; "
                             f"accepted: {', '.join(sorted(accepted))}. "
                             "Review code compatibility before updating the exact revision allowlist.")


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
    if c["schema"] not in ("gr00t_fulltask_rlt.v1", "gr00t_fulltask_rlt.v2", "pi05_fulltask_rlt.v1", "pi05_fulltask_editor.v1", "pi05_fulltask_editor.v2", "pi05_fulltask_editor.v3") or phase not in c["phases"]:
        raise ValueError("Unsupported full-task RLT recipe/phase")
    c["phase"] = phase
    phases = c.pop("phases")
    phase_config = phases[phase]
    if is_shared(c):
        if {k: v["held_action_indices"] for k, v in phases.items()} != V2_PHASE_HELD:
            raise ValueError("Shared editor requires all five phase hand masks")
        c["phase_held_action_indices"] = V2_PHASE_HELD
        if (c["reward"].get("outcome_scope") != "full_task_episode"
                or c["reward"].get("phase_reward") != 1.0
                or c["reward"].get("success_reward") != 11.0):
            raise ValueError("Continuous task requires +1 phase / +11 final reward")
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
    expected_views = ["rgb", "focus"] if is_focus(c) else ["rgb", "wrist_left"]
    if c["views"] != expected_views or c["joint_seed_policy"] != "frozen_reference":
        raise ValueError("Unexpected camera/joint-seed contract")
    if is_focus(c) and (not is_pi05(c) or c["focus"].get("mode") != "causal_background_stream"
                        or c["focus"].get("sam") != "small" or c["focus"].get("rate_hz") != 10.0):
        raise ValueError("FOCUS RL requires PI05 and the pinned causal 10-Hz background tracker")
    frozen = c.get("freeze_backbone") if is_pi05(c) else c.get("freeze_gr00t")
    if c["live_enabled"] or not frozen or not c["freeze_token_during_rl"]:
        raise ValueError("Require frozen backbone/token and live_enabled=false; live collection uses an explicit separate entrypoint")
    if is_pi05(c) and (c["feature_layer"] != "prefix_final" or c["baseline"].get("state_input") is not False):
        raise ValueError("PI05 v5 requires final prefix features and a no-state frozen reference")
    if is_pi05(c) and c.get("image_source") != "raw_video":
        raise ValueError("PI05 requires original pixels, not the GR00T resize270 image cache")
    from .act_context import CONTRACT, CONTEXT_DIM, SUPPORTED_ACTION_HZ
    if is_async_editor(c):
        # action_hz is the PLAYBACK rate of the 60 Hz-trained 30-row chunk
        # (one row per publish tick): 30 = half-speed execution. It is part of
        # the recipe, hence of the contract hash: never mix replay across it.
        if (c["execution_contract"] != CONTRACT or c.get("scheduler") != dict(
                context_dim=CONTEXT_DIM, inference_hz=5, action_hz=c.get("scheduler", {}).get("action_hz"),
                mode="act", decay=0.01, handoff="latency")
                or c["scheduler"]["action_hz"] not in SUPPORTED_ACTION_HZ):
            raise ValueError("Async editor requires pinned ACT scheduler/context contract "
                             f"(action_hz one of {SUPPORTED_ACTION_HZ})")
    elif c["execution_contract"] != "sequential_chunk_no_ensemble_v1":
        raise ValueError("Asynchronous/ensembled replay requires its own scheduler-aware adapter")
    if is_editor(c):
        if (not is_focus(c) or c.get("editor", {}).get("features") != "pi05_two_view_spatial_2x2_v1"
                or c["token"]["embed_dim"] != 16384 or c["token"]["input_dim"] != 2048
                or c["learner"]["reference_dropout"] != 0
                or not 0 < c["editor"].get("residual_scale", 0) <= 1):
            raise ValueError("Invalid fixed-feature residual editor contract")
    if c["reward"]["clock"] != "monotonic_elapsed" or not c["reward"]["terminal_replaces_last_tick"]:
        raise ValueError("Unsupported reward clock/terminal convention")
    expected_thumb = ("absolute" if c["schema"] == "gr00t_fulltask_rlt.v1" and phase == "p1"
                      else "chunk_relative")
    if c["baseline"]["left_thumb"] != expected_thumb:
        raise ValueError("Wrong left-thumb semantics for phase")
    # Runtime instruction is not a different network/checkpoint/replay contract.
    contract = {k: v for k, v in c.items() if k not in ("phase", "held_action_indices")} if is_shared(c) else c
    c["contract_sha256"] = digest(contract)
    for name in ("models_root", "prepared_root", "output_root", "starvla_root", "rlinf_root"):
        c[name] = str((Path(workspace) / c[name]).resolve())
    c["run_root"] = str(Path(c["models_root"]) / c["baseline"]["repo"].split("/")[-1] / ("pi05" if is_pi05(c) else "gr00t"))
    c["checkpoint"] = str(Path(c["run_root"]) / "checkpoints" / f"steps_{c['baseline']['step']}_pytorch_model.pt")
    c["output"] = str(Path(c["output_root"]) / ("shared" if is_shared(c) else phase))
    stats_path = Path(c["run_root"]) / "dataset_statistics.json"
    if stats_path.is_file():
        action = json.loads(stats_path.read_text())[c["unnorm_key"]]["action"]
        active = np.asarray(action["q99"])[:18] != np.asarray(action["q01"])[:18]
        if not is_shared(c):
            active[c["held_action_indices"]] = False
        c["action_active"] = active.tolist()
        if len(c["action_active"]) != 18 or not any(c["action_active"]):
            raise ValueError("Expected nonempty 18-channel physical control mask")
    return c


def reward_and_discount(start_ns, end_ns, outcome, reward, *, terminal_ns=None, phase_reward_ns=0):
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
    if phase_reward_ns:
        if (reward.get("outcome_scope") != "full_task_episode"
                or not start_ns <= phase_reward_ns <= end_ns):
            raise ValueError("Invalid phase reward event timestamp")
        event_tick = max(0, round((phase_reward_ns - start_ns) * reward["rate_hz"] / 1e9) - 1)
        total += gamma ** event_tick * reward["phase_reward"]
    return float(total), 0.0 if terminal else float(gamma ** ticks)


def validate_assets(c, *, verify_weights=False):
    """Check metadata without instantiating GR00T or downloading base weights."""
    root = Path(c["run_root"])
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    stats = json.loads((root / "dataset_statistics.json").read_text())[c["unnorm_key"]]
    f = cfg["framework"]
    if is_pi05(c):
        if (f["name"] != "PI05" or f["discrete_state_input"] is not False
                or f["action_dim"] != 32 or f["action_horizon"] != 30
                or f["paligemma_config"]["width"] != c["token"]["input_dim"]
                or f["image_resolution"] != [224, 224] or f["num_inference_steps"] != 10):
            raise ValueError("Wrong PI05 v5 no-state backbone/action/view layout")
        if is_editor(c) and f.get("image_keys", ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]) != ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            raise ValueError("Editor requires pinned ego/focus/dummy image-token order")
    else:
        if f["name"] != "CosmosGR00TN1d7" or f["qwenvl"]["select_layer"] != c["feature_layer"]:
            raise ValueError("Wrong GR00T backbone")
        if any(f["action_model"][k] != v for k, v in
               (("state_dim", 59), ("action_dim", 32), ("action_horizon", 30))):
            raise ValueError("Wrong baseline state/action layout")
    relative = cfg["datasets"]["vla_data"]["action_mode_apply_keys"]
    if is_focus(c) and cfg["datasets"]["vla_data"]["data_mix"] != "unitree_g1_pipette_egofocus_5task447v12_pi05sq224_train_mix":
        raise ValueError("FOCUS checkpoint must consume ego + focus, not wrist")
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

    Sequential RLT and asynchronous ACT editor have distinct contracts. This
    does not import jm_pipette's different observation/action buffer.
    """
    for k, expected in (("schema", "gr00t_rlt_replay.v1"), ("phase", "fulltask" if is_shared(c) else c["phase"]),
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
    if is_async_editor(c):
        from .act_context import CONTEXT_DIM
        if meta.get("collector") != "bridge_async_act_v1" or meta.get("action_evidence") != "requested_chunk_not_act_blend":
            raise ValueError("Async editor needs proposal-level ACT replay, not sequential publication rows")
        shapes.update(scheduler_context=(n, CONTEXT_DIM), next_scheduler_context=(n, CONTEXT_DIM),
                      proposal_accepted=(n,), autonomy=(n,), next_autonomy=(n,))
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid replay array: {key}, expected {shape}")
    for key in ("valid", "human", "bc_eligible"):
        if data[key].dtype != np.bool_:
            raise ValueError(f"Mask must be boolean: {key}")
    if not data["valid"].reshape(n, -1).any(1).all():
        raise ValueError("Empty action window")
    active = np.asarray(c.get("action_active", [True] * 18))
    if is_shared(c):
        for name in ("phase", "next_phase"):
            a = data.get(name)
            if a is None or a.shape != (n,) or a.dtype.kind not in "iu" or not ((0 <= a) & (a < 5)).all():
                raise ValueError("Invalid phase label: " + name)
        active = np.asarray([phase_active(c, f"p{int(i)+1}") for i in data["phase"]])[:, None, :]
        stamps = data.get("phase_reward_ns", np.empty(0))
        if stamps.shape != (n,) or stamps.dtype.kind not in "iu" or (stamps < 0).any():
            raise ValueError("Missing phase reward events")
    if (data["valid"] & ~active).any():
        raise ValueError("Zero-span or phase-held action channels cannot be RL degrees of freedom")
    if (data["human"] & ~data["valid"]).any() or (data["bc_eligible"] & ~data["valid"]).any():
        raise ValueError("Supervision outside valid action window")
    if "td_eligible" in data:
        td = data["td_eligible"]
        if td.dtype != np.bool_ or td.shape != (n,):
            raise ValueError("td_eligible must be a per-window boolean")
        if meta.get("collector") in ("bridge_sequential_v1", "bridge_async_act_v1") and (td & data["human"].any(axis=(1, 2))).any():
            raise ValueError("PICO controller windows are BC-only, not autonomous Bellman samples")
        if is_async_editor(c):
            for name in ("proposal_accepted", "autonomy", "next_autonomy"):
                if data[name].dtype != np.bool_:
                    raise ValueError("ACT eligibility must be boolean")
            if (td & ~(data["proposal_accepted"] & data["autonomy"] & data["next_autonomy"])).any():
                raise ValueError("Cancelled/intervened ACT requests cannot be TD samples")
    elif meta.get("collector") in ("bridge_sequential_v1", "bridge_async_act_v1"):
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
                                  c["reward"], terminal_ns=int(data["terminal_ns"][i]) if outcome != "continuing" else None,
                                  phase_reward_ns=int(data["phase_reward_ns"][i]) if is_shared(c) else 0)
        if is_shared(c):
            p, nxt, event = int(data["phase"][i]), int(data["next_phase"][i]), int(data["phase_reward_ns"][i])
            if nxt not in (p, p+1) or nxt > 4 or (outcome == "continuing" and bool(event) != (nxt == p+1)):
                raise ValueError("Phase change must have exactly one success reward")
            if outcome == "success" and (p != 4 or event):
                raise ValueError("Final success is P5 +11, not an extra phase event")
        rewards.append(r)
        discounts.append(d)
    return np.asarray(rewards, np.float32), np.asarray(discounts, np.float32)
