"""Single-owner RLToken inference service. No robot or collection I/O."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .core import check_code, sha256, validate_assets, is_focus, is_editor, representation_sha, is_shared, phase_active
from .features import frozen_features
from .model import Heads, token_model, make_heads, spatial_features


def check_live_weights(config):
    """CPU-only preflight. Never instantiate the VLM, bind a port or touch replay."""
    check_code(config)
    assets = validate_assets(config)
    root = Path(config["output"])
    if is_editor(config):
        from .bootstrap import check_bootstrap
        check_bootstrap(config)
        policy = LivePolicy.__new__(LivePolicy)
        policy.c, policy.device, policy.token_sha = config, "cpu", representation_sha(config)
        policy._load_actor()
        return dict(assets=assets, actor=policy.actor_path, actor_kind=policy.actor_kind,
                    token_sha256=policy.token_sha, behavior_sha256=policy.behavior,
                    weights_ready=True, hardware_validated=False)
    if is_focus(config):
        for stage in ("token", "bc"):
            marker = json.loads((root / stage / "metadata.json").read_text())
            if marker["checkpoint_sha256"] != sha256(root / stage / "best.pt"):
                raise ValueError("FOCUS initialization incomplete or artifact changed: " + stage)
    token_path = root / "token/best.pt"
    token_sha = sha256(token_path)
    saved = torch.load(token_path, map_location="cpu", weights_only=True)
    if saved["contract_sha256"] != config["contract_sha256"]:
        raise ValueError("Token recipe mismatch")
    token_model(config).load_state_dict(saved["state"])
    # Reuse the exact publication/hash and actor/token checks of the server.
    policy = LivePolicy.__new__(LivePolicy)
    policy.c, policy.device, policy.token_sha = config, "cpu", token_sha
    policy.heads = Heads(config)
    policy._load_actor()
    return dict(assets=assets, actor=policy.actor_path, actor_kind=policy.actor_kind,
                token_sha256=token_sha, behavior_sha256=policy.behavior,
                weights_ready=True, hardware_validated=False)


class LivePolicy:
    def __init__(self, config, device="cuda", *, bootstrap=False, run=None):
        self.c, self.device = config, device
        self.bootstrap = bootstrap
        if bootstrap and (not is_focus(config) or is_editor(config)):
            raise ValueError("Reference-only bootstrap is explicit and FOCUS-v6 only")
        self.capture_root = Path(run).resolve() if run is not None and is_focus(config) else None
        if is_focus(config) and self.capture_root is None:
            raise ValueError("FOCUS requires a capture root for exact crops/features")
        check_code(config)
        validate_assets(config, verify_weights=True)
        root = Path(config["output"])
        self.token_path = root / "token/best.pt"
        if bootstrap:
            from .bootstrap import bootstrap_token
            self.token_sha = bootstrap_token(config)
            self.behavior = config["baseline"]["weights_sha256"]
            self.actor_path, self.actor_kind = config["checkpoint"], "frozen_reference"
        elif is_editor(config):
            check_live_weights(config)
            self.token_sha = representation_sha(config)
            self._load_actor()
        else:
            if is_focus(config):
                check_live_weights(config)
            self.token_sha = sha256(self.token_path)
            saved = torch.load(self.token_path, map_location=device, weights_only=True)
            if saved["contract_sha256"] != config["contract_sha256"]:
                raise ValueError("Token recipe mismatch")
            self.token = token_model(config).to(device).eval().requires_grad_(False)
            self.token.load_state_dict(saved["state"])
            self.heads = Heads(config).to(device).eval().requires_grad_(False)
            self._load_actor()
        self.episode = None
        self.features = frozen_features(config, device)
        self.features.keep_device_features = True
        self.capture_writer = None
        if self.capture_root is not None and is_editor(config):
            from .capture_writer import CaptureWriter
            self.capture_writer = CaptureWriter(self.capture_root)
        self.stats = json.loads((Path(config["run_root"])/"dataset_statistics.json").read_text())[config["unnorm_key"]]["action"]
        sys.path.insert(0, str(Path(config["starvla_root"]).parent / "VLAPolicyBridge"))

    def _load_actor(self):
        root = Path(self.c["output"])
        latest = root / "learner/latest.json"
        if is_editor(self.c):
            if not latest.exists():
                if (list((root / "learner").glob("step-*.pt"))
                        or getattr(self, "actor_kind", None) == "editor_rl"):
                    raise ValueError("Editor learner publication missing; refuse zero-policy fallback")
                if not hasattr(self, "heads"):
                    # Local seed scope: no effect on the frozen flow sampler RNG.
                    with torch.random.fork_rng(devices=[]):
                        torch.default_generator.manual_seed(self.c["seed"])
                        self.heads = make_heads(self.c).to(self.device).eval().requires_grad_(False)
                self.actor_path, self.actor_kind = "zero-initialized residual editor", "editor_zero"
                self.behavior = "editor_zero:" + self.c["contract_sha256"]
                return
            entry = json.loads(latest.read_text())
            path = Path(entry["checkpoint"]).resolve()
            if path.parent != (root / "learner").resolve() or sha256(path) != entry["sha256"]:
                raise ValueError("Invalid published editor checkpoint")
            saved = torch.load(path, map_location=self.device, weights_only=True)
            if saved["contract_sha256"] != self.c["contract_sha256"] or saved["token_sha256"] != self.token_sha:
                raise ValueError("Editor actor/representation contract mismatch")
            if not hasattr(self, "heads"):
                self.heads = make_heads(self.c).to(self.device).eval().requires_grad_(False)
            self.heads.load_state_dict(saved["learner"]["model"])
            if any(not torch.isfinite(v).all() for v in self.heads.state_dict().values()):
                raise ValueError("Non-finite editor weights")
            self.behavior = sha256(path)
            self.actor_path, self.actor_kind = str(path), "editor_rl"
            return
        path = root / "bc/best.pt"
        kind = "bc"
        if latest.is_file():
            entry = json.loads(latest.read_text())
            path = Path(entry["checkpoint"]).resolve()
            if path.parent != (root / "learner").resolve() or sha256(path) != entry["sha256"]:
                raise ValueError("Invalid published learner checkpoint")
            kind = "rl"
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved["contract_sha256"] != self.c["contract_sha256"] or saved["token_sha256"] != self.token_sha:
            raise ValueError("Actor/token contract mismatch; no random-policy fallback")
        if kind == "rl":
            self.heads.load_state_dict(saved["learner"]["model"])
        else:
            self.heads.actor.load_state_dict(saved["actor"])
        self.behavior = sha256(path)
        self.actor_path, self.actor_kind = str(path), kind

    def reset(self, options=None):
        return {"status": "ok"}  # reloading requires an explicit NEW episode id

    def close(self):
        writer = getattr(self, "capture_writer", None)
        if writer is not None:
            writer.close()

    @torch.inference_mode()
    def get_action(self, observation, options=None):
        started = time.perf_counter()
        writer = getattr(self, "capture_writer", None)
        if writer is not None:
            writer.check()
        from vla_policy_bridge.pi05_fulltask import observation_to_inputs, INSTRUCTIONS
        data = observation_to_inputs(observation, ego_only=is_focus(self.c))
        phase = "p" + str(INSTRUCTIONS.index(data["prompt"]) + 1)
        if not is_shared(self.c) and phase != self.c["phase"]:
            raise ValueError(f"Server prepared for {self.c['phase']}, not {phase}; pause and select that phase's server")
        episode = (options or {}).get("episode")
        if not isinstance(episode, str) or not episode:
            raise ValueError("RL server requires an explicit episode id; refuse ordinary IL clients")
        # Request generation changes at a phase boundary, but the shared actor
        # must remain pinned for the entire physical trajectory (and pauses).
        rollout = (options or {}).get("rollout") if is_shared(self.c) else episode
        if not isinstance(rollout, str) or not rollout:
            raise ValueError("Invalid full-task rollout identity")
        if rollout != self.episode:
            if not getattr(self, "bootstrap", False):
                self._load_actor()
            if hasattr(self.features, "reset"):
                self.features.reset()
            self.episode = rollout
        sample = {"observation.state": data["state"], "task": data["prompt"],
                  "observation.images.rgb": data["base_0_rgb"]}
        if not is_focus(self.c):
            sample["observation.images.wrist_left"] = data["left_wrist_0_rgb"]
        prepared = time.perf_counter()
        f = self.features.extract(sample)
        extracted = time.perf_counter()
        if getattr(self, "bootstrap", False):
            # Explicit non-trainable placeholder; never an untrained network action.
            z = torch.zeros((1, self.c["token"]["embed_dim"]), device=self.device)
            action = f["reference"][:, :18].copy()
        else:
            prefix = (f["_prefix_device"] if "_prefix_device" in f else
                      torch.as_tensor(f["prefix"], device=self.device)).float()[None]
            mask = (f["_mask_device"] if "_mask_device" in f else
                    torch.as_tensor(f["mask"], device=self.device))[None]
            z = spatial_features(prefix, mask) if is_editor(self.c) else self.token.encode_flat(prefix, mask)
            phase_args = {"phase": torch.tensor([int(phase[1])-1], device=self.device)} if is_shared(self.c) else {}
            action = self.heads.act(z, torch.as_tensor(f["state"], device=self.device)[None],
                               torch.as_tensor(f["reference"], device=self.device)[None], **phase_args)[0].cpu().numpy()
        acted = time.perf_counter()
        # RLT clamps absolute active actions; editor bounds only the residual.
        # Both preserve inactive channels and all 14 frozen IK seed channels.
        action = action.astype(np.float32)
        physical = self.features.physical_actions(action, f["reference"])
        keys = ("left_wrist_rel6", "right_wrist_rel6", "left_hand_rel", "right_hand_abs", "left_arm_abs", "right_arm_abs")
        offsets = (0, 6, 12, 13, 18, 25, 32)
        arrays = {k: physical[None, :, offsets[i]:offsets[i+1]] for i,k in enumerate(keys)}
        info = dict(schema="gr00t_rlt_live.v1", phase=phase, episode=episode,
                    contract_sha256=self.c["contract_sha256"], token_sha256=self.token_sha,
                    behavior_sha256=self.behavior, actor_kind=self.actor_kind,
                    token=z[0].cpu().numpy(), state=f["state"], reference=f["reference"], action=action,
                    physical_state=data["state"], active=phase_active(self.c, phase),
                    q01=self.stats["q01"][:18], q99=self.stats["q99"][:18])
        packaged = time.perf_counter()
        if getattr(self, "capture_root", None) is not None:
            from .bootstrap import save_observation
            if writer is not None:
                info.update(writer.submit(f, {**self.c, "phase": phase}, rollout))
                info["capture_writer"] = writer.status()
            else:
                # Legacy token bootstrap retains its synchronous evidence
                # contract. The direct editor does not need disk for actions.
                info.update(save_observation(self.capture_root, f, {**self.c, "phase": phase}, rollout))
        ended = time.perf_counter()
        info["focus_diagnostics"] = getattr(self.features, "last_focus_diagnostics", None)
        info["timing_ms"] = dict(getattr(self.features, "last_timings", {}),
            request_setup_ms=(prepared-started)*1000,
            feature_total_ms=(extracted-prepared)*1000,
            actor_ms=(acted-extracted)*1000,
            package_ms=(packaged-acted)*1000,
            capture_enqueue_or_save_ms=(ended-packaged)*1000,
            server_total_ms=(ended-started)*1000)
        print("[rlt timing] " + json.dumps(dict(phase=phase, timing_ms=info["timing_ms"],
              focus_diagnostics=info["focus_diagnostics"],
              capture_writer=info.get("capture_writer")), allow_nan=False), flush=True)
        return {**arrays, "info": {"rlt": info}}
