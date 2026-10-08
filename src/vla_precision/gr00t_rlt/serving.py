"""Single-owner RLToken inference service. No robot or collection I/O."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .core import check_code, sha256, validate_assets
from .features import FrozenGR00TFeatures
from .model import Heads, token_model


class LivePolicy:
    def __init__(self, config, device="cuda"):
        self.c, self.device = config, device
        check_code(config)
        validate_assets(config, verify_weights=True)
        root = Path(config["output"])
        self.token_path = root / "token/best.pt"
        self.token_sha = sha256(self.token_path)
        saved = torch.load(self.token_path, map_location=device, weights_only=True)
        if saved["contract_sha256"] != config["contract_sha256"]:
            raise ValueError("Token recipe mismatch")
        self.token = token_model(config).to(device).eval().requires_grad_(False)
        self.token.load_state_dict(saved["state"])
        self.heads = Heads(config).to(device).eval().requires_grad_(False)
        self.episode = None
        self._load_actor()
        self.features = FrozenGR00TFeatures(config, device)
        self.stats = json.loads((Path(config["run_root"])/"dataset_statistics.json").read_text())[config["unnorm_key"]]["action"]
        sys.path.insert(0, str(Path(config["starvla_root"]).parent / "VLAPolicyBridge"))

    def _load_actor(self):
        root = Path(self.c["output"])
        latest = root / "learner/latest.json"
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

    @torch.inference_mode()
    def get_action(self, observation, options=None):
        from vla_policy_bridge.pi05_fulltask import observation_to_inputs, INSTRUCTIONS
        data = observation_to_inputs(observation)
        phase = "p" + str(INSTRUCTIONS.index(data["prompt"]) + 1)
        if phase != self.c["phase"]:
            raise ValueError(f"Server prepared for {self.c['phase']}, not {phase}; pause and select that phase's server")
        episode = (options or {}).get("episode")
        if not isinstance(episode, str) or not episode:
            raise ValueError("RL server requires an explicit episode id; refuse ordinary IL clients")
        if episode != self.episode:
            self._load_actor()
            self.episode = episode
        f = self.features.extract({"observation.state": data["state"], "task": data["prompt"],
            "observation.images.rgb": data["base_0_rgb"],
            "observation.images.wrist_left": data["left_wrist_0_rgb"]})
        z = self.token.encode_flat(torch.as_tensor(f["prefix"], device=self.device).float()[None],
                                  torch.as_tensor(f["mask"], device=self.device)[None])
        action = self.heads.act(z, torch.as_tensor(f["state"], device=self.device)[None],
                               torch.as_tensor(f["reference"], device=self.device)[None])[0].cpu().numpy()
        # Heads clamps active channels; inactive channels retain frozen reference
        # exactly, including constant-channel values outside the learned envelope.
        action = action.astype(np.float32)
        physical = self.features.physical_actions(action, f["reference"])
        keys = ("left_wrist_rel6", "right_wrist_rel6", "left_hand_rel", "right_hand_abs", "left_arm_abs", "right_arm_abs")
        offsets = (0, 6, 12, 13, 18, 25, 32)
        arrays = {k: physical[None, :, offsets[i]:offsets[i+1]] for i,k in enumerate(keys)}
        info = dict(schema="gr00t_rlt_live.v1", phase=phase, episode=episode,
                    contract_sha256=self.c["contract_sha256"], token_sha256=self.token_sha,
                    behavior_sha256=self.behavior, actor_kind=self.actor_kind,
                    token=z[0].cpu().numpy(), state=f["state"], reference=f["reference"], action=action,
                    physical_state=data["state"], active=self.c["action_active"],
                    q01=self.stats["q01"][:18], q99=self.stats["q99"][:18])
        return {**arrays, "info": {"rlt": info}}
