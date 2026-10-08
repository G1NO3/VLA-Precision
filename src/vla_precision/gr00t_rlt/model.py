"""RLinf token readout and dimension-aware chunk actor/critic for GR00T."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn


def token_model(c):
    path = Path(c["rlinf_root"]) / "rlinf/models/embodiment/modules/rlt_token_transformer.py"
    spec = importlib.util.spec_from_file_location("gr00t_upstream_rlt_token", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = {k: v for k, v in c["token"].items() if k not in ("lr", "batch_size", "updates")}
    return module.RLTTokenTransformer(**options)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.LayerNorm(256), nn.SiLU(),
                         nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, outputs))


class Heads(nn.Module):
    """Direct chunk actor, not an additive Cartesian residual.

    The critic also sees frozen joint seeds because they affect IK/posture.
    Channel-level human masks allow one arm to be corrected without labelling
    the untouched arm as human. No hardware limits are inferred from q99 stats.
    """
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c["horizon"], c["actor_action_dim"]
        self.full = c["model_action_dim"]
        self.register_buffer("active", torch.tensor(c.get("action_active", [True] * self.d), dtype=torch.bool))
        state_dim = c["token"]["embed_dim"] + c["state_dim"]
        self.actor = mlp(state_dim + self.h * self.full, self.h * self.d)
        qdim = state_dim + self.h * (self.full - self.d) + 2 * self.h * self.d
        self.critics = nn.ModuleList([mlp(qdim, 1), mlp(qdim, 1)])

    def act(self, z, state, reference, *, dropout=0.0):
        ref = reference
        if dropout:
            ref = ref * (torch.rand((len(ref), 1, 1), device=ref.device) >= dropout)
        result = self.actor(torch.cat([z, state, ref.flatten(1)], -1)).reshape(-1, self.h, self.d).clamp(-2.2, 2.2)
        return torch.where(self.active, result, reference[..., :self.d])

    def q(self, z, state, reference, action, valid):
        x = torch.cat([z, state, reference[..., self.d:].flatten(1),
                       (action * valid).flatten(1), valid.float().flatten(1)], -1)
        return torch.cat([net(x) for net in self.critics], -1)


def supervised_losses(prediction, action, reference, human, eligible, valid):
    """Only explicitly eligible channels get supervision; TD eligibility is separate."""
    hmask = human & eligible & valid
    rmask = ~human & eligible & valid
    def masked(error, mask):
        return (error.square() * mask).sum() / mask.sum().clamp(min=1)
    human_loss = masked(prediction - action, hmask)
    reference_loss = masked(prediction - reference[..., :prediction.shape[-1]], rmask)
    return human_loss, reference_loss, hmask.sum(), rmask.sum()


def supervised_loss(prediction, action, reference, human, eligible, valid, options):
    human_loss, reference_loss, _, _ = supervised_losses(prediction, action, reference, human, eligible, valid)
    return options["human_bc_weight"] * human_loss + options["reference_weight"] * reference_loss


class Learner:
    """Offline TD3-style RLT heads; checkpoints include targets and optimizers."""
    def __init__(self, c, device="cpu"):
        self.c, self.options = c, c["learner"]
        self.model = Heads(c).to(device)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.ao = torch.optim.Adam(self.model.actor.parameters(), lr=self.options["actor_lr"])
        self.qo = torch.optim.Adam(self.model.critics.parameters(), lr=self.options["critic_lr"])
        self.step = 0
        self.critic_updates = 0

    def update(self, b):
        o = self.options
        self.step += 1
        z, state, ref = b["token"], b["state"], b["reference"]
        with torch.no_grad():
            nxt = self.target.act(b["next_token"], b["next_state"], b["next_reference"])
            noise = (torch.randn_like(nxt) * o["target_noise"]).clamp(-o["target_noise_clip"], o["target_noise_clip"])
            nv = torch.ones_like(b["valid"]) & self.model.active
            smoothed = torch.where(self.model.active, (nxt + noise).clamp(-2.2, 2.2), nxt)
            nq = self.target.q(b["next_token"], b["next_state"], b["next_reference"], smoothed, nv)
            target = b["reward"] + b["discount"] * nq.min(-1).values
        q = self.model.q(z, state, ref, b["action"], b["valid"])
        td = b.get("td_eligible", torch.ones_like(b["reward"], dtype=torch.bool))
        errors = (q - target[:, None]).square()
        qloss = (errors * td[:, None]).sum() / (2 * td.sum()).clamp(min=1)
        if not torch.isfinite(qloss):
            raise FloatingPointError("Non-finite critic loss")
        self.qo.zero_grad()
        qloss.backward()
        nn.utils.clip_grad_norm_(self.model.critics.parameters(), o["max_grad_norm"], error_if_nonfinite=True)
        if td.any():
            self.qo.step()
            self.critic_updates += 1
        metrics = {"step": self.step, "critic_loss": float(qloss.detach()),
                   "q1_loss": float((errors[:, 0]*td).sum().detach()/td.sum().clamp(min=1)),
                   "q2_loss": float((errors[:, 1]*td).sum().detach()/td.sum().clamp(min=1)),
                   "td_samples": int(td.sum()),
                   "critic_updates": self.critic_updates,
                   "target_q": float(target.mean()), "actor_updated": False}
        if self.step % o["actor_update_interval"] == 0:
            self.model.critics.requires_grad_(False)
            try:
                pred = self.model.act(z, state, ref, dropout=o["reference_dropout"])
                human_loss, ref_loss, nh, nr = supervised_losses(
                    pred, b["action"], ref, b["human"], b["bc_eligible"], b["valid"])
                bc = o["human_bc_weight"] * human_loss + o["reference_weight"] * ref_loss
                qvalues = self.model.q(z, state, ref, pred, b["valid"])[:, 0]
                qpi = (qvalues * td).sum() / td.sum().clamp(min=1)
                weight = o["q_weight"] if self.critic_updates > o["critic_warmup_steps"] else 0.0
                loss = bc - weight * qpi
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite actor loss")
                self.ao.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), o["max_grad_norm"], error_if_nonfinite=True)
                self.ao.step()
                metrics.update(actor_updated=True, actor_loss=float(loss.detach()),
                    bc_loss=float(human_loss.detach()), human_bc_loss=float(human_loss.detach()),
                    ref_loss=float(ref_loss.detach()), reference_loss=float(ref_loss.detach()),
                    rl_loss=float(-qpi.detach()), supervised_loss=float(bc.detach()),
                    weighted_bc_loss=float(o["human_bc_weight"] * human_loss.detach()),
                    weighted_ref_loss=float(o["reference_weight"] * ref_loss.detach()),
                    weighted_rl_loss=float(-weight * qpi.detach()), q_pi=float(qpi.detach()),
                    q_weight=weight, human_bc_elements=int(nh), reference_elements=int(nr))
            finally:
                self.model.critics.requires_grad_(True)
            with torch.no_grad():
                for dst, src in zip(self.target.parameters(), self.model.parameters()):
                    dst.lerp_(src, o["target_tau"])
        return metrics

    def state_dict(self):
        return dict(model=self.model.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.ao.state_dict(), critic_optimizer=self.qo.state_dict(),
                    step=self.step, critic_updates=self.critic_updates)

    def load_state_dict(self, value):
        self.model.load_state_dict(value["model"])
        self.target.load_state_dict(value["target"])
        self.ao.load_state_dict(value["actor_optimizer"])
        self.qo.load_state_dict(value["critic_optimizer"])
        self.step = value["step"]
        self.critic_updates = value.get("critic_updates", self.step)
