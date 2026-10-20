"""RLinf token readout and dimension-aware chunk actor/critic for GR00T."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn
from .core import is_editor, is_async_editor, is_shared, phase_active


def token_model(c):
    if is_editor(c):
        raise ValueError("Direct editor uses fixed spatial pooling, not a learned token/autoencoder")
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

    def act(self, z, state, reference, *, dropout=0.0, phase=None):
        ref = reference
        if dropout:
            ref = ref * (torch.rand((len(ref), 1, 1), device=ref.device) >= dropout)
        result = self.actor(torch.cat([z, state, ref.flatten(1)], -1)).reshape(-1, self.h, self.d).clamp(-2.2, 2.2)
        return torch.where(self.active, result, reference[..., :self.d])

    def q(self, z, state, reference, action, valid, context=None, *, phase=None):
        x = torch.cat([z, state, reference[..., self.d:].flatten(1),
                       (action * valid).flatten(1), valid.float().flatten(1)], -1)
        return torch.cat([net(x) for net in self.critics], -1)

    def active_for(self, phase=None):
        return self.active

    def smooth_action(self, action, reference, noise, *, phase=None):
        return torch.where(self.active, (action + noise).clamp(-2.2, 2.2), action)


def spatial_features(prefix, mask):
    """Pinned PI05 prefix: ego256, focus256, masked dummy256, language.

    Each 224px SigLIP image is a row-major 16x16 grid. Pool each quadrant
    separately; do not average cameras or include padding/language as pixels.
    Prefix features are frozen, though contextualized by the language tokens.
    """
    if (prefix.ndim != 3 or prefix.shape[-1] != 2048 or prefix.shape[1] < 768
            or mask.shape != prefix.shape[:2] or mask.dtype != torch.bool):
        raise ValueError("Unexpected PI05 image-prefix layout")
    if not mask[:, :512].all() or mask[:, 512:768].any():
        raise ValueError("Editor requires two real views and a masked third view")
    grid = prefix[:, :512].detach().float().reshape(-1, 2, 2, 8, 2, 8, 2048)
    result = grid.mean(dim=(3, 5)).flatten(1)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite editor visual features")
    return result


class ContextCritic(nn.Module):
    def __init__(self, inputs, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.context_encoder = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 64), nn.SiLU())
        self.net = mlp(inputs + 64, 1)

    def forward(self, x, context):
        if context is None or context.shape != (len(x), self.context_dim):
            raise ValueError("Critic requires observation-time ACT scheduler context")
        return self.net(torch.cat([x, self.context_encoder(context)], -1))


class EditorHeads(Heads):
    """Zero-initialized, bounded action editor of the frozen reference chunk."""
    def __init__(self, c):
        super().__init__(c)
        self.shared = is_shared(c)
        self.phase_dim = 5 if self.shared else 0
        if self.shared:
            self.actor = mlp(c["token"]["embed_dim"] + c["state_dim"] + self.h*self.full + 5, self.h*self.d)
            self.register_buffer("phase_masks", torch.tensor([phase_active(c, f"p{i+1}") for i in range(5)], dtype=torch.bool))
        self.scale = float(c["editor"]["residual_scale"])
        # "tanh" (recipe): edit bounded by +/-scale. "linear" (learner option,
        # operator 2026-10-19): scale is only the output gain, the edit is
        # unbounded and the edited action is clamped to the action range.
        # Not part of the state dict: serving takes it from the checkpoint.
        self.activation = c.get("learner", {}).get("editor_activation", "tanh")
        if self.activation not in EDITOR_ACTIVATIONS:
            raise ValueError(f"editor_activation must be one of {EDITOR_ACTIVATIONS}")
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        qdim = c["token"]["embed_dim"] + c["state_dim"] + self.h * (self.full + 2*self.d) + self.phase_dim
        self.context_dim = c["scheduler"]["context_dim"] if is_async_editor(c) else 0
        self.critics = nn.ModuleList([ContextCritic(qdim, self.context_dim) if self.context_dim else mlp(qdim, 1)
                                     for _ in range(2)])

    def _phase_features(self, phase, z):
        if not self.shared:
            return z
        self.active_for(phase)  # validate before indexing/one-hot
        if phase.shape != (len(z),):
            raise ValueError("Phase batch differs from observations")
        return torch.cat([z, torch.nn.functional.one_hot(phase.long(), 5).to(z)], -1)

    def active_for(self, phase=None):
        if not self.shared:
            return self.active
        if (phase is None or phase.ndim != 1 or not torch.isfinite(phase).all()
                or not ((phase >= 0) & (phase < 5) & (phase == phase.long())).all()):
            raise ValueError("Shared editor requires integer phase labels 0..4")
        return self.phase_masks[phase.long()][:, None, :]

    def act(self, z, state, reference, *, dropout=0.0, phase=None):
        if dropout:
            raise ValueError("Editor cannot drop its reference action")
        out = self.actor(torch.cat([self._phase_features(phase, z), state, reference.flatten(1)], -1))
        delta = (self.scale * (out if self.activation == "linear" else out.tanh())).reshape(-1, self.h, self.d)
        return self._edited(reference[..., :self.d], delta, phase)

    def _edited(self, ref, delta, phase):
        action = ref + torch.where(self.active_for(phase), delta, torch.zeros_like(delta))
        return action.clamp(-ACTION_RANGE, ACTION_RANGE) if self.activation == "linear" else action

    def q(self, z, state, reference, action, valid, context=None, *, phase=None):
        x = torch.cat([self._phase_features(phase, z), state, reference.flatten(1), (action*valid).flatten(1), valid.float().flatten(1)], -1)
        return torch.cat([net(x, context) if self.context_dim else net(x) for net in self.critics], -1)

    def smooth_action(self, action, reference, noise, *, phase=None):
        ref = reference[..., :self.d]
        delta = action + noise - ref
        if self.activation == "tanh":
            delta = delta.clamp(-self.scale, self.scale)
        return self._edited(ref, delta, phase)


EDITOR_ACTIVATIONS = ("tanh", "linear")
ACTION_RANGE = 2.2   # normalized action clamp (Heads.smooth_action)


def make_heads(c):
    return EditorHeads(c) if is_editor(c) else Heads(c)


def supervised_losses(prediction, action, reference, human, eligible, valid):
    """Only explicitly eligible channels get supervision; TD eligibility is separate."""
    hmask = human & eligible & valid
    rmask = ~human & eligible & valid
    def masked(error, mask):
        return (error.square() * mask).sum() / mask.sum().clamp(min=1)
    human_loss = masked(prediction - action, hmask)
    reference_loss = masked(prediction - reference[..., :prediction.shape[-1]], rmask)
    return human_loss, reference_loss, hmask.sum(), rmask.sum()


def term_gradients(params, terms):
    """Per-term gradients of the actor objective, {name: [grad per param]}.
    Their sum is exactly the gradient of sum(terms) (what backward() gave)."""
    params = list(params)
    names = list(terms)
    out = {}
    for i, name in enumerate(names):
        grads = torch.autograd.grad(terms[name], params, retain_graph=i < len(names) - 1, allow_unused=True)
        out[name] = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
    return out


def gradient_stats(grads):
    """Norm of each term's gradient and the cosine between each pair (0 when a
    term has no gradient), for the actor's BC / reference / Q tug of war."""
    flat = {k: torch.cat([g.reshape(-1) for g in v]) for k, v in grads.items()}
    norms = {k: float(v.norm()) for k, v in flat.items()}
    stats = {f"grad_norm_{k}": n for k, n in norms.items()}
    names = list(flat)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            den = norms[a] * norms[b]
            stats[f"grad_cos_{a}_{b}"] = float(flat[a] @ flat[b]) / den if den > 0 else 0.0
    return stats


def supervised_loss(prediction, action, reference, human, eligible, valid, options):
    human_loss, reference_loss, _, _ = supervised_losses(prediction, action, reference, human, eligible, valid)
    return options["human_bc_weight"] * human_loss + options["reference_weight"] * reference_loss


class Learner:
    """Offline TD3-style RLT heads; checkpoints include targets and optimizers."""
    def __init__(self, c, device="cpu"):
        self.c, self.options = c, c["learner"]
        self.model = make_heads(c).to(device)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.ao = torch.optim.Adam(self.model.actor.parameters(), lr=self.options["actor_lr"])
        self.qo = torch.optim.Adam(self.model.critics.parameters(), lr=self.options["critic_lr"])
        self.step = 0
        self.critic_updates = 0
        self.anchor = None

    def set_anchor(self, model_state):
        """Reference-loss target = a frozen copy of another editor (e.g. the best
        served round) instead of the frozen pi0.5 reference: a trust region
        around that editor (operator 2026-10-20). None = the plain reference."""
        self.anchor = copy.deepcopy(self.model).requires_grad_(False)
        self.anchor.load_state_dict(model_state)
        self.anchor.activation = self.model.activation

    def update(self, b):
        o = self.options
        self.step += 1
        z, state, ref = b["token"], b["state"], b["reference"]
        with torch.no_grad():
            nxt = self.target.act(b["next_token"], b["next_state"], b["next_reference"], phase=b.get("next_phase"))
            noise = (torch.randn_like(nxt) * o["target_noise"]).clamp(-o["target_noise_clip"], o["target_noise_clip"])
            nv = torch.ones_like(b["valid"]) & self.model.active_for(b.get("next_phase"))
            smoothed = self.target.smooth_action(nxt, b["next_reference"], noise, phase=b.get("next_phase"))
            nq = self.target.q(b["next_token"], b["next_state"], b["next_reference"], smoothed, nv, b.get("next_scheduler_context"), phase=b.get("next_phase"))
            target = b["reward"] + b["discount"] * nq.min(-1).values
        q = self.model.q(z, state, ref, b["action"], b["valid"], b.get("scheduler_context"), phase=b.get("phase"))
        td = b.get("td_eligible", torch.ones_like(b["reward"], dtype=torch.bool))
        errors = (q - target[:, None]).square()
        qloss = (errors * td[:, None]).sum() / (2 * td.sum()).clamp(min=1)
        if not torch.isfinite(qloss):
            raise FloatingPointError("Non-finite critic loss")
        self.qo.zero_grad()
        qloss.backward()
        critic_norm = float(nn.utils.clip_grad_norm_(self.model.critics.parameters(), o["max_grad_norm"], error_if_nonfinite=True))
        if td.any():
            self.qo.step()
            self.critic_updates += 1
        metrics = {"step": self.step, "critic_loss": float(qloss.detach()),
                   "q1_loss": float((errors[:, 0]*td).sum().detach()/td.sum().clamp(min=1)),
                   "q2_loss": float((errors[:, 1]*td).sum().detach()/td.sum().clamp(min=1)),
                   "td_samples": int(td.sum()),
                   "critic_updates": self.critic_updates,
                   "target_q": float(target.mean()), "actor_updated": False,
                   "critic_grad_norm": critic_norm, "critic_grad_clipped": critic_norm > o["max_grad_norm"]}
        if self.step % o["actor_update_interval"] == 0:
            self.model.critics.requires_grad_(False)
            try:
                pred = self.model.act(z, state, ref, dropout=o["reference_dropout"], phase=b.get("phase"))
                anchor = ref
                if self.anchor is not None:
                    with torch.no_grad():
                        anchor = self.anchor.act(z, state, ref, phase=b.get("phase"))
                human_loss, ref_loss, nh, nr = supervised_losses(
                    pred, b["action"], anchor, b["human"], b["bc_eligible"], b["valid"])
                bc = o["human_bc_weight"] * human_loss + o["reference_weight"] * ref_loss
                qvalues = self.model.q(z, state, ref, pred, b["valid"], b.get("scheduler_context"), phase=b.get("phase"))[:, 0]
                qpi = (qvalues * td).sum() / td.sum().clamp(min=1)
                weight = o["q_weight"] if self.critic_updates > o["critic_warmup_steps"] else 0.0
                loss = bc - weight * qpi
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite actor loss")
                self.ao.zero_grad()
                # One gradient per term (summed = loss.backward()), so the log
                # shows which of BC / reference / Q drives the actor.
                params = list(self.model.actor.parameters())
                grads = term_gradients(params, dict(bc=o["human_bc_weight"] * human_loss,
                                                    ref=o["reference_weight"] * ref_loss, rl=-weight * qpi))
                for i, p in enumerate(params):
                    p.grad = sum(g[i] for g in grads.values())
                actor_norm = float(nn.utils.clip_grad_norm_(params, o["max_grad_norm"], error_if_nonfinite=True))
                self.ao.step()
                metrics.update(gradient_stats(grads), actor_grad_norm=actor_norm,
                               actor_grad_clipped=actor_norm > o["max_grad_norm"])
                metrics.update(actor_updated=True, actor_loss=float(loss.detach()),
                    bc_loss=float(human_loss.detach()), human_bc_loss=float(human_loss.detach()),
                    ref_loss=float(ref_loss.detach()), reference_loss=float(ref_loss.detach()),
                    rl_loss=float(-qpi.detach()), supervised_loss=float(bc.detach()),
                    weighted_bc_loss=float(o["human_bc_weight"] * human_loss.detach()),
                    weighted_ref_loss=float(o["reference_weight"] * ref_loss.detach()),
                    weighted_rl_loss=float(-weight * qpi.detach()), q_pi=float(qpi.detach()),
                    q_weight=weight, human_bc_elements=int(nh), reference_elements=int(nr))
                if is_editor(self.c):
                    delta = (pred - ref[..., :self.model.d]).detach()[self.model.active_for(b.get("phase")).expand_as(pred)]
                    human_mask = b["human"] & b["bc_eligible"] & b["valid"]
                    outside = (b["action"] - ref[..., :self.model.d]).abs() > self.model.scale
                    metrics.update(editor_delta_mean_abs=float(delta.abs().mean()),
                                   editor_delta_max_abs=float(delta.abs().max()),
                                   editor_saturation_fraction=float((delta.abs() >= .99*self.model.scale).float().mean()),
                                   editor_human_outside_fraction=float((outside & human_mask).sum()/human_mask.sum().clamp(min=1)))
                    if self.anchor is not None:
                        metrics["anchor_gap_mean_abs"] = float((pred - anchor).detach().abs().mean())
            finally:
                self.model.critics.requires_grad_(True)
            with torch.no_grad():
                for dst, src in zip(self.target.parameters(), self.model.parameters()):
                    dst.lerp_(src, o["target_tau"])
        return metrics

    def state_dict(self):
        return dict(model=self.model.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.ao.state_dict(), critic_optimizer=self.qo.state_dict(),
                    step=self.step, critic_updates=self.critic_updates,
                    **({} if self.anchor is None else dict(anchor=self.anchor.state_dict())))

    def load_state_dict(self, value):
        self.model.load_state_dict(value["model"])
        self.target.load_state_dict(value["target"])
        self.ao.load_state_dict(value["actor_optimizer"])
        self.qo.load_state_dict(value["critic_optimizer"])
        self.step = value["step"]
        self.critic_updates = value.get("critic_updates", self.step)
        self.anchor = None
        if "anchor" in value:
            self.set_anchor(value["anchor"])
