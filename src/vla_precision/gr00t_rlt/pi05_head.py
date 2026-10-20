"""Full PI05 action-head RL: BPTT through the deterministic flow sampler.

The served reference is a deterministic function of the observation: every
predict_action reseeds (``reference_seed``) before drawing the initial noise
and runs ``num_inference_steps`` Euler steps of the action expert over the
frozen VLM prefix. So a(o) = Phi_theta(KV(o), eps) is differentiable in the
action-head weights theta, and the editor's three losses (human BC,
reference, -Q) backpropagate through all denoising steps (operator 2026-10-20).

The VLM (PaliGemma) stays frozen. Its per-layer prefix K/V is computed once
per observation and cached on disk, valid tokens only, with the pad mask.
Loading scatters them back to their original positions (zeros where masked):
masked tokens get -inf attention, so the sampler sees exactly the served
tensor shapes and reproduces the served bf16 reference bit for bit. (Packing
the valid tokens instead is mathematically exact too, but the shorter bf16
attention reductions moved the reference by up to 0.024.)

Losses on the head's full 30x32 output ``pred``:
  human BC   rows the operator drove (human & eligible & valid)  -> human label
  reference  other eligible active rows                           -> frozen reference
  hold       held channels (~valid) and the 14 IK joint seeds     -> frozen reference
  Q          -q_weight * Q1(pred with held channels = reference)
The editor's twin critic (model.EditorHeads, same inputs) is reused; its TD
target takes the target head's action at the next observation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import EditorHeads, gradient_stats, term_gradients, ACTION_RANGE

TRAINABLE_EXCLUDE = ("gemma_expert.lm_head",)   # unused vocab projection of the expert


def trainable(head):
    return [p for n, p in head.named_parameters() if not n.startswith(TRAINABLE_EXCLUDE)]


def head_state(head):
    return {k: v for k, v in head.state_dict().items() if not k.startswith(TRAINABLE_EXCLUDE)}


# --- prefix K/V cache -------------------------------------------------------

def compact_prefix(past_key_values, pad_mask):
    """(kv, mask): kv (2, depth, kv_heads, L_valid, head_dim) of one observation's
    prefix cache, mask (L,) its pad mask (where the valid tokens sit)."""
    if pad_mask.ndim != 2 or pad_mask.shape[0] != 1:
        raise ValueError("Expected one observation's [1, L] prefix pad mask")
    keep = pad_mask[0].bool()
    layers = [past_key_values[i] for i in range(len(past_key_values))]
    kv = torch.stack([torch.stack([kv[j][0][:, keep] for kv in layers]) for j in (0, 1)])
    return kv, keep.cpu()


def save_prefix(path, prefix):
    """bf16 bits as uint16 plus the pad mask, uncompressed .npz; atomic rename."""
    kv, mask = prefix
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, kv=kv.detach().to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16),
             mask=mask.cpu().numpy().astype(bool))
    tmp.replace(path)


def load_prefix(path):
    with np.load(path) as f:
        return torch.from_numpy(f["kv"].view(np.int16)).view(torch.bfloat16), torch.from_numpy(f["mask"])


def collate_prefix(prefixes):
    """One batch at the served layout: kv (2, depth, B, kv_heads, L, head_dim) with
    each observation's valid tokens at their original positions (zeros where
    masked or past its end), valid (B, L) = the pad masks."""
    longest = max(len(mask) for _, mask in prefixes)
    two, depth, heads, _, dim = prefixes[0][0].shape
    device = prefixes[0][0].device
    out = torch.zeros(two, depth, len(prefixes), heads, longest, dim, dtype=prefixes[0][0].dtype, device=device)
    valid = torch.zeros(len(prefixes), longest, dtype=torch.bool, device=device)
    for b, (kv, mask) in enumerate(prefixes):
        if int(mask.sum()) != kv.shape[3]:
            raise ValueError("Prefix mask disagrees with its cached tokens")
        valid[b, :len(mask)] = mask.to(device)
        out[:, :, b, :, valid[b]] = kv
    return out, valid


def prefix_cache(kv):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer in range(kv.shape[1]):
        cache.update(kv[0, layer], kv[1, layer], layer)
    return cache


# --- the deterministic sampler ----------------------------------------------

def denoise(head, kv, valid, noise, steps, *, checkpoint=False, jacobian_steps=None):
    """PI0.sample_actions over a cached prefix, differentiable in ``head``.

    kv (2, depth, B, kv_heads, L, head_dim), valid (B, L) bool, noise (B, H, D).
    Same Euler schedule, masks and positions as PI0.denoise_step.
    ``jacobian_steps`` = K (operator 2026-10-20): every step's velocity still
    backpropagates into the weights, but dx_{k+1}/dx_k = I + dt dv/dx is kept
    only for the last K steps; earlier steps see x_k detached, i.e. their
    state Jacobian is truncated to I. Same forward values. None = exact BPTT."""
    from starVLA.model.modules.action_model.OpenPI_ActionHead import make_att_2d_masks, make_att_4d_masks
    device = noise.device
    cache = prefix_cache(kv.to(device))
    valid = valid.to(device)
    bsize, prefix_len = valid.shape
    bf16 = head.action_in_proj.weight.dtype == torch.bfloat16

    def step(x_t, timestep):
        embs, pad, att, cond = head.embed_action_suffix(None, x_t, timestep)
        suffix_len = pad.shape[1]
        prefix_2d = valid[:, None, :].expand(bsize, suffix_len, prefix_len)
        mask = make_att_4d_masks(torch.cat([prefix_2d, make_att_2d_masks(pad, att)], dim=2))
        position_ids = valid.sum(-1)[:, None] + torch.cumsum(pad, dim=1) - 1
        return head(attention_mask=mask, position_ids=position_ids, past_key_values=cache,
                    suffix_embs=embs.to(torch.bfloat16) if bf16 else embs, adarms_cond=cond)

    x_t = noise.to(torch.float32)
    dt = torch.tensor(-1.0 / max(steps, 1), dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    exact_from = 0 if jacobian_steps is None else max(steps - int(jacobian_steps), 0)
    done = 0
    while time >= -dt / 2:
        t = time.expand(bsize)
        x_in = x_t if done >= exact_from else x_t.detach()
        v_t = (torch.utils.checkpoint.checkpoint(step, x_in, t, use_reentrant=False)
               if checkpoint and torch.is_grad_enabled() else step(x_in, t))
        x_t = x_t + dt * v_t
        time = time + dt
        done += 1
    return x_t


def reference_noise(seed, shape, device):
    """The served initial noise: first draw after torch.manual_seed(seed)."""
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)


# --- learner ----------------------------------------------------------------

def hold_mask(valid, width):
    """Elements anchored to the reference whatever the row: held channels of the
    18 actor dims and every remaining model dim (IK joint seeds)."""
    held = ~valid
    rest = torch.ones(*valid.shape[:-1], width - valid.shape[-1], dtype=torch.bool, device=valid.device)
    return torch.cat([held, rest], -1)


class HeadLearner:
    """TD3-style: twin editor critic, the PI05 action head as the actor."""
    def __init__(self, c, head, noise, *, steps, device="cpu", amp=False, checkpoint=False, jacobian_steps=None,
                 micro_batch=None):
        self.c, self.options = c, c["learner"]
        o = self.options
        self.device, self.steps, self.checkpoint, self.jacobian_steps = device, int(steps), checkpoint, jacobian_steps
        self.micro_batch = micro_batch
        self.amp = amp
        head.gemma_expert.lm_head = None     # 257k-vocab projection, never used by the sampler
        self.head = head.to(device=device, dtype=torch.float32).train()
        for name, p in self.head.named_parameters():
            p.requires_grad_(not name.startswith(TRAINABLE_EXCLUDE))
        self.target_head = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.noise = noise.to(device=device, dtype=torch.float32)
        self.critic = EditorHeads(c).to(device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.ao = torch.optim.Adam(trainable(self.head), lr=o["head_lr"])
        self.qo = torch.optim.Adam(self.critic.critics.parameters(), lr=o["critic_lr"])
        self.step = 0
        self.critic_updates = 0
        self.d = self.critic.d

    def load_critic(self, editor_learner_state):
        """Warm-start the twin critic (and its target) from an editor checkpoint."""
        critics = {k: v for k, v in editor_learner_state["model"].items() if k.startswith("critics.")}
        self.critic.load_state_dict(critics, strict=False)
        target = {k: v for k, v in editor_learner_state["target"].items() if k.startswith("critics.")}
        self.target_critic.load_state_dict(target, strict=False)

    def _chunks(self, n):
        size = self.micro_batch or n
        return [slice(i, min(i + size, n)) for i in range(0, n, size)]

    def act(self, head, kv, valid):
        noise = self.noise.expand(len(valid), *self.noise.shape[-2:])
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.amp):
            return denoise(head, kv, valid, noise, self.steps, checkpoint=self.checkpoint,
                           jacobian_steps=self.jacobian_steps).float()

    def _edited(self, pred, reference, phase):
        """Actor dims with held channels kept at the frozen reference (as served)."""
        return torch.where(self.critic.active_for(phase), pred[..., :self.d], reference[..., :self.d])

    def update(self, b, *, term_stats=False):
        o = self.options
        self.step += 1
        z, state, ref = b["token"], b["state"], b["reference"]
        with torch.no_grad():
            nxt = torch.cat([self.act(self.target_head, b["next_kv"][:, :, sl], b["next_kv_valid"][sl])
                             for sl in self._chunks(len(b["reward"]))])
            nxt = self._edited(nxt, b["next_reference"], b["next_phase"])
            noise = (torch.randn_like(nxt) * o["target_noise"]).clamp(-o["target_noise_clip"], o["target_noise_clip"])
            smoothed = self._edited((nxt + noise).clamp(-ACTION_RANGE, ACTION_RANGE), b["next_reference"], b["next_phase"])
            nv = torch.ones_like(b["valid"]) & self.critic.active_for(b["next_phase"])
            nq = self.target_critic.q(b["next_token"], b["next_state"], b["next_reference"], smoothed, nv,
                                      b.get("next_scheduler_context"), phase=b["next_phase"])
            target = b["reward"] + b["discount"] * nq.min(-1).values
        q = self.critic.q(z, state, ref, b["action"], b["valid"], b.get("scheduler_context"), phase=b["phase"])
        td = b.get("td_eligible", torch.ones_like(b["reward"], dtype=torch.bool))
        errors = (q - target[:, None]).square()
        qloss = (errors * td[:, None]).sum() / (2 * td.sum()).clamp(min=1)
        if not torch.isfinite(qloss):
            raise FloatingPointError("Non-finite critic loss")
        self.qo.zero_grad()
        qloss.backward()
        critic_norm = float(nn.utils.clip_grad_norm_(self.critic.critics.parameters(), o["max_grad_norm"], error_if_nonfinite=True))
        if td.any():
            self.qo.step()
            self.critic_updates += 1
        metrics = {"step": self.step, "critic_loss": float(qloss.detach()), "td_samples": int(td.sum()),
                   "critic_updates": self.critic_updates, "target_q": float(target.mean()), "actor_updated": False,
                   "critic_grad_norm": critic_norm, "critic_grad_clipped": critic_norm > o["max_grad_norm"]}
        if self.step % o["actor_update_interval"] == 0:
            self.critic.critics.requires_grad_(False)
            try:
                # Micro-batched backward (gradient accumulation). Each term is a
                # masked mean over the WHOLE batch: the denominators come from the
                # masks up front, so the summed gradient equals the full-batch one.
                hmask = b["human"] & b["bc_eligible"] & b["valid"]
                rmask = ~b["human"] & b["bc_eligible"] & b["valid"]
                hold = hold_mask(b["valid"], self.noise.shape[-1])
                nh, nr, nhold, ntd = (m.sum().clamp(min=1) for m in (hmask, rmask, hold, td))
                weight = o["q_weight"] if self.critic_updates > o["critic_warmup_steps"] else 0.0
                params = trainable(self.head)
                self.ao.zero_grad()
                totals = dict(bc=0., ref=0., hold=0., q=0.)
                grads, edits = None, []
                for n_chunk, sl in enumerate(self._chunks(len(ref))):
                    full = self.act(self.head, b["kv"][:, :, sl], b["kv_valid"][sl])
                    pred = self._edited(full, ref[sl], b["phase"][sl])
                    human_loss = ((pred - b["action"][sl]).square() * hmask[sl]).sum() / nh
                    ref_loss = ((pred - ref[sl, :, :self.d]).square() * rmask[sl]).sum() / nr
                    hold_loss = ((full - ref[sl]).square() * hold[sl]).sum() / nhold
                    qvalues = self.critic.q(z[sl], state[sl], ref[sl], pred, b["valid"][sl],
                                            None if b.get("scheduler_context") is None else b["scheduler_context"][sl],
                                            phase=b["phase"][sl])[:, 0]
                    qpi = (qvalues * td[sl]).sum() / ntd
                    terms = dict(bc=o["human_bc_weight"] * human_loss,
                                 ref=o["reference_weight"] * (ref_loss + hold_loss), rl=-weight * qpi)
                    if not torch.isfinite(sum(terms.values())):
                        raise FloatingPointError("Non-finite actor loss")
                    if term_stats and n_chunk == 0:
                        # Per-term gradients (one backward per term) on the first micro-batch
                        # only: a logged estimate; their sum is this chunk's exact contribution.
                        grads = term_gradients(params, terms)
                        for i, p in enumerate(params):
                            g = sum(t[i] for t in grads.values())
                            p.grad = g if p.grad is None else p.grad + g
                    else:
                        sum(terms.values()).backward()
                    for k, v in (("bc", human_loss), ("ref", ref_loss), ("hold", hold_loss), ("q", qpi)):
                        totals[k] += float(v.detach())
                    edits.append((pred - ref[sl, :, :self.d]).detach()[self.critic.active_for(b["phase"][sl]).expand_as(pred)])
                if grads is not None:
                    metrics.update(gradient_stats(grads), term_stats_windows=int(self._chunks(len(ref))[0].stop))
                    del grads
                actor_norm = float(nn.utils.clip_grad_norm_(params, o["max_grad_norm"], error_if_nonfinite=True))
                self.ao.step()
                edit = torch.cat(edits)
                loss = (o["human_bc_weight"] * totals["bc"] + o["reference_weight"] * (totals["ref"] + totals["hold"])
                        - weight * totals["q"])
                metrics.update(actor_updated=True, actor_loss=loss, actor_grad_norm=actor_norm,
                               actor_grad_clipped=actor_norm > o["max_grad_norm"],
                               human_bc_loss=totals["bc"], ref_loss=totals["ref"], hold_loss=totals["hold"],
                               rl_loss=-totals["q"], q_pi=totals["q"], q_weight=weight,
                               human_bc_elements=int(hmask.sum()), reference_elements=int(rmask.sum()),
                               head_edit_mean_abs=float(edit.abs().mean()), head_edit_max_abs=float(edit.abs().max()))
            finally:
                self.critic.critics.requires_grad_(True)
            with torch.no_grad():
                for dst, src in zip(trainable(self.target_head), trainable(self.head)):
                    dst.lerp_(src, o["target_tau"])
                for dst, src in zip(self.target_critic.parameters(), self.critic.parameters()):
                    dst.lerp_(src, o["target_tau"])
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        return metrics

    def state_dict(self):
        return dict(head=head_state(self.head), target_head=head_state(self.target_head),
                    critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
                    head_optimizer=self.ao.state_dict(), critic_optimizer=self.qo.state_dict(),
                    step=self.step, critic_updates=self.critic_updates)

    def load_state_dict(self, value):
        self.head.load_state_dict(value["head"], strict=False)
        self.target_head.load_state_dict(value["target_head"], strict=False)
        self.critic.load_state_dict(value["critic"])
        self.target_critic.load_state_dict(value["target_critic"])
        self.ao.load_state_dict(value["head_optimizer"])
        self.qo.load_state_dict(value["critic_optimizer"])
        self.step, self.critic_updates = value["step"], value["critic_updates"]
