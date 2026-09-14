"""Native ACoB action-expert LoRA initialized from the pinned relative BC."""

import dataclasses, hashlib, json, os
from pathlib import Path
import numpy as np
from .config import IMAGE_MAP, PROMPT, WORKSPACE


def create_agent(config, *, discount=None):
    os.environ.setdefault("OPENPI_DATA_HOME", str(Path(config["baseline"]).parents[1] / "tokenizer"))
    from openpi.training import config as oc, weight_loaders, optimizer
    from vla_precision.integrations.openpi.configs import get_config
    from vla_precision.acob.agent import _create_acob_agent

    base = get_config("pi05_acob_pipette")
    norm = Path(config["norm_stats"])
    # Point the native data transform at the checkpoint's exact asset tree.
    asset_id = str(norm.parent.relative_to(Path(config["baseline"]) / "assets"))
    data = dataclasses.replace(
        base.data,
        repo_id=asset_id,
        assets=oc.AssetsConfig(assets_dir=str(Path(config["baseline"]) / "assets"), asset_id=asset_id),
    )
    train = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, action_horizon=10, action_dim=32, max_token_len=200),
        exp_name=config["name"],
        data=data,
        batch_size=config["batch_size"],
        fsdp_devices=1,
        checkpoint_base_dir=config["output"],
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(config["baseline"]) / "params")),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10, peak_lr=config["learning_rate"], decay_steps=10000, decay_lr=config["learning_rate"] / 4
        ),
    )
    sample = {"state": np.zeros((1, 32), np.float32), **{k: np.zeros((1, 224, 224, 3), np.uint8) for k in IMAGE_MAP}}
    agent = _create_acob_agent(
        seed=config["seed"],
        sample_obs=sample,
        sample_action=np.zeros((1, 3), np.float32),
        task_desc=PROMPT,
        pi_train_config=train,
        image_keys=tuple(IMAGE_MAP),
        encoder_type="resnet-pretrained",
        discount=config["discount"] if discount is None else discount,
        fix_gripper=True,
        pipette_xyz=True,
        actor_flow_weight=config["actor_flow_weight"],
        actor_imp_weight=config["actor_improvement_weight"],
        actor_ref_weight=config["actor_reference_weight"],
        critic_vla_cons_weight=0.0,
        critic_vla_cons_frequency_scale=2,
        actor_imp_margin=0.0,  # m_pi: start at zero (operator decision).
        actor_imp_tau=config["actor_improvement_temperature"],
        critic_intervention_pref_weight=config["critic_preference_weight"],
        critic_intervention_pref_margin=config["critic_preference_margin"],
        resume_pi=False,
        action_horizon=1,
        pi_sample_steps=10,
        debug_enabled=False,
        critic_resnet10_params_path=str(WORKSPACE / "VLA-Precision/assets/resnet10_params.pkl"),
    )
    # All base leaves remain frozen; reference parameters keep the initial BC.
    from vla_precision.acob_stream.checkpoints import initialize_fresh_lora

    return initialize_fresh_lora(agent)


def trainable(agent):
    return agent.state.pi_state.params.filter(agent.pi_train_config.trainable_filter)


def save_checkpoint(agent, config, step, path, *, training_episode_ids=None, replay_sampling=None, parent_checkpoint=None):
    """Atomic portable LoRA + critic + optimizer snapshot; no frozen 3B duplication."""
    import jax
    from flax import serialization

    pi = agent.state.pi_state
    critic = agent.state.critic_state
    payload = {
        "step": step,
        "pi_step": int(pi.step),
        "lora": trainable(agent).to_pure_dict(),
        "pi_opt_leaves": jax.tree.leaves(pi.opt_state),
        "critic_leaves": jax.tree.leaves(critic),
    }
    data = serialization.msgpack_serialize(jax.device_get(payload))
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / "state.msgpack").write_bytes(data)
    meta = {
        "format": "pipette.acob.v1",
        "baseline": config.get("checkpoint_baseline", config["baseline"]),
        "norm_sha256": config["norm_sha256"],
        "contract_sha256": config["contract_sha256"],
        "action_representation": "delta_xyz",
        "policy_profile": "pipette_pi05_hil293",
        "model_horizon": 10,
        "credit_horizon": 1,
        "step": step,
        "state_sha256": hashlib.sha256(data).hexdigest(),
        "m_pi": 0.0,
        "discount": float(agent.algo_config['discount']),
        "parent_checkpoint": str(Path(parent_checkpoint).resolve()) if parent_checkpoint else None,
    }
    if training_episode_ids is not None:
        meta["training_episode_ids"] = sorted(training_episode_ids)
    meta["replay_sampling"] = replay_sampling
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))
    (path / "READY").write_text("complete\n")
    return path


def read_checkpoint(path, config=None):
    from flax import serialization

    path = Path(path).resolve()
    if not (path / "READY").is_file():
        raise ValueError("Incomplete ACoB checkpoint")
    m = json.loads((path / "metadata.json").read_text())
    raw = (path / "state.msgpack").read_bytes()
    if (
        m.get("format") != "pipette.acob.v1"
        or m.get("action_representation") != "delta_xyz"
        or hashlib.sha256(raw).hexdigest() != m["state_sha256"]
    ):
        raise ValueError("Invalid ACoB checkpoint")
    if config and (m["contract_sha256"] != config["contract_sha256"]
                   or m["baseline"] != config.get("checkpoint_baseline", config["baseline"])
                   or m.get("norm_sha256") != config["norm_sha256"]):
        raise ValueError("ACoB checkpoint run/baseline contract mismatch")
    return m, serialization.msgpack_restore(raw)


def restore_checkpoint(agent, config, path):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    m, p = read_checkpoint(path, config)
    lora = trainable(agent)
    lora.replace_by_pure_dict(jax.tree.map(jnp.asarray, p["lora"]))
    model = nnx.merge(agent.state.pi_state.model_def, agent.state.pi_state.params)
    nnx.update(model, lora)
    pi = dataclasses.replace(
        agent.state.pi_state,
        params=nnx.state(model),
        step=p["pi_step"],
        opt_state=_restore_leaves(agent.state.pi_state.opt_state, p["pi_opt_leaves"]),
    )
    critic = _restore_leaves(agent.state.critic_state, p["critic_leaves"])
    return agent.replace(state=agent.state.replace(pi_state=pi, critic_state=critic)), m["step"]


class ContextCache:
    """Frozen prefix contexts, bounded in RAM, keyed by exact input content."""

    def __init__(self, agent, limit=16):
        self.agent = agent
        self.limit = limit
        self.cache = {}

    def get(self, observation):
        import jax

        h = hashlib.sha256()
        for k in sorted(observation):
            h.update(k.encode())
            h.update(np.asarray(observation[k]).tobytes())
        key = h.hexdigest()
        if key not in self.cache:
            e, m, o = jax.device_get(self.agent.encode_context(observation))
            self.cache[key] = (e[0], m[0], o[0])
            if len(self.cache) > self.limit:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def make_batch(rows, cache):
    import jax.numpy as jnp
    from flax.core import freeze

    batch = {
        k: np.stack([r[k] for r in rows])
        for k in ["actions", "intervention_bad_actions", "rewards", "masks", "dones", "intervened", "episode_succeed"]
    }
    batch["bc_intervened"] = np.asarray([r.get("bc_intervened", r["intervened"]) for r in rows])
    batch["bc_eligible"] = np.asarray([r.get("bc_eligible", True) for r in rows])
    for key, prefix in [("observations", ""), ("next_observations", "next_")]:
        batch[key] = {k: np.stack([r[key][k] for r in rows]) for k in rows[0][key]}
        contexts = [cache.get(r[key]) for r in rows]
        for index, name in enumerate(["context_embeddings", "context_masks", "context_offsets"]):
            batch[prefix + name] = np.stack([c[index] for c in contexts])
    return freeze(__import__("jax").tree.map(jnp.asarray, batch))


def _restore_leaves(template, leaves):
    import jax
    import jax.numpy as jnp

    before, tree = jax.tree.flatten(template)
    if len(before) != len(leaves):
        raise ValueError("Optimizer/critic checkpoint structure mismatch")
    for a, b in zip(before, leaves):
        if np.shape(a) != np.shape(b):
            raise ValueError("Optimizer/critic checkpoint shape mismatch")
    return jax.tree.unflatten(tree, [jnp.asarray(x) for x in leaves])
