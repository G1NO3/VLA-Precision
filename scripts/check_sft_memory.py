#!/usr/bin/env python3
"""Compile a native SFT update using abstract weights; allocate no train state."""
import argparse
import functools
import json
from pathlib import Path
import jax
import numpy as np
from openpi.shared import array_typing as at
from openpi.training import sharding
from vla_precision.config import load_stage1_config
from vla_precision.integrations.openpi.configs import build_stage1_train_config
from vla_precision.integrations.openpi.train import init_train_state, train_step

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/stage1/pipette_pi05.yaml'))
    p.add_argument('--output', type=Path, default=Path('../outputs/pi05-sft-20260909/memory-compile.json'))
    args = p.parse_args()
    c = build_stage1_train_config(load_stage1_config(args.config).config)
    mesh = sharding.make_mesh(c.fsdp_devices)
    state, ss = init_train_state(c, jax.random.key(0), mesh, resume=True)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    step = jax.jit(functools.partial(train_step, c), in_shardings=(rep, ss, data),
                   out_shardings=(ss, rep), donate_argnums=(1,))
    print(f'Compiling abstract SFT {c.name}: batch={c.batch_size}, horizon={c.model.action_horizon}, EMA={c.ema_decay}', flush=True)
    # JAX .lower creates ArgInfo sentinels that the pinned jaxtyping rejects.
    # Disable checks only for this abstract compilation, not actual training.
    with at.disable_typechecking(), sharding.set_mesh(mesh):
        executable = step.lower(jax.random.key(0), state, c.model.inputs_spec(batch_size=c.batch_size)).compile()
    analysis = executable.memory_analysis()
    result = {key: int(getattr(analysis, key)) for key in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes', 'temp_size_in_bytes')}
    result['estimated_peak_GiB'] = (result['argument_size_in_bytes'] + result['output_size_in_bytes']
                                  - result['alias_size_in_bytes'] + result['temp_size_in_bytes']) / 2**30
    result['parameter_count'] = int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params)))
    result['trainable_parameter_count'] = int(sum(np.prod(x.shape) for x in jax.tree.leaves(state.params.filter(c.trainable_filter))))
    result['config'] = str(args.config.resolve())
    result['note'] = 'Compiler estimate, not a successfully executed optimizer step; runtime may need additional memory.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
