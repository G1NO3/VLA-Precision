"""Learner reward overrides preserve the collector contract and replay labels."""
import json
import math
from pathlib import Path


def validate_rewards(value):
    keys={'schema_version','success_terminal','failure_terminal','time_reward'}
    if not isinstance(value,dict) or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1:
        raise ValueError('Unknown learner reward schema')
    for key in keys-{'schema_version'}:
        v=value[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError(f'Invalid reward: {key}')
    return dict(value)


def load_rewards(path):
    return validate_rewards(json.loads(Path(path).read_text()))


def resolve_training_rewards(config, requested=None, resumed_metadata=None):
    if requested is not None:
        return validate_rewards(requested)
    inherited=(resumed_metadata or {}).get('training_rewards')
    if inherited is not None:
        return validate_rewards(inherited)
    return validate_rewards(dict(schema_version=1,time_reward=config['time_reward'],
                                 success_terminal=config['time_reward']+config['success_reward'],
                                 failure_terminal=config['time_reward']+config['failure_reward']))


def check_reward_fork(output, source, previous, requested):
    if previous!=requested and Path(output).resolve()==Path(source).resolve().parent:
        raise ValueError('Changed rewards require a separate --output directory')
