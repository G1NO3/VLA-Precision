"""Learner-only sampling settings; do not change collected transition identity."""
import json
import math
from pathlib import Path
from .config import WORKSPACE

DEFAULT_SAMPLING_CONFIG = WORKSPACE / 'VLA-Precision/configs/pipette_rl/terminal_tail_sampling.json'


def validate_sampling(value):
    if value is None:
        return None  # Exact legacy correction + uniform sampler.
    expected={'schema_version','success_tail_fraction','success_tail_seconds','terminal_fraction_within_tail'}
    if not isinstance(value,dict):
        raise ValueError('Unknown replay sampling schema')
    version=value.get('schema_version')
    if version==2:
        expected |= {'recent_episode_fraction','recent_episode_count'}
    if version in (3,4):
        expected |= {'latest_collection_fraction'}
    if version==4:
        expected -= {'success_tail_fraction','success_tail_seconds','terminal_fraction_within_tail'}
        expected |= {'terminal_tail_fraction','terminal_tail_seconds'}
    if isinstance(version,bool) or version not in (1,2,3,4) or set(value)!=expected:
        raise ValueError('Unknown replay sampling schema')
    result=dict(value)
    for key in expected-{'schema_version'}:
        v=value[key]
        if key=='recent_episode_count':
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError(f'Invalid sampling value: {key}')
            continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError(f'Invalid sampling value: {key}')
        if (key.endswith('_seconds') and v<=0) or (not key.endswith('_seconds') and not 0<=v<=1):
            raise ValueError(f'Invalid sampling range: {key}')
    return result


def load_sampling(path=DEFAULT_SAMPLING_CONFIG):
    return validate_sampling(json.loads(Path(path).read_text()))


def tail_settings(sampling):
    """Return (fraction, seconds, successes_only), preserving old run semantics."""
    if not sampling:
        return 0., 0., False
    if sampling['schema_version']==4:
        return sampling['terminal_tail_fraction'], sampling['terminal_tail_seconds'], False
    return sampling['success_tail_fraction'], sampling['success_tail_seconds'], True
