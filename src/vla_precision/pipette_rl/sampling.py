"""Learner-only sampling settings; do not change collected transition identity."""
import json
import math
from pathlib import Path
from .config import WORKSPACE

DEFAULT_SAMPLING_CONFIG = WORKSPACE / 'VLA-Precision/configs/pipette_rl/success_tail_sampling.json'


def validate_sampling(value):
    if value is None:
        return None  # Exact legacy correction + uniform sampler.
    expected={'schema_version','success_tail_fraction','success_tail_seconds','terminal_fraction_within_tail'}
    if not isinstance(value,dict) or set(value)!=expected or value['schema_version']!=1:
        raise ValueError('Unknown success-tail sampling schema')
    result=dict(value)
    for key in expected-{'schema_version'}:
        v=value[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError(f'Invalid sampling value: {key}')
        if (key.endswith('_seconds') and v<=0) or (not key.endswith('_seconds') and not 0<=v<=1):
            raise ValueError(f'Invalid sampling range: {key}')
    return result


def load_sampling(path=DEFAULT_SAMPLING_CONFIG):
    return validate_sampling(json.loads(Path(path).read_text()))
