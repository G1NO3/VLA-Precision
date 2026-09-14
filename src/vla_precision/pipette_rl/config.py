"""Pipette integration contract, separate from UR hardware configuration."""

from pathlib import Path
import hashlib, json
import math
import yaml

WORKSPACE = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = WORKSPACE / "VLA-Precision/configs/pipette_rl/hil293.yaml"
IMAGE_MAP = {"base_0_rgb": "rgb", "left_wrist_0_rgb": "wrist_left", "right_wrist_0_rgb": "wrist_right"}
PROMPT = "Attach a green pipette tip from the tip rack to the pipette."


RUNTIME_PATHS = {"baseline", "replay_db", "output", "norm_stats"}


def resolve_training_discount(config, requested=None, resumed_metadata=None):
    """Discount is a learner experiment setting; preserve existing data identity.

    Explicit overrides create a new experiment. Continuing a saved experiment
    without an override inherits its discount, including legacy checkpoints.
    """
    value=requested if requested is not None else (resumed_metadata or {}).get('discount',config['discount'])
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
        raise ValueError('Training discount must be finite and strictly between 0 and 1')
    return float(value)


def check_discount_fork(output, source, previous, requested):
    if previous!=requested and Path(output).resolve()==Path(source).resolve().parent:
        raise ValueError('Changed discount requires a separate --output directory; preserve the previous experiment')


def bind_source_contract(c, source):
    """Keep the collector identity while relocating only filesystem paths.

    An explicit source contract is required; never silently accept a foreign
    replay hash, changed learning hyperparameters or normalization assets.
    """
    origin = dict(source)
    digest = origin.pop("contract_sha256")
    if hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest() != digest:
        raise ValueError("Source contract checksum mismatch")
    local_semantics = {k: v for k, v in c.items() if k not in RUNTIME_PATHS | {"contract_sha256"}}
    source_semantics = {k: v for k, v in origin.items() if k not in RUNTIME_PATHS}
    if local_semantics != source_semantics:
        raise ValueError("Remote configuration differs from collector contract beyond paths")
    c["contract_sha256"] = digest
    c["checkpoint_baseline"] = origin["baseline"]
    return c


def load_config(path=DEFAULT_CONFIG, *, source_contract=None, replay_db=None):
    c = yaml.safe_load(Path(path).read_text())
    for key in ("baseline", "replay_db", "output"):
        c[key] = str((WORKSPACE / c[key]).resolve())
    if c["action_hz"] != 30 or c["model_horizon"] != 10 or c["credit_horizon"] != 1:
        raise ValueError("Pipette integration requires 30 Hz, model horizon 10, credit horizon 1")
    if c["policy_profile"] != "pipette_pi05_hil293":
        raise ValueError("Only the HIL293 relative baseline is supported")
    base = Path(c["baseline"])
    profile = WORKSPACE / "VLAPolicyBridge/vla_policy_bridge/config/hil_profiles/pipette_pi05_hil293.json"
    p = json.loads(profile.read_text())
    stats = list((base / "assets").rglob("norm_stats.json"))
    if len(stats) != 1 or hashlib.sha256(stats[0].read_bytes()).hexdigest() != p["norm_stats_sha256"]:
        raise ValueError("Baseline normalization does not match HIL293 delta XYZ")
    if not (base / "params/_METADATA").is_file():
        raise FileNotFoundError(base)
    provenance = json.loads((base.parents[1] / "provenance.json").read_text())
    if provenance.get("action_representation") not in (
        "delta_xyz",
        "consecutive reference FK XYZ displacement, pelvis frame, metres per 30 Hz step",
    ):
        raise ValueError("Absolute/unknown model refused")
    c["norm_stats"] = str(stats[0])
    c["norm_sha256"] = p["norm_stats_sha256"]
    c["contract_sha256"] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    if source_contract is not None:
        c = bind_source_contract(c, json.loads(Path(source_contract).read_text()))
    if replay_db is not None:
        if source_contract is None:
            raise ValueError("--replay-db requires an explicit source contract")
        c["replay_db"] = str(Path(replay_db).resolve())
    return c
