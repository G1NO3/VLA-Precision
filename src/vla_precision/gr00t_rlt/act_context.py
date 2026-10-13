"""Versioned critic-only execution context; no Torch or robot dependencies."""
import numpy as np

MAX_CHUNKS = 16
HORIZON = 30
GROUPS = (("left_arm", 7), ("right_arm", 7), ("waist", 3), ("left_hand", 6), ("right_hand", 6))
PLAN_DIM = sum(n for _, n in GROUPS)
ROW_DIM = 2 + HORIZON * PLAN_DIM  # present, effective age, complete decoded plan
MODES = ("normal", "await_neutral", "correcting", "resume_hold", "resume_ramp", "fault")
TAIL_DIM = 1 + 29 + 12 + len(MODES) + 2  # catchup, commands, mode, boundary wait, publication age
CONTEXT_DIM = MAX_CHUNKS * ROW_DIM + TAIL_DIM
CONTRACT = "async_act_proposal_v1"


def pack_context(chunks, now, *, command_q, hands, catchup, mode="normal", boundary=None, published_at=None):
    """Snapshot the ACT queue at observation/request time, not response time.

    Effective age follows the production scheduler's latency-compensated clock.
    Only expired plans are omitted. Hands are divided by 1000, joints by pi.
    Overflow/shape/NaN are errors, never silently truncate the real queue.
    """
    live = sorted((c for c in chunks if now-(c.issue_time-c.obs_latency_s) < c.horizon*c.step_dt),
                  key=lambda c: c.issue_time)
    if len(live) > MAX_CHUNKS:
        raise ValueError("ACT critic context overflow")
    rows = np.zeros((MAX_CHUNKS, ROW_DIM), np.float32)
    for i, chunk in enumerate(live):
        if chunk.horizon != HORIZON or not np.isclose(chunk.step_dt, 1/60):
            raise ValueError("ACT context requires 30 rows at 60 Hz")
        values = []
        for key, size in GROUPS:
            arr = np.asarray(chunk.arrays[key], np.float32)
            if arr.shape != (HORIZON, size):
                raise ValueError("ACT context plan shape: " + key)
            values.append(arr / (1000 if key.endswith("_hand") else np.pi))
        rows[i, :2] = (1, now-(chunk.issue_time-chunk.obs_latency_s))
        rows[i, 2:] = np.concatenate(values, axis=-1).ravel()
    mode_bits = np.zeros(len(MODES), np.float32)
    mode_bits[MODES.index(mode)] = 1
    q, h = np.asarray(command_q), np.asarray(hands)
    if q.shape != (29,) or h.shape != (12,):
        raise ValueError("ACT context command shape mismatch")
    tail = np.r_[float(catchup), q/np.pi, h/1000, mode_bits,
                 max(0., boundary-now) if boundary is not None else 0.,
                 max(0., now-published_at) if published_at is not None else 0.]
    result = np.r_[rows.ravel(), tail].astype(np.float32)
    if result.shape != (CONTEXT_DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid ACT context")
    return result
