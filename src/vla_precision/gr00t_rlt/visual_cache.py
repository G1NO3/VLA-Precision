"""Per-observation cache of the 512 frozen visual prefix tokens for the
cross-attention editor (model.XAttnEditorHeads).

Each captured observation (``<run>/observations/<id>.npz``) already holds the
served PI05 prefix (968 x 2048, fp16). The editor needs only the two real
views' tokens, so they are copied once to ``<cache>/<id>.npy`` (512 x 2048
fp16, 2 MB) and read back per batch; nothing is recomputed. The replay export
has already hash-checked these observations against their receipts.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np

from .model import VISUAL_TOKENS


def default_cache(c):
    return Path(c["output"]) / "visual-tokens"


def cache_file(cache, observation_file):
    name = Path(str(observation_file)).name
    if not name.endswith(".npz"):
        raise ValueError(f"Unexpected observation file {observation_file}")
    return Path(cache) / (name[:-4] + ".npy")


def extract(path, contract_sha256):
    """(512, 2048) fp16 visual tokens of one observation file."""
    with np.load(path, allow_pickle=False) as f:
        if str(f["contract_sha256"]) != contract_sha256:
            raise ValueError(f"{Path(path).name}: observation recipe mismatch")
        prefix, mask = f["prefix"], f["mask"]
    if prefix.ndim != 2 or prefix.shape[1] != 2048 or len(prefix) < 768 or mask.shape != prefix.shape[:1]:
        raise ValueError(f"{Path(path).name}: unexpected PI05 prefix layout")
    if not mask[:VISUAL_TOKENS].all() or mask[VISUAL_TOKENS:768].any():
        raise ValueError(f"{Path(path).name}: editor requires two real views and a masked third view")
    tokens = np.ascontiguousarray(prefix[:VISUAL_TOKENS], dtype=np.float16)
    if not np.isfinite(tokens).all():
        raise ValueError(f"{Path(path).name}: non-finite visual tokens")
    return tokens


def build(run, files, cache, contract_sha256, *, workers=8, report=None):
    """Cache every listed observation not cached yet; returns (cached, added)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    files = list(dict.fromkeys(str(f) for f in files))
    todo = [f for f in files if not cache_file(cache, f).exists()]

    def one(rel):
        dest = cache_file(cache, rel)
        tmp = dest.with_name(dest.name + ".tmp.npy")
        np.save(tmp, extract(Path(run) / rel, contract_sha256))
        tmp.replace(dest)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(one, todo), 1):
            if report and (n % 500 == 0 or n == len(todo)):
                report(n, len(todo))
    return len(files) - len(todo), len(todo)


class Loader:
    """Visual tokens of a batch's current and next observations, read one batch ahead."""
    def __init__(self, cache, observation_files, next_observation_files, workers=8):
        self.cache = Path(cache)
        self.files, self.next_files = observation_files, next_observation_files
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)       # file reads
        self.ahead = concurrent.futures.ThreadPoolExecutor(1)            # the next batch (never nested in pool)

    def _stack(self, files):
        return np.stack(list(self.pool.map(lambda f: np.load(cache_file(self.cache, f)), files)))

    def _read(self, ids):
        return self._stack(self.files[ids]), self._stack(self.next_files[ids])

    def submit(self, ids):
        return self.ahead.submit(self._read, np.asarray(ids))
