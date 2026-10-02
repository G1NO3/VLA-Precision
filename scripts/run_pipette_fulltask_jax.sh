#!/usr/bin/env bash
# Native JAX only. One process owns the CUDA_VISIBLE_DEVICES mesh; no torchrun.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-check}"
CONFIG="${2:?usage: bash scripts/run_pipette_fulltask_jax.sh check|norm|train CONFIG.yaml}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/../models/openpi}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$REPO_ROOT/../.cache/jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export LD_LIBRARY_PATH="$REPO_ROOT/.runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
case "$MODE" in
  norm)
    JAX_PLATFORMS=cpu "$PYTHON" main.py --stage stage1 --mode norm-stats --config "$CONFIG"
    ;;
  check)
    JAX_PLATFORMS=cpu "$PYTHON" scripts/check_pipette_fulltask.py --config "$CONFIG"
    ;;
  train)
    # Fail before GPU allocation if data/assets/caches are incomplete.
    JAX_PLATFORMS=cpu "$PYTHON" scripts/check_pipette_fulltask.py --config "$CONFIG" --require-cached
    JAX_PLATFORMS=cuda "$PYTHON" main.py --stage stage1 --mode train --config "$CONFIG"
    ;;
  *) echo "Unknown mode: $MODE (check|norm|train)" >&2; exit 2 ;;
esac
