#!/usr/bin/env bash
# Source this file from Bash; it also exposes the local FFmpeg shared libraries.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source scripts/activate_training_env.sh" >&2
    exit 1
fi
_vlap_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$_vlap_repo/.venv/bin/activate" || ! -d "$_vlap_repo/.runtime/lib" ]]; then
    echo "Environment missing; run bash scripts/setup_training_env.sh first." >&2
    unset _vlap_repo
    return 1
fi
source "$_vlap_repo/.venv/bin/activate"
export LD_LIBRARY_PATH="$_vlap_repo/.runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$_vlap_repo/.runtime/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_vlap_repo/../.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$_vlap_repo/../.cache/uv-python}"
# Avoid eager GPU reservation when Actor and Learner share the single local GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$_vlap_repo/../models/openpi}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$_vlap_repo/../.cache/jax}"
unset _vlap_repo
