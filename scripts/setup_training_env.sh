#!/usr/bin/env bash
# Linux x86_64 training environment, including the local Blackwell CUDA build.
set -euo pipefail
vlap_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vlap_workspace="$(dirname "$vlap_repo")"
cd "$vlap_repo"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$vlap_workspace/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$vlap_workspace/.cache/uv-python}"
export GIT_LFS_SKIP_SMUDGE=1

# evdev is a transitive LeRobot dependency. Reuse only the compiler and headers;
# do not activate or modify the existing starVLA Python environment.
vlap_toolchain="${VLA_BUILD_TOOLCHAIN:-$vlap_workspace/.conda-envs/starVLA}"
if [[ -x "$vlap_toolchain/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="${CC:-$vlap_toolchain/bin/x86_64-conda-linux-gnu-gcc}"
    export CXX="${CXX:-$vlap_toolchain/bin/x86_64-conda-linux-gnu-g++}"
    export CPATH="$vlap_toolchain/x86_64-conda-linux-gnu/sysroot/usr/include${CPATH:+:$CPATH}"
fi
uv sync --frozen --group stage2 --python 3.11

if [[ ! -f "$vlap_repo/.runtime/conda-meta/history" ]]; then
    vlap_conda="${VLA_CONDA_EXE:-$vlap_workspace/.miniconda3/bin/conda}"
    if [[ ! -x "$vlap_conda" ]]; then
        vlap_conda="$(command -v conda)"
    fi
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$vlap_workspace/.conda-pkgs}"
    "$vlap_conda" create --prefix "$vlap_repo/.runtime" \
        --file "$vlap_repo/scripts/runtime-linux-64.lock" --yes --quiet
fi
echo "Ready. Run: source scripts/activate_training_env.sh"
