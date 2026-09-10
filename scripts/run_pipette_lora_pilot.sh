#!/usr/bin/env bash
# Native JAX pilot and held-out evaluation, suitable for a dedicated tmux session.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_training_env.sh
run_dir="$(realpath ../outputs/pi05-lora-sft-20260909)"
config_file=configs/stage1/pipette_pi05_lora.yaml
trap 'code=$?; printf "failed (exit %s)\n" "$code" > "$run_dir/status.txt"; exit "$code"' ERR
python - "$run_dir/memory-compile.json" <<'PY'
import json,subprocess,sys
report=json.load(open(sys.argv[1]))
free_mib=float(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if report['estimated_peak_GiB'] + 2 > free_mib / 1024:
    raise SystemExit('Compiled training estimate plus 2 GiB headroom exceeds available GPU memory')
print(f'Preflight: estimated {report["estimated_peak_GiB"]:.2f} GiB; free {free_mib/1024:.2f} GiB',flush=True)
PY
python -m vla_precision.cli --stage stage1 --mode resolve-config --config "$config_file" \
    --resolved-config-out "$run_dir/resolved.yaml" > "$run_dir/config.log" 2>&1
printf 'training\n' > "$run_dir/status.txt"
python -u -m vla_precision.cli --stage stage1 --mode train --config "$config_file" > "$run_dir/train.log" 2>&1
printf 'validating\n' > "$run_dir/status.txt"
checkpoint_dir="$(python - "$config_file" <<'PY'
import sys
from vla_precision.config import load_stage1_config
from vla_precision.integrations.openpi.configs import build_stage1_train_config
c=build_stage1_train_config(load_stage1_config(sys.argv[1]).config)
path=c.checkpoint_dir / str(c.num_train_steps - 1)
if not (path / 'params').is_dir():
    raise SystemExit(f'Missing final checkpoint: {path}')
print(path)
PY
)"
python -u scripts/evaluate_pipette_sft.py --config "$config_file" \
    --checkpoint "$checkpoint_dir" --samples-per-episode 16 \
    --output "$run_dir/validation.json" > "$run_dir/validation.log" 2>&1
printf 'complete\n' > "$run_dir/status.txt"
