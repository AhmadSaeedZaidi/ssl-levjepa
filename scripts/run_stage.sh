#!/usr/bin/env bash
# One monitored process, with an actual 8h wall cap and nonzero errors preserved.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash scripts/run_stage.sh RUN_DIR FINAL_STEP [train_toy.py options]' >&2
  exit 2
fi
run_dir=$1
steps=$2
shift 2
test ! -e "$run_dir" || { echo "Refusing to overwrite $run_dir" >&2; exit 2; }
test -x .venv/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv
mkdir -p logs
log="logs/$(basename "$run_dir").log"
test ! -e "$log" || { echo "Refusing to overwrite $log" >&2; exit 2; }
# Telemetry lives outside run_dir until Python atomically claims the new directory.
gpu_log="logs/$(basename "$run_dir")-gpu.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total,fan.speed --format=csv -l 30 > "$gpu_log" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true; wait "$telemetry" 2>/dev/null || true' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HUB_OFFLINE=1
set +e
timeout --signal=INT --kill-after=60s 8h .venv/bin/python -u tools/train_toy.py \
  --run-dir "$run_dir" --steps "$steps" "$@" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}
set -e
if [ -d "$run_dir" ]; then
  cp "$gpu_log" "$run_dir/gpu.csv"
  printf '%s\n' "$exit_code" > "$run_dir/exit_code.txt"
fi
exit "$exit_code"
