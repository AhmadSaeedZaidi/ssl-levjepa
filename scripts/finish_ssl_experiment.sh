#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
experiment=$1
dataset=data/ssl_sumatra_pilot_v2
test ! -e "$experiment/baseline" || { echo 'Existing baseline preserved; resume explicitly into a new directory' >&2; exit 2; }
test -f "$dataset/audit.json"
batch=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "$experiment/batch_benchmark/recommendation.json")
nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,memory.used --format=csv -lms 500 > "$experiment/baseline_gpu.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true; wait "$telemetry" 2>/dev/null || true' EXIT
timeout --signal=INT --kill-after=60s 30m .venv/bin/python -u tools/train_ssl.py \
  --run-dir "$experiment/baseline" --steps 300 --warmup 20 --batch-size "$batch" \
  2>&1 | tee "$experiment/baseline.log"
kill "$telemetry" 2>/dev/null || true
wait "$telemetry" 2>/dev/null || true
.venv/bin/python tools/verify_checkpoint.py "$experiment/baseline/checkpoints/final.ckpt" | tee "$experiment/checkpoint_reload.log"
.venv/bin/python -u tools/evaluate_ball_motion.py --dataset "$dataset" --run "$experiment/baseline" \
  --output "$experiment/probe" 2>&1 | tee "$experiment/probe.log"
.venv/bin/python tools/report_ssl_experiment.py --dataset "$dataset" --experiment "$experiment"
