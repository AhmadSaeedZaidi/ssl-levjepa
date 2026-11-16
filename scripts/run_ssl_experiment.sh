#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
dataset=data/ssl_sumatra_pilot_v2
test -f "$dataset/metadata.json"
.venv/bin/python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$dataset/audit.json"
stamp=$(date +%Y%m%d_%H%M%S)
experiment="runs/ssl_experiment_$stamp"
mkdir -p "$experiment"
printf '%s\n' "$experiment" > logs/latest_ssl_experiment.txt
for split in train validation test; do
  .venv/bin/python tools/preflight.py "$dataset/$split" | tee "$experiment/preflight_$split.log"
done
.venv/bin/python -u tools/benchmark_batches.py --output "$experiment/batch_benchmark" 2>&1 | tee "$experiment/benchmark.log"
bash scripts/finish_ssl_experiment.sh "$experiment"
