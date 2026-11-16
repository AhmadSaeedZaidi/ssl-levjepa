#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f data/toy_ssl/preflight.json
stamp=$(date +%Y%m%d_%H%M%S)
prefix="runs/validation_$stamp"
bash scripts/run_stage.sh "${prefix}_one" 1
bash scripts/run_stage.sh "${prefix}_smoke50" 51 --resume "${prefix}_one/checkpoints/final.ckpt"
bash scripts/run_stage.sh "${prefix}_profile" 276 --warmup 25 --resume "${prefix}_smoke50/checkpoints/final.ckpt"
.venv/bin/python tools/check_gates.py "$prefix"
