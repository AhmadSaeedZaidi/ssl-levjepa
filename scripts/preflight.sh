#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
mkdir -p logs
if [ ! -d data/toy_pilot ]; then
  .venv/bin/python tools/make_toy_ssl_lance.py --output data/toy_pilot --episodes 4 --frames 32
fi
.venv/bin/python tools/preflight.py data/toy_pilot | tee logs/preflight-pilot.log
if [ ! -d data/toy_ssl ]; then
  .venv/bin/python tools/make_toy_ssl_lance.py --output data/toy_ssl --episodes 160 --frames 96
fi
.venv/bin/python tools/preflight.py data/toy_ssl | tee logs/preflight-full.log
cp data/toy_ssl/preview.png preview.png
