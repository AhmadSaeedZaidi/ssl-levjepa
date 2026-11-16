#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f data/toy_ssl/preflight.json
test -f logs/validation_passed.json
checkpoint=$(.venv/bin/python -c 'import json; v=json.load(open("logs/validation_passed.json")); assert v["passed"]; print(v["checkpoint"])')
test -f "$checkpoint"
run_dir=${1:-"runs/toy_vit_tiny_$(date +%Y%m%d_%H%M%S)"}
exit_code=0
bash scripts/run_stage.sh "$run_dir" 1200 --resume "$checkpoint" || exit_code=$?
.venv/bin/python tools/report.py --run "$run_dir"
exit "$exit_code"
