#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:-data/ssl_sumatra_pilot_v2}
test ! -e "$output" || { echo "Refusing to overwrite $output" >&2; exit 2; }
mkdir -p "$output/trajectories" logs
for session in $(seq 0 11); do
  name=$(printf 'session_%03d' "$session")
  timeout --signal=INT --kill-after=15s 180s bash scripts/record_sumatra.sh \
    "$output/trajectories/$name.jsonl" "$session" 6 128 20261116 \
    2>&1 | tee "$output/trajectories/$name.log"
done
bash scripts/record_sumatra.sh "$output/replay_check/session_000.jsonl" 0 1 128 20261116 \
  2>&1 | tee "$output/replay_check.log"
.venv/bin/python tools/render_ssl_dataset.py --input "$output"
.venv/bin/python tools/audit_ssl_dataset.py --dataset "$output"
