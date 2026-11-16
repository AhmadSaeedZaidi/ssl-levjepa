#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .sim/generated
.venv/bin/python -m grpc_tools.protoc \
  -I .sim/erforce/src/protobuf --python_out=.sim/generated \
  .sim/erforce/src/protobuf/protobuf/ssl_sim/*.proto \
  .sim/erforce/src/protobuf/protobuf/ssl_vision/*.proto \
  .sim/erforce/src/protobuf/protobuf/ssl_gc/state/ssl_gc_common.proto
