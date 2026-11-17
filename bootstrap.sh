#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .tools .cache/bootstrap
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
if [ ! -x .tools/uv ]; then
  curl --fail --location https://astral.sh/uv/install.sh -o .cache/bootstrap/install-uv.sh
  UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh .cache/bootstrap/install-uv.sh
fi
log=".cache/bootstrap/$(date +%Y%m%d_%H%M%S).log"
.tools/uv sync --locked --extra simulation 2>&1 | tee "$log"
.venv/bin/ssl-levjepa --help
