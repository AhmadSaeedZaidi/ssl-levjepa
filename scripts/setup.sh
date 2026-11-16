#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools logs
export UV_CACHE_DIR="$PWD/.cache/uv"
if [ ! -x .tools/uv ]; then
  curl --fail --location https://astral.sh/uv/install.sh -o .tools/install-uv.sh
  UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh .tools/install-uv.sh
fi
.tools/uv --version | tee logs/uv-version.txt
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
date -Is | tee -a logs/setup.log
.tools/uv sync --locked --python /usr/bin/python3 2>&1 | tee -a logs/setup.log
.tools/uv pip freeze --python .venv/bin/python > logs/requirements-frozen.txt
