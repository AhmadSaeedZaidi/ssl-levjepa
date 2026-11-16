#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
.tools/uv sync --locked --extra simulation
mkdir -p .sim
pin=0bb4c653ef8611f9e71a880165f101209be97b3c
if [ ! -d .sim/Sumatra ]; then
  git clone --no-checkout https://github.com/TIGERs-Mannheim/Sumatra.git .sim/Sumatra
fi
if [ "$(git -C .sim/Sumatra rev-parse HEAD)" != "$pin" ]; then
  test -z "$(git -C .sim/Sumatra status --porcelain)" || { echo 'Sumatra has local changes; preserving them' >&2; exit 2; }
  git -C .sim/Sumatra fetch --depth 1 origin "$pin"
  git -C .sim/Sumatra switch --detach "$pin"
fi
if [ ! -f .sim/sumatra.zip ]; then
  curl --fail --location https://github.com/TIGERs-Mannheim/Sumatra/releases/download/2025/sumatra.zip -o .sim/sumatra.zip
fi
printf '%s\n' '262a8355112464c70ebcf5de6bf895039cef73706b3383c9b7d5aa7b13023b37  .sim/sumatra.zip' | sha256sum --check
if [ ! -d .sim/sumatra-dist ]; then
  .venv/bin/python - <<'PY'
from pathlib import Path
import zipfile
root=Path('.sim/sumatra-dist').resolve()
with zipfile.ZipFile('.sim/sumatra.zip') as archive:
    for member in archive.namelist():
        if not (root/member).resolve().is_relative_to(root):
            raise ValueError('Archive path escapes runtime directory')
    archive.extractall(root)
PY
fi
java -version
java -m jdk.compiler/com.sun.tools.javac.Main -version
