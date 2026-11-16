#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(realpath .sim/erforce-image/rootfs)
loader=$(find "$root" -type f -name 'ld-*.so' -print -quit)
test -n "$loader"
library_path="$root/usr/lib/x86_64-linux-gnu:$root/lib/x86_64-linux-gnu:$root/usr/local/lib"
export QT_QPA_PLATFORM=offscreen
cd "$root/home/default"
exec "$loader" --library-path "$library_path" "$root/home/default/build/bin/simulator-cli" "$@"
