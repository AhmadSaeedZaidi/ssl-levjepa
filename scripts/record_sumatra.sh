#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
project=$PWD
runtime="$project/.sim/sumatra-dist/sumatra"
mkdir -p .sim/classes
.venv/bin/python - "$runtime" <<'PY'
import pathlib, sys, shutil
runtime=pathlib.Path(sys.argv[1]); source=pathlib.Path('.sim/Sumatra/config')
if not (runtime/'config').exists(): shutil.copytree(source, runtime/'config')
xml=(source/'moduli/sim.xml').read_text()
xml=xml.replace('<geometry>DIV_A</geometry>', '<geometry>DIV_B</geometry>')
(runtime/'config/moduli/ssl_record.xml').write_text(xml)
PY
printf -v classpath '%s:' "$runtime"/lib/*.jar
java -m jdk.compiler/com.sun.tools.javac.Main -proc:none -cp "$classpath" -d .sim/classes simulation/SumatraRecorder.java
output=$(realpath -m "$1")
shift
cd "$runtime"
exec java -Xmx3g -cp "$project/.sim/classes:$runtime/lib/*" SumatraRecorder "$output" "$@"
