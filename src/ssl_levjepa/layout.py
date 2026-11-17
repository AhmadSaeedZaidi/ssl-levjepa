"""Repository locations and read-only resolution of historical artifact paths."""
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(os.environ.get("SSL_LEVJEPA_ROOT", Path(__file__).resolve().parents[2])).resolve()
UPSTREAM = ROOT / "upstream/LeVJEPA"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / ".cache/simulators"
DEFAULT_DATASET = OUTPUTS / "datasets/sumatra_pilot_v2"
DEFAULT_CONFIG = ROOT / "configs/sumatra_pilot.yaml"
TOY_CHECKPOINT = OUTPUTS / "archive/toy_infrastructure/runs/toy_vit_tiny_20261116_045716/checkpoints/final.ckpt"


def resolve_path(value, base=None):
    """Resolve current paths or an old saved path without editing the saved artifact."""
    raw = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", raw):
        raw = "/mnt/" + raw[0].lower() + raw[2:]
    path = Path(raw)
    if not path.is_absolute():
        path = Path(base or ROOT) / path
    path = path.resolve()
    if path.exists():
        return path
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        # Historical Linux paths can be inspected from Windows, too.
        marker = "/ssl-levjepa/"
        if marker not in raw:
            return path
        relative = raw.split(marker, 1)[1]
    migration = OUTPUTS / "migration.json"
    if migration.exists():
        mappings = json.loads(migration.read_text(encoding="utf-8-sig"))["paths"]
        for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
            if relative == old or relative.startswith(old + "/"):
                suffix = relative[len(old):].lstrip("/")
                return (ROOT / new / suffix).resolve()
    return path


def dataset_path(value=None):
    if value is None:
        return DEFAULT_DATASET
    raw = str(value)
    if "/" not in raw and "\\" not in raw:
        return OUTPUTS / "datasets" / raw
    return resolve_path(value)


def experiment_dataset(experiment):
    experiment = resolve_path(experiment)
    reference = json.loads((experiment / "dataset.json").read_text())
    return resolve_path(reference["path"], base=experiment)


def enable_upstream():
    """Keep the pinned upstream's existing absolute module imports intact."""
    path = str(UPSTREAM)
    if path not in sys.path:
        sys.path.insert(0, path)
