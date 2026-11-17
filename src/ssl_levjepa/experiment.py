"""Experiment bundles, stage orchestration and their dataset references."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .layout import DEFAULT_CONFIG, DEFAULT_DATASET, OUTPUTS, ROOT, dataset_path, experiment_dataset, resolve_path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def paths(output=None, name="sumatra_baseline"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Experiment name must contain letters, numbers, underscores or hyphens")
    return resolve_path(output) if output else OUTPUTS / (datetime.now().strftime("%Y-%m-%d_%H%M%S_") + name)


def prepare_config(source, dataset, output, batch_size=None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(resolve_path(source))
    cfg.dataset.path = str(dataset / "train/train.lance")
    cfg.dataset.episodes_path = str(dataset / "train/train.episodes.npz")
    if cfg.get("validation_dataset"):
        cfg.validation_dataset.path = str(dataset / "validation/train.lance")
        cfg.validation_dataset.episodes_path = str(dataset / "validation/train.episodes.npz")
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    OmegaConf.save(cfg, output)
    return cfg


def stage_commands(experiment, dataset, config):
    executable = [sys.executable, "-m", "ssl_levjepa"]
    stages = []
    for split in ("train", "validation", "test"):
        stages.append(("validate_" + split, executable + ["dataset", "validate", str(dataset / split),
                       "--output", str(experiment / "provenance" / ("preflight_" + split + ".json"))]))
    stages.append(("benchmark", executable + ["benchmark", "--config", str(config),
                   "--output", str(experiment / "benchmark"), "--log-dir", str(experiment / "logs/benchmark")]))
    stages.append(("training", executable + ["train", "--config", str(config), "--output", str(experiment / "training"), "--warmup", "20"]))
    stages.append(("checkpoint", executable + ["checkpoint", "verify", str(experiment / "training/checkpoints/final.ckpt"),
                   "--output", str(experiment / "provenance/checkpoint_verified.json")]))
    stages.append(("evaluation", executable + ["evaluate", "--run", str(experiment), "--output", str(experiment / "evaluation")]))
    stages.append(("report", executable + ["report", "--run", str(experiment)]))
    return stages


def execute(command, log, timeout=1800):
    """Preserve child exit codes and allow graceful shutdown before forced termination."""
    with Path(log).open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt")
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGINT)
            else:
                child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(config=DEFAULT_CONFIG, dataset=DEFAULT_DATASET, name="sumatra_baseline", output=None, plan=False):
    config = resolve_path(config)
    dataset = dataset_path(dataset)
    destination = paths(output, name)
    commands = stage_commands(destination, dataset, destination / "config.yaml")
    result = {"experiment": str(destination), "dataset": str(dataset), "stages": [{"stage": s, "command": c} for s, c in commands]}
    if plan:
        return result
    audit = json.loads((dataset / "audit.json").read_text())
    if not audit.get("passed"):
        raise ValueError("Dataset audit has not passed")
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "reports", "provenance"):
        (destination / directory).mkdir()
    source_config = destination / "provenance/source_config.yaml"
    source_config.write_bytes(config.read_bytes())
    prepare_config(config, dataset, destination / "config.yaml")
    write_json(destination / "dataset.json", {
        "id": dataset.name, "path": os.path.relpath(dataset, destination).replace(os.sep, "/"),
        "metadata_sha256": checksum(dataset / "metadata.json"), "clips_sha256": checksum(dataset / "clips.jsonl"),
    })
    state = {"created_at": datetime.now().astimezone().isoformat(), "state": "running", "stages": {}}
    for stage, command in commands:
        state["stages"][stage] = "running"
        write_json(destination / "status.json", state)
        try:
            execute(command, destination / "logs" / (stage + ".log"))
            if stage == "benchmark":
                selected = json.loads((destination / "benchmark/recommendation.json").read_text())["batch_size"]
                prepare_config(source_config, dataset, destination / "config.yaml", selected)
        except BaseException as error:
            state.update(state="failed", failed_stage=stage, error=str(error))
            write_json(destination / "status.json", state)
            raise
        state["stages"][stage] = "complete"
    state["state"] = "complete"
    write_json(destination / "status.json", state)
    write_json(OUTPUTS / "latest.json", {"experiment": destination.name, "path": os.path.relpath(destination, ROOT)})
    return destination


def build_dataset(config=DEFAULT_CONFIG, output=None):
    from . import dataset as data
    from . import simulator
    # Generation currently reproduces the documented 12-session pilot protocol.
    target = dataset_path(output)
    if target.exists():
        raise FileExistsError(f"Dataset already exists and will be preserved: {target}")
    if not resolve_path(config).is_file():
        raise FileNotFoundError(config)
    simulator.generate(target)
    data.build(target)
    data.audit(target, output=target / "audit.json")
    return target


def info(run=None):
    if run is None:
        latest = json.loads((OUTPUTS / "latest.json").read_text(encoding="utf-8-sig"))
        run = latest["path"]
    experiment = resolve_path(run)
    result = {"experiment": str(experiment), "dataset": str(experiment_dataset(experiment)),
              "reports": str(experiment / "reports"), "checkpoint": str(experiment / "training/checkpoints/final.ckpt")}
    status = experiment / "status.json"
    if status.exists():
        result["status"] = json.loads(status.read_text())
    summary = experiment / "training/summary.json"
    if summary.exists():
        result["training"] = json.loads(summary.read_text())
    return result
