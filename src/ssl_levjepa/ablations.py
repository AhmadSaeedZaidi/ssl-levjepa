"""Matched crop/flip and temporal coverage experiments on a completed v2 baseline."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from .experiment import checksum, execute, write_json
from .layout import ROOT, UPSTREAM, experiment_dataset, resolve_path

ARMS = {
    "augmentation": {"views": {"local_scale": [0.8, 1.0], "flip_p": 0.0}},
    "per_frame": {"masking": {"iid": False}},
}
PROBE_NAMES = ("global_readout_ridge", "mean_patch_pool_ridge", "spatial_patch_ridge", "spatial_pointwise", "spatial_conv")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol(baseline):
    original = read(baseline / "protocol.json")
    if read(baseline / "status.json")["state"] != "complete" or original["version"] != 2:
        raise ValueError("A completed v2 baseline is required")
    seeds = original["seeds"]
    if len(set(seeds)) != 3:
        raise ValueError("The control must contain three independent seeds")
    dataset = experiment_dataset(baseline)
    cohort = baseline / "evaluation/cohort"
    cohort_hashes = read(cohort / "protocol.json")
    dataset_ref = read(baseline / "dataset.json")
    if checksum(dataset / "clips.jsonl") != dataset_ref["clips_sha256"]:
        raise ValueError("Control dataset manifest changed")
    if checksum(cohort / "clips.jsonl") != cohort_hashes["cohort_sha256"] or checksum(cohort / "targets.npy") != cohort_hashes["targets_sha256"]:
        raise ValueError("Control probe cohort changed")
    return {
        "kind": "baseline_v2_ablations",
        "baseline": str(baseline),
        "dataset": str(dataset),
        "cohort": str(cohort),
        "dataset_sha256": dataset_ref["clips_sha256"],
        "cohort_sha256": cohort_hashes["cohort_sha256"],
        "targets_sha256": cohort_hashes["targets_sha256"],
        "seeds": seeds,
        "steps": original["steps"],
        "batch_size": original["micro_batch"],
        "extraction_batch_size": original["frozen_extraction_batch"],
        "arms": ARMS,
        "control_configs_sha256": {
            str(seed): checksum(baseline / "seeds" / str(seed) / "training/config.yaml") for seed in seeds
        },
        "primary": "Component-wise test velocity RMSE of spatial_conv, paired to original trained control",
        "uncertainty": "2000 paired session-cluster bootstrap draws. Two primary comparisons use 97.5% percentile intervals (Bonferroni).",
        "decision": "All three seed deltas and the upper interval bound must be negative. Secondary metrics are descriptive.",
        "interpretation": "Joint augmentation intervention. Per-frame dropping retains identical token counts. No combined arm.",
    }


def _stages(output, protocol):
    cli = [sys.executable, "-m", "ssl_levjepa"]
    for seed in protocol["seeds"]:
        for arm in ARMS:
            base = output / "arms" / arm / "seeds" / str(seed)
            training = base / "training"
            checkpoint = training / "checkpoints/final.ckpt"
            features = base / "evaluation/features"
            probes = base / "evaluation/probes"
            cohort = protocol["cohort"]
            yield arm, seed, "train", training, cli + [
                "train", "--config", str(base / "config.yaml"), "--output", str(training), "--warmup", "20"
            ]
            verification = base / "checkpoint_verified.json"
            yield arm, seed, "verify", verification, cli + [
                "checkpoint", "verify", str(checkpoint), "--output", str(verification)
            ]
            yield arm, seed, "extract", features, cli + [
                "probe", "extract", "--dataset", protocol["dataset"], "--run", str(training),
                "--cohort", cohort, "--weights", "trained", "--batch-size", str(protocol["extraction_batch_size"]),
                "--output", str(features),
            ]
            yield arm, seed, "probe", probes, cli + [
                "probe", "fit", "--cohort", cohort, "--features", str(features), "--output", str(probes),
            ]


def _prepare(output, protocol):
    from omegaconf import OmegaConf

    baseline = Path(protocol["baseline"])
    expected_upstream = read(baseline / "provenance/source_manifest.json")["upstream_commit"]
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() != expected_upstream:
        raise RuntimeError("Upstream revision differs from control")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Upstream checkout has local changes")
    for file in ("training.py", "probes.py"):
        snap = baseline / "provenance/source/src/ssl_levjepa" / file
        if checksum(Path(__file__).parent / file) != checksum(snap):
            raise RuntimeError(f"{file} differs from the measured control source")
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    source = provenance / "source"
    shutil.copytree(Path(__file__).parent, source / "ssl_levjepa", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "uv.lock", provenance / "uv.lock")
    write_json(provenance / "source_manifest.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_commit": expected_upstream,
        "files": {p.name: checksum(p) for p in (source / "ssl_levjepa").glob("*.py")},
    })
    for seed in protocol["seeds"]:
        original = OmegaConf.load(baseline / "seeds" / str(seed) / "training/config.yaml")
        if original.seed != seed or original.trainer.max_steps != protocol["steps"] or original.data.batch_size != protocol["batch_size"]:
            raise RuntimeError(f"Control seed {seed} has inconsistent settings")
        if not original.masking.iid or original.masking.keep_global != .1 or original.masking.keep_local != .1:
            raise RuntimeError(f"Control seed {seed} does not use the declared uniform mask")
        for arm, overrides in ARMS.items():
            base = output / "arms" / arm / "seeds" / str(seed)
            base.mkdir(parents=True)
            OmegaConf.save(OmegaConf.merge(original, overrides), base / "config.yaml")
    write_json(output / "dataset.json", {
        "id": Path(protocol["dataset"]).name,
        "path": os.path.relpath(protocol["dataset"], output),
        "clips_sha256": protocol["dataset_sha256"],
    })
    write_json(output / "protocol.json", protocol)


def _check_stage(phase, target, seed, protocol):
    baseline = Path(protocol["baseline"]) / "seeds" / str(seed)
    if phase == "train":
        report = read(target / "summary.json")
        if report["initial_step"] != 0 or report["final_step"] != protocol["steps"] or report["nonfinite_count"]:
            raise RuntimeError(f"Training did not complete at the fixed budget: {target}")
        init_path = target / "initial_encoder.pt"
        reference_path = baseline / "training/initial_encoder.pt"
        if checksum(init_path) != checksum(reference_path):
            import torch
            a = torch.load(init_path, map_location="cpu", weights_only=True)
            b = torch.load(reference_path, map_location="cpu", weights_only=True)
            if a.keys() != b.keys() or not all(torch.equal(a[name], b[name]) for name in a):
                raise RuntimeError(f"Initialization differs from paired seed {seed}")
        model = read(target / "model.json")
        reference_model = read(baseline / "training/model.json")
        if model != reference_model:
            raise RuntimeError(f"Model or rounded token budget changed: {target}")
    elif phase == "verify":
        report = read(target)
        if not report["strict_encoder_reload"] or not report["inference_finite"] or report["global_step"] != protocol["steps"]:
            raise RuntimeError(f"Checkpoint verification failed: {target}")
    elif phase == "extract":
        report = read(target / "extraction.json")
        if report["cohort_sha256"] != protocol["cohort_sha256"] or not report["encoder_frozen_verified"]:
            raise RuntimeError(f"Frozen feature integrity check failed: {target}")
    else:
        metrics = read(target / "metrics.json")
        if not set(PROBE_NAMES).issubset(metrics):
            raise RuntimeError(f"Missing required probes: {target}")


def run(baseline, output, plan=False):
    baseline, output = resolve_path(baseline), resolve_path(output)
    protocol = _protocol(baseline)
    stages = list(_stages(output, protocol))
    if plan:
        return {"protocol": protocol, "commands": [command for *_, command in stages]}

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(output, protocol, stages)


def _run_locked(output, protocol, stages):
    from .benchmark import _telemetry

    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    declaration = output / "protocol.json"
    if declaration.exists():
        if read(declaration) != protocol:
            raise RuntimeError("Saved ablation protocol differs")
    else:
        if {path.name for path in output.iterdir()} - {".run.lock", "logs"}:
            raise FileExistsError(f"Choose a new experiment directory: {output}")
        _prepare(output, protocol)
    env = dict(os.environ, SSL_LEVJEPA_ROOT=str(ROOT), PYTHONPATH=str(output / "provenance/source"))
    status_path = output / "status.json"
    status = read(status_path) if status_path.exists() else {"stages": {}}
    for arm, seed, phase, target, command in stages:
        key = f"{arm}_{seed}_{phase}"
        previous = status["stages"].get(key, {})
        if previous.get("state") == "complete":
            continue
        attempt = previous.get("attempt", 0) + 1
        if target.exists():
            failed = output / "interrupted" / f"{key}_{attempt-1}"
            failed.mkdir(parents=True)
            target.rename(failed / target.name)
        log = logs / f"{key}.{attempt}.log"
        record = {"state": "running", "attempt": attempt, "log": str(log),
                  "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
        status.update(state="running", current_stage=key)
        status.pop("error", None)
        status["stages"][key] = record
        write_json(status_path, status)
        stop = threading.Event()
        telemetry = []
        thread = threading.Thread(target=_telemetry, args=(stop, telemetry, log.with_suffix(".gpu.jsonl")), daemon=True)
        thread.start()
        started = time.perf_counter()
        print(f"Starting {key}, attempt {attempt}", flush=True)
        try:
            execute(command, log, timeout=7200, env=env)
            _check_stage(phase, target, seed, protocol)
        except BaseException as error:
            record.update(state="failed", error=str(error))
            status.update(state="failed", error=str(error))
            write_json(status_path, status)
            raise
        finally:
            stop.set()
            thread.join(timeout=10)
        record.update(state="complete", wall_sec=time.perf_counter()-started, gpu_telemetry_samples=len(telemetry))
        write_json(status_path, status)
        if phase == "probe":
            cache = target.parent / "features/patches.npy"
            cache.unlink(missing_ok=True)
            write_json(cache.parent / "cache_policy.json", {"patches": "Regenerate from the saved checkpoint and shared cohort."})
    status.update(state="reporting", current_stage="report")
    write_json(status_path, status)
    try:
        from .ablation_report import write_report
        report = write_report(output)
    except BaseException as error:
        status.update(state="failed", error=str(error))
        write_json(status_path, status)
        raise
    status.update(state="complete", current_stage=None, report=str(report))
    write_json(status_path, status)
    return {"experiment": str(output), "report": str(report)}
