"""Run disposable batch-size measurements through the package training CLI."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from .layout import resolve_path


def _telemetry(stop, rows, path):
    with path.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
            except OSError:
                stop.wait(.5)
                continue
            if result.returncode == 0:
                values = [float(value.strip()) for value in result.stdout.strip().split(",")]
                row = dict(zip(["gpu_util_percent", "temperature_c", "power_w", "device_used_mib",
                                "device_total_mib"], values))
                row["unix_time"] = time.time()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            stop.wait(.5)


def run(config: Path, output: Path, batch_sizes=(8, 16, 32, 64), steps=120,
        warmup=20, log_dir: Path | None = None, dataset: Path | None = None) -> list[dict]:
    """Measure requested batch sizes using isolated package training runs."""
    if not 1 <= warmup < steps:
        raise ValueError("Use at least one warmup step and one measured step")
    if not batch_sizes or any(batch <= 0 for batch in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Batch sizes must be positive and unique")
    config = resolve_path(config)
    output = resolve_path(output)
    logs = resolve_path(log_dir) if log_dir is not None else output
    if logs != output and logs.exists():
        raise FileExistsError(f"Benchmark logs already exist and will be preserved: {logs}")
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=True)
    dataset = resolve_path(dataset) if dataset is not None else None
    results = []
    for batch in batch_sizes:
        directory = output / f"batch_{batch}"
        gpu = []
        stop = threading.Event()
        thread = threading.Thread(target=_telemetry,
                                  args=(stop, gpu, logs / f"gpu_{batch}.jsonl"), daemon=True)
        thread.start()
        command = [sys.executable, "-m", "ssl_levjepa", "train", "--config", str(config),
                   "--output", str(directory), "--batch-size", str(batch), "--steps", str(steps),
                   "--warmup", str(warmup), "--benchmark"]
        if dataset is not None:
            command.extend(["--dataset", str(dataset)])
        log_path = logs / f"batch_{batch}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            thread.join(timeout=10)
        record = {"batch_size": batch, "exit_code": code, "command": command,
                  "warmup_steps": warmup, "measured_steps": max(steps - warmup, 0)}
        if code == 0:
            summary = json.loads((directory / "summary.json").read_text())
            rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            assert len(rows) == steps and summary["nonfinite_count"] == 0
            window = [row for row in gpu if rows[warmup - 1]["unix_time"] <= row["unix_time"] <= rows[-1]["unix_time"]]
            assert window, "No GPU telemetry in timed interval"
            intervals = [rows[index]["wall_elapsed_sec"] - rows[index - 1]["wall_elapsed_sec"]
                         for index in range(warmup, len(rows))]
            record.update({"status": "passed", "samples_per_sec": summary["decoded_clips_per_sec"],
                           "optimizer_steps_per_sec": summary["steps_per_sec"],
                           "mean_step_sec": statistics.mean(intervals),
                           "median_step_sec": statistics.median(intervals),
                           "mean_compute_sec": summary["mean_compute_sec"],
                           "mean_data_wait_sec": summary["mean_data_wait_sec"],
                           "peak_allocated_bytes": summary["peak_allocated_bytes"],
                           "peak_reserved_bytes": summary["peak_reserved_bytes"],
                           "gpu_util_mean_percent": statistics.mean(row["gpu_util_percent"] for row in window),
                           "gpu_temperature_max_c": max(row["temperature_c"] for row in window),
                           "device_peak_used_mib": max(row["device_used_mib"] for row in window),
                           "telemetry_samples": len(window)})
        else:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            record["status"] = ("out_of_memory" if "OutOfMemoryError" in log_text or "out of memory" in log_text else
                                "memory_safety_limit" if "exceeds safety threshold" in log_text else "error")
            incomplete = directory / "summary.json"
            if incomplete.exists():
                state = json.loads(incomplete.read_text())
                record["completed_steps"] = state["completed_steps"]
                record["measured_steps"] = state["measured_steps"]
        results.append(record)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    eligible = [record for record in results if record["status"] == "passed"
                and record["peak_reserved_bytes"] < 11 * 1024**3
                and record["device_peak_used_mib"] < 11264]
    if not eligible:
        raise RuntimeError("No batch passed the measured memory gate")
    best = max(eligible, key=lambda record: record["samples_per_sec"])
    recommendation = {"batch_size": best["batch_size"],
                      "criterion": "highest measured samples/sec with <=11 GiB allocator and device memory gates"}
    (output / "recommendation.json").write_text(json.dumps(recommendation, indent=2) + "\n")
    return results
