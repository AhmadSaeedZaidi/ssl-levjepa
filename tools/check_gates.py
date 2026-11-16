"""Accept only measured, finite, resumed validation runs before unattended training."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix")
    args = ap.parse_args()
    summaries = []
    for suffix, initial, final in [("one", 0, 1), ("smoke50", 1, 51), ("profile", 51, 276)]:
        path = Path(f"{args.prefix}_{suffix}")
        summary = json.loads((path / "summary.json").read_text())
        evidence = json.loads((path / "resume_evidence.json").read_text())
        assert summary["status"] == "reached_steps" and summary["final_step"] == final
        assert summary["initial_step"] == initial and summary["completed_steps"] == final-initial
        assert evidence["global_step"] == initial
        assert evidence["optimizers"][0]["min_step"] == initial
        assert evidence["optimizers"][0]["max_step"] == initial
        assert summary["nonfinite_count"] == 0 and summary["peak_reserved_bytes"] < 11*1024**3
        assert (path / "checkpoints/final.ckpt").is_file()
        assert (path / "exit_code.txt").read_text().strip() == "0"
        summaries.append(summary)
    profile = summaries[-1]
    assert profile["measured_steps"] == 200
    # A slow input pipeline requires investigation before an unattended run.
    assert profile["mean_data_wait_sec"] < profile["mean_compute_sec"], "Input wait exceeds training compute"
    validation_losses = []
    for metric_file in Path(f"{args.prefix}_profile").glob("csv/version_*/metrics.csv"):
        with metric_file.open() as stream:
            for row in csv.DictReader(stream):
                if row.get("val/loss"):
                    validation_losses.append(float(row["val/loss"]))
    assert validation_losses and all(math.isfinite(v) for v in validation_losses), "No finite held-out validation loss"
    profile["heldout_losses"] = validation_losses
    Path("logs/benchmark.json").write_text(json.dumps(profile, indent=2)+"\n")
    gate = {"passed": True, "prefix": args.prefix,
            "checkpoint": f"{args.prefix}_profile/checkpoints/final.ckpt",
            "profile_steps": 200, "estimated_remaining_sec": (1200-276)/profile["steps_per_sec"]}
    Path("logs/validation_passed.json").write_text(json.dumps(gate, indent=2)+"\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
