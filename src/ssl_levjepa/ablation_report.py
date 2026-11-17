"""Compare two v2 interventions with the paired, previously measured control."""
import math
from pathlib import Path

import numpy as np

from .ablations import ARMS, PROBE_NAMES, read
from .experiment import checksum, write_json
from .probes import REGIMES, load_cohort

LABELS = {"control": "Original v2", "augmentation": "Wider local crops + no flips", "per_frame": "Uniform per frame"}


def _rmse(errors, indices):
    return np.sqrt(errors[:, indices].mean(axis=1))


def paired_comparison(control, treatment, session_indices):
    """Bootstrap complete held-out sessions and preserve pairing across seeds."""
    indices = np.concatenate(session_indices)
    difference = _rmse(treatment, indices) - _rmse(control, indices)
    generator = np.random.default_rng(20261118)
    draws = []
    for _ in range(2000):
        groups = generator.integers(0, len(session_indices), len(session_indices))
        sampled = np.concatenate([session_indices[group] for group in groups])
        draws.append((_rmse(treatment, sampled) - _rmse(control, sampled)).mean())
    interval = np.quantile(draws, [.0125, .9875])
    return {
        "seedwise_delta_mps": difference.tolist(),
        "mean_delta_mps": float(difference.mean()),
        "sd_delta_mps": float(difference.std(ddof=1)),
        "session_bootstrap_interval_97_5": interval.tolist(),
        "consistency_rule_passed": bool((difference < 0).all() and interval[1] < 0),
    }


def _token_coverage():
    frames = 16
    result = {}
    for view, total, keep in (("global", 196, 20), ("local", 36, 4)):
        retained = frames * keep
        p_empty = math.comb((frames - 1) * total, retained) / math.comb(frames * total, retained)
        p_all_covered = sum(
            (-1)**missing * math.comb(frames, missing) * math.comb((frames - missing)*total, retained)
            for missing in range(frames+1)
        ) / math.comb(frames*total, retained)
        result[view] = {
            "retained_per_clip": retained, "clip_iid_probability_any_empty_frame": 1-p_all_covered,
            "clip_iid_expected_empty_frames": frames*p_empty, "per_frame_expected_empty_frames": 0,
        }
    return result


def _crop_visibility(baseline, output, targets, train, seed):
    import torch
    from omegaconf import OmegaConf
    from .layout import enable_upstream

    enable_upstream()
    from data import ViewBuilder

    positions = torch.from_numpy(targets[train, 2:] * 256)
    result = {}
    for name, config in (
        ("control", baseline / "seeds" / str(seed) / "training/config.yaml"),
        ("augmentation", output / "arms/augmentation/seeds" / str(seed) / "config.yaml"),
    ):
        view = ViewBuilder(**OmegaConf.load(config).views)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261118)
            boxes = view.sample_boxes(view.num_local * len(positions), 256, 256, view.local_scale, "cpu")
        boxes = boxes.reshape(view.num_local, len(positions), 4)
        contains = ((positions[None] >= boxes[..., :2]) & (positions[None] < boxes[..., 2:])).all(-1)
        result[name] = {"last_frame_ball_center_inside_local_crop": float(contains.float().mean()),
                        "opposite_horizontal_orientation_probability": 2 * view.flip_p * (1 - view.flip_p)}
    return result


def _stats(values, decimals=4):
    return f"{np.mean(values):.{decimals}f} ± {np.std(values, ddof=1):.{decimals}f}"


def write_report(output):
    """Summarize all completed runs, without fitting or selecting anything on test."""
    output = Path(output)
    protocol = read(output / "protocol.json")
    baseline = Path(protocol["baseline"])
    seeds = protocol["seeds"]
    folder, clips, targets, masks = load_cohort(Path(protocol["cohort"]))
    if checksum(folder / "clips.jsonl") != protocol["cohort_sha256"]:
        raise RuntimeError("Probe cohort changed between training and reporting")
    session_ids = np.array([clip["match_id"] for clip in clips])
    test_sessions = sorted(set(session_ids[masks["test"]]))
    sessions = [np.flatnonzero(masks["test"] & (session_ids == session)) for session in test_sessions]
    test = np.flatnonzero(masks["test"])
    speed = np.linalg.norm(targets[:, :2], axis=1)
    groups = {name: np.flatnonzero(masks["test"] & (speed >= lower) & (speed < upper))
              for name, (lower, upper) in REGIMES.items()}
    results, diagnostics, training, errors = {}, {}, {}, {}
    for arm in LABELS:
        results[arm], diagnostics[arm], training[arm] = [], [], []
        primary = []
        for seed in seeds:
            base = baseline / "seeds" / str(seed) if arm == "control" else output / "arms" / arm / "seeds" / str(seed)
            evaluation = base / ("evaluation/trained" if arm == "control" else "evaluation")
            probe_dir = evaluation / "probes"
            metrics = read(probe_dir / "metrics.json")
            evidence = read(evaluation / "features/extraction.json")
            if evidence["cohort_sha256"] != protocol["cohort_sha256"] or not evidence["encoder_frozen_verified"]:
                raise RuntimeError(f"Frozen encoder or cohort check failed in {evaluation}")
            checkpoint = read(base / "checkpoint_verified.json")
            if (evidence["source_sha256"] != checkpoint["sha256"]
                    or checksum(base / "training/checkpoints/final.ckpt") != checkpoint["sha256"]):
                raise RuntimeError(f"Checkpoint digest mismatch in {base}")
            for probe in PROBE_NAMES:
                with np.load(probe_dir / probe / "predictions.npz", allow_pickle=False) as archive:
                    prediction = archive["prediction"]
                if prediction.shape != targets.shape or not np.isfinite(prediction).all():
                    raise RuntimeError(f"Invalid {probe} predictions in {probe_dir}")
                difference = prediction - targets
                scores = metrics[probe]["test"]["aggregate"]
                rmse = np.sqrt(np.mean(difference[test, :2]**2))
                position = 256 * np.sqrt(np.mean(difference[test, 2:]**2))
                if not np.isclose(rmse, scores["velocity_rmse_mps"]) or not np.isclose(position, scores["localization_rmse_px"]):
                    raise RuntimeError(f"Metrics differ from {probe} predictions in {probe_dir}")
                if probe == "spatial_conv":
                    primary.append((difference[:, :2]**2).mean(axis=1, dtype=np.float64))
            results[arm].append(metrics)
            diagnostics[arm].append(evidence["readout_train"])
            training[arm].append(read(base / "training/summary.json"))
        errors[arm] = np.array(primary)

    comparisons = {arm: paired_comparison(errors["control"], errors[arm], sessions) for arm in ARMS}
    coverage = _token_coverage()
    visibility = _crop_visibility(baseline, output, targets, masks["train"], seeds[0])
    report_dir = output / "reports"
    report_dir.mkdir(exist_ok=True)
    evidence = {
        "seeds": seeds, "test_sessions": test_sessions, "paired_comparisons": comparisons,
        "readout_diagnostics": diagnostics, "training": training, "token_coverage": coverage,
        "crop_visibility": visibility,
        "cohort_sha256": protocol["cohort_sha256"], "predictions_checked": len(LABELS)*len(seeds)*len(PROBE_NAMES),
    }
    write_json(report_dir / "metrics.json", evidence)
    successful = [LABELS[name] for name, comparison in comparisons.items() if comparison["consistency_rule_passed"]]
    if successful:
        conclusion = f"The matched comparison supports {', '.join(successful)} under the declared rule."
    else:
        conclusion = "Neither intervention met the declared consistency rule on velocity. Keep the current model and dataset sizes for follow-up objective experiments."
    lines = [
        "# Augmentation and temporal coverage ablations",
        "",
        conclusion,
        "",
        f"Each arm used seeds {', '.join(map(str, seeds))}, {protocol['steps']:,} steps, batch {protocol['batch_size']}, "
        "the same ViT-Tiny and the original 10,240 training clips. The 1,280-clip validation and test sets were reused "
        "with identical labels and session assignments. This is a follow-up on the same test cohort used in v2.",
        "",
        "| Arm | Change from original v2 |",
        "|---|---|",
        "| Original v2 | Previously measured trained checkpoints used as the paired control |",
        "| Wider local crops + no flips | Local crop area range 0.02–0.40 → 0.80–1.00 and horizontal flip probability 0.50 → 0 |",
        "| Uniform per frame | Uniform clip-level sampling → uniform sampling with a fixed count per frame |",
        "",
        "The augmentation arm tests the two changes jointly. Crop dimensions, view count, photometric jitter, "
        "loss and optimizer settings follow v2. The coverage arm changes the training mask only. Both mask policies "
        "retain 320 global tokens and 64 tokens per local view in each 16-frame clip. Inference uses dense frozen features.",
        "",
        f"Training-label diagnostics put the last-frame ball center inside "
        f"{visibility['control']['last_frame_ball_center_inside_local_crop']:.1%} of original local crops and "
        f"{visibility['augmentation']['last_frame_ball_center_inside_local_crop']:.1%} of wider crops. "
        "The augmentation intervention also removes the 50% chance of opposing horizontal orientations between global and local views.",
        "",
        f"With clip-level sampling, the analytical probability of at least one empty frame in a view is "
        f"{coverage['global']['clip_iid_probability_any_empty_frame']:.3g} globally and "
        f"{coverage['local']['clip_iid_probability_any_empty_frame']:.1%} locally. "
        "Fixed per-frame counts guarantee a sampled patch in each frame. Coverage refers to sampled patches anywhere in the view.",
        "",
        "## Primary velocity comparison",
        "",
        "Component-wise test RMSE is measured in m/s with the same spatial convolutional probe. "
        "Deltas subtract the paired original-v2 score. Intervals resample complete sessions with the three seeds kept paired. "
        "The 97.5% intervals account for two comparisons using Bonferroni. The declared rule requires all seed differences "
        "and the interval’s upper bound to be below zero.",
        "",
        "| Arm | Test RMSE, mean ± SD | Seed deltas | Mean delta | 97.5% interval |",
        "|---|---:|---|---:|---|",
        f"| Original v2 | {_stats(_rmse(errors['control'], test))} | — | — | — |",
    ]
    for name, comparison in comparisons.items():
        deltas = ", ".join(f"{x:+.4f}" for x in comparison["seedwise_delta_mps"])
        low, high = comparison["session_bootstrap_interval_97_5"]
        lines.append(f"| {LABELS[name]} | {_stats(_rmse(errors[name], test))} | {deltas} | {comparison['mean_delta_mps']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += ["", "## Motion regimes and frozen probes", "",
              "| Test regime | Clips | Original v2 | Wider crops + no flips | Uniform per frame |",
              "|---|---:|---:|---:|---:|"]
    for name, indices in groups.items():
        scores = [_stats(_rmse(values, indices)) if len(indices) else "—" for values in errors.values()]
        lines.append(f"| {name} | {len(indices)} | {' | '.join(scores)} |")
    lines += ["", "All probe rows report test velocity RMSE in m/s and localization RMSE in 256-pixel image coordinates. "
              "Means and sample standard deviations are across the three seeds. The training-only normalization and validation selection rules match v2.",
              "", "| Probe | Arm | Velocity | Localization |", "|---|---|---:|---:|"]
    for probe in PROBE_NAMES:
        for arm, runs in results.items():
            scores = [run[probe]["test"]["aggregate"] for run in runs]
            velocity = _stats([score["velocity_rmse_mps"] for score in scores])
            position = _stats([score["localization_rmse_px"] for score in scores], 2)
            lines.append(f"| {probe} | {LABELS[arm]} | {velocity} | {position} |")
    lines += ["", "## Readout and compute", "",
              "| Arm | Train readout channel SD | Effective rank |",
              "|---|---:|---:|"]
    for arm, stats in diagnostics.items():
        lines.append(f"| {LABELS[arm]} | {_stats([s['mean_dimension_std'] for s in stats], 6)} | {_stats([s['effective_rank'] for s in stats], 2)} |")
    lines += ["", "| Arm | Seed | Clips/s | GPU-stage wall hours | Peak reserved GiB |",
              "|---|---:|---:|---:|---:|"]
    for arm, summaries in training.items():
        for seed, summary in zip(seeds, summaries):
            lines.append(f"| {LABELS[arm]} | {seed} | {summary['decoded_clips_per_sec']:.2f} | {summary['gpu_hours']:.4f} | {summary['peak_reserved_bytes']/1024**3:.3f} |")
    hours = sum(summary["gpu_hours"] for arm in ARMS for summary in training[arm])
    lines += ["", f"The six new trainings used {hours:.4f} GPU-stage wall hours. Original-v2 throughputs are historical measurements.",
              "", "The comparison uses eight held-out sessions and three paired training seeds. Wider crops also shrink objects "
              "at fixed output resolution. A favorable joint augmentation result would need a separate crop-versus-flip comparison "
              "to identify which change matters. The experiment estimates the two main interventions separately and has no interaction arm.",
              "", "The experiment preserves the exact protocol, training configs, checkpoints, validation selections, predictions, logs and source hashes. "
              "The original v2 folder retains the paired random and supervised controls.", ""]
    report = report_dir / "ABLATION_REPORT.md"
    temporary = report.with_suffix(".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report)
    return report
