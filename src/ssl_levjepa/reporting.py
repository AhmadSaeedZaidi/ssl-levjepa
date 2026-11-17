"""Write local dataset and baseline reports from measured artifacts."""
from datetime import datetime
import json
from pathlib import Path

from .layout import ROOT, resolve_path


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_first(paths, default=None):
    for path in paths:
        if path.exists():
            return read(path, default)
    return default


def write(dataset: Path, experiment: Path, output: Path | None = None) -> tuple[Path, Path]:
    """Write measured dataset and experiment summaries under reports/."""
    dataset=resolve_path(dataset)
    experiment=resolve_path(experiment)
    output=resolve_path(output) if output is not None else experiment/"reports"
    dataset=dataset.resolve(); experiment=experiment.resolve()
    report_paths=(output/"DATASET_REPORT.md", output/"BASELINE_REPORT.md")
    existing=[str(path) for path in report_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing reports: "+", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    meta=read(dataset/"metadata.json"); audit=read(dataset/"audit.json")
    environment=read_first([experiment/"provenance/preflight_train.json", dataset/"train/preflight.json"],{})
    stats=meta["stats"]; repeat=audit["rollout_repeat"]
    checkpoint_hash=audit.get("original_toy_checkpoint_sha256")
    checkpoint_statement=(
        f"The archived infrastructure-only toy checkpoint is unchanged: `{checkpoint_hash}`. Toy training was not continued."
        if checkpoint_hash else "No archived toy checkpoint is available in this checkout; preservation status is unknown."
    )
    splits="\n".join(f"| {name} | {len(meta['split_sessions'][name])} | {counts['episodes']} | {counts['frames']} | {counts['clips']} |" for name,counts in meta["split_counts"].items())
    data_report=f"""# SSL simulator dataset pilot

Generated report: {datetime.now().astimezone().isoformat()}

Status: actual Sumatra native AI-vs-AI simulator recordings, with idealized physics. This replaces hand-written toy trajectories. It is a controlled simulation pilot, not real camera footage or a validated model of real robots.

## Integration choice and provenance

Sumatra [release 2025](https://github.com/TIGERs-Mannheim/Sumatra/releases/tag/2025), source commit `{meta['sumatra_source_sha']}`.
Public release ZIP SHA256: `{meta['sumatra_release_zip_sha256']}`. Java: `{meta['java_version'].splitlines()[0]}`.
Recorder SHA256: `{meta['recorder_sha256']}`; runtime module config SHA256: `{meta['runtime_moduli_sha256']}`.
Exact sources and inspected alternatives: `simulation/sources.json`. Each original trajectory's SHA256 is in `metadata.json`.

The adapter loads the released application JARs, runs both built-in team AIs, initializes seeded snapshots, and calls the existing blocking simulator step. It reads the pinned private `simState` field only while the simulator is paused. No robot or ball dynamics are implemented by this project. No upstream LeVJEPA or Sumatra source files were patched.

Sumatra's native simulator uses idealized robot trajectory following, ball rolling/flying dynamics and collision handling. It is not Bullet/ODE and does not establish hardware realism. See the [upstream simulation documentation](https://github.com/TIGERs-Mannheim/Sumatra/tree/2025) and [TIGERs description of the native simulator](https://ssl.robocup.org/wp-content/uploads/2022/04/2022_ETDP_TIGERs-Mannheim.pdf).

ER-Force's [standalone simulator](https://github.com/robotics-erlangen/framework#simulator-cli) and grSim were inspected. ER-Force current source needs Qt6 development libraries, absent locally. Its official public image was fetched by digest and its CLI tested without a system-wide install. That image was built in 2022, and its source revision is not identified by its labels. Its public interface exposes vision detections, rather than the complete ground-truth state required here. The image is not asserted to match the inspected current source and was not used for this dataset. grSim requires Qt5/ODE and similarly adds a separate control/recording integration. Sumatra was the smallest tested path combining native state access and AI-vs-AI behavior.

## Recorded data and splits

72 episodes in 12 independently started JVM sessions; 6 controlled starts per session, 6 blue and 6 yellow robots on the native 9 x 6 m field. Session is the split unit. These are short FORCE_START scenarios, not complete refereed matches. Initial positions, ball velocities and scenario families vary. All subsequent trajectories and applied commands come from Sumatra.

| Split | Sessions | Episodes | RGB frames | Non-overlapping 16-frame clips |
|---|---:|---:|---:|---:|
{splits}

Full recording: {audit['physics_ticks']} ticks at 100 Hz; native physics substeps are 1 ms. Simulation timestamps are nanoseconds within each episode and reset with the unique episode ID. Video samples use the nearest 100 Hz state to nominal 15 FPS: actual intervals are 60 or 70 ms, retained explicitly. Each episode has 128 RGB frames spanning 8.47 seconds.

State rows contain session/episode IDs, initialization seed, tick/frame indices, timestamps, robot IDs/team/poses/velocities/contact flags, ball XYZ position/velocity/acceleration, referee command and selected applied robot commands. `ball_contact` is Sumatra's barrier-interrupted/kicker-zone proximity flag, not an exhaustive collision-solver contact event. Robot and ball state positions/linear velocities are normalized to m and m/s; orientation/angular velocity use rad and rad/s. Raw action target positions and local target velocities retain upstream mm and mm/s (field names include units); kick speed is m/s. Actions are aligned to the preceding 10 ms interval. Wheel commands and movement-constraint objects are not serialized; this is not yet a complete action-replay system.

Observed in the 15 FPS samples: {stats['contact_frames']} frames with the native barrier/proximity flag, {stats['kick_request_frames']} frames with positive kick commands, {stats['turning_episodes']} episodes with a moving-ball direction change >0.2 rad, maximum ball height {stats['max_ball_z_m']:.4f} m. These are descriptive coverage checks, not match performance measures.

## Rendering and validation

CPU Pillow rendering at 3x supersampling to 256 x 256 RGB, then JPEG quality 95. Rendered frames depend only on the recorded state. Colors identify the two teams and ball; robot pose determines heading. The ball has a documented minimum marker radius of 1.5 px to remain visible; field-line widths and a simple height cue are graphical conventions, not calibrated camera optics. No state vectors, IDs or target velocities are supplied as encoder input channels.

Lance columns: `episode_idx`, `step_idx`, `timestamp_ns`, `frame`, `h`, `w`, `label=-1`. Ground-truth evaluation labels remain in the separate clip manifest/raw recordings. Episode index files are built by upstream LeVJEPA's builder.

Passed: every 10 ms tick interval; finite recorded states; 12 robots/sample; exactly 128 samples/episode; {meta['render_and_label_alignment_checks']}; all {audit['clip_labels_checked']} clip-target timestamps and ball-velocity labels checked against source states; no temporal boundary crossing; zero cross-split episode/session overlap and zero identical RGB clips crossing splits. Actual upstream dataloader/view checks are saved in experiment `provenance/preflight_*.json`; older evidence remains in each split's `preflight.json`.

Preview: `{dataset.relative_to(ROOT)}/previews/preview.png`; animated 16-frame example: `{dataset.relative_to(ROOT)}/previews/clip16.gif`.

## Reproducibility limits

The repeat test compared {repeat['compared_ticks']} ticks with the same initialized seed. Identical state/action record values: **{repeat['identical_record_values']}**. First different tick: {repeat['first_different_tick']}; maximum subsequent ball-position divergence: {repeat['max_ball_position_difference_m']:.4f} m. The initialization seed does not control all native AI randomness/timing or game-controller state. Do not claim bitwise rollout reproducibility.

The fixed archived recordings, hashes, split manifest and deterministic renderer make this dataset and training input reproducible. Regenerating a new AI rollout from seed alone remains an integration limitation. The early draft with an incorrectly named action-velocity unit was stopped, kept as debug output only, and excluded entirely from this pilot.

## Reproduction commands and next experiment

The original command history is retained in `provenance/legacy_reports/`; the commands below describe the current CLI and are not presented as historical test records. Audit the existing dataset and use `--plan` to inspect a fresh run without creating output:

```bash
.venv/bin/ssl-levjepa dataset audit {dataset.relative_to(ROOT)}
.venv/bin/ssl-levjepa run --dataset {dataset.name} --output outputs/reproduction_plan --plan
```

{checkpoint_statement}

Next dataset action: control Sumatra AI/game-controller randomness and state, then collect a larger cohort of independent matches with fixed held-out sessions. For stronger physical fidelity, build a pinned modern ER-Force simulator and export its internal `world::SimulatorState`, while retaining Sumatra as the AI. Never replace missing simulator state with invented trajectories.
"""
    (output/"DATASET_REPORT.md").write_text(data_report, encoding="utf-8")
    bench=read(experiment/"benchmark/results.json",[])
    table=[]
    for r in bench:
        if r["status"]=="passed":
            table.append(f"| {r['batch_size']} | {r['samples_per_sec']:.2f} | {r['mean_step_sec']*1000:.2f} | {r['peak_allocated_bytes']/1024**3:.3f} / {r['peak_reserved_bytes']/1024**3:.3f} | {r['gpu_util_mean_percent']:.1f} | {r['device_peak_used_mib']:.0f} |")
        else:
            table.append(f"| {r['batch_size']} | {r['status']} | — | — | — | — |")
    summary=read(experiment/"training/summary.json")
    reload_path=experiment/"provenance/checkpoint_verified.json"
    if not reload_path.exists():
        reload_path=experiment/"training/checkpoints/final.verified.json"
    reload=read(reload_path)
    evaluation=read(experiment/"evaluation/metrics.json")
    training_text=(f"Completed {summary['completed_steps']} optimizer steps from step {summary['initial_step']} to {summary['final_step']}, "
        f"with effective batch {summary['samples_seen']//summary['completed_steps']} and {summary['samples_seen']} clip presentations. "
        f"Exit reason: {summary['status']}. Mean/final training loss: {summary['mean_loss']:.6f} / {summary['final_loss']:.6f}; "
        f"non-finite losses: {summary['nonfinite_count']}. Training wall time including worker startup: {summary['wall_sec']:.2f} s "
        f"(~{summary['gpu_hours']:.4f} active training GPU-hours). After {summary['warmup_steps_excluded']} warmup steps, "
        f"{summary['measured_steps']} steps measured {summary['decoded_clips_per_sec']:.2f} clips/sec, "
        f"{summary['mean_compute_sec']*1000:.2f} ms compute and {summary['mean_data_wait_sec']*1000:.2f} ms blocking input wait per step. "
        f"Peak reserved memory: {summary['peak_reserved_bytes']/1024**3:.3f} GiB. Exact values: `training/summary.json`."
        if summary else "Pending; no completed training result is claimed.")
    training_context=(
        "Fresh initialization, seed 42; no continuation of the infrastructure-only toy checkpoint. "
        f"Upstream `{meta.get('levjepa_sha','a748e0fbaa0ec548cad283978b818779708a3bd7')}` unchanged. Encoder: 192 width, 12 blocks, 3 heads, patch 16. "
        "One global and four local views; uniform token retention (20 global and 4 local tokens/frame-equivalent); no motion-aware token selection, compile or optional evaluation EMA. "
        "Unmodified invariance + 0.06*SIGReg, upstream AdamW lr 4e-4/betas (0.9,0.95)/weight decay .04 and norm clipping 1. WSD warmup 30, stable until step 200, decay over final 100 of the 300-step pilot. "
        "Training sees only the 48 training episodes; 12 validation episodes are in a separate store. Test episodes never enter pretraining or hyperparameter selection."
        if summary else "No completed baseline training summary is present; training measurements and completed-run claims are not available."
    )
    reload_text=(f"Passed strict encoder reload and finite {reload['inference_shape']} inference. "
        f"Global step, all {reload['optimizer_state_entries']} AdamW state entries and scheduler are at step {reload['global_step']}. "
        f"Checkpoint SHA256: `{reload['sha256']}`. Evidence: `{reload_path.relative_to(experiment)}`."
        if reload else "Pending; no checkpoint reload is claimed.")
    results="Pending. No evaluation result is claimed."
    if evaluation:
        evaluation_table="\n".join(f"| {name} | {v['validation']['rmse_mps']:.5f} | {v['test']['rmse_mps']:.5f} | {v['test']['mae_mps']:.5f} |" for name,v in evaluation["results"].items())
        results=f"""Feature extraction: {evaluation['clips']}; excluded: {evaluation['excluded']}.
All 196 patch tokens/frame, deterministic 224px resize and ImageNet normalization, encoder eval mode, bf16, frozen 192-dimensional readout. Labels are simulator vx/vy at the last input frame. StandardScaler and Ridge fit training clips only; alpha in [0.1,1,10,100,1000] selected by validation RMSE; the test split is evaluated after selection. Neither state nor action labels enter pretraining.

| Frozen representation / training | Validation RMSE m/s | Test RMSE m/s | Test MAE m/s |
|---|---:|---:|---:|
{evaluation_table}

Per-session test metrics, alpha search and target distributions are in `{experiment.relative_to(ROOT)}/evaluation/metrics.json`. The random model uses the exact saved pretraining initialization and identical clips/preprocessing. The constant predictor uses the training-label mean. There are only two test sessions and one training seed; these point estimates do not establish a statistically reliable research improvement.
"""
        paired={s:evaluation["results"]["trained"]["test_per_session"][s]["rmse_mps"]-
                  evaluation["results"]["random_initialization"]["test_per_session"][s]["rmse_mps"]
                  for s in evaluation["results"]["trained"]["test_per_session"]}
        results+=f"\nTrained-minus-random RMSE by held-out session (negative is better): `{json.dumps(paired)}`. The average gain is not consistent across both sessions.\n"
    benchmark_context=(
        "Fresh disposable runs on simulator RGB only; no toy checkpoint or toy data used. Same ViT-Tiny architecture and unmodified upstream loss/optimizer, bf16, uniform 10% IID retention, 16 frames, 224px global + four 96px local views. Each size: 20 warmup optimizer steps + 100 timed optimizer steps, CUDA synchronization at boundaries. Samples/sec measures decoded training clip presentations end to end, including input wait/transfer and logging. Peak allocator memory includes warmup; GPU utilization/device memory sampled every 0.5s and filtered to the timed interval. Device memory/utilization also include the Windows desktop."
        if any(r.get("status")=="passed" for r in bench)
        else "No completed benchmark measurements are present in this experiment. No performance measurements are claimed."
    )
    training_report=f"""# LeVJEPA simulator pilot baseline

Report generated {datetime.now().astimezone().isoformat()}. Experiment: `{experiment.relative_to(ROOT)}`.

## Batch-size benchmark

Measured software/device: {environment.get('device')}, PyTorch {environment.get('torch')}, CUDA wheel runtime {environment.get('torch_cuda')}, driver {environment.get('driver')}, bf16 supported {environment.get('bf16_supported')}. Dependencies are pinned in `uv.lock`; per-run environment files retain the actual installed versions.

{benchmark_context}

| Micro-batch | Samples/sec | Mean step ms | Allocated / reserved GiB | GPU util % | Device peak MiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table) if table else '| pending | — | — | — | — | — |'}

Per-step, GPU and failure logs: `{experiment.relative_to(ROOT)}/logs/benchmark/`; aggregate results remain in `benchmark/`. Recommendation uses highest measured throughput while leaving the measured memory safety margin; it is saved separately.
Upstream SIGReg's default statistic scales with batch size, so loss magnitudes across these micro-batches are not accuracy comparisons.

## Baseline training and compatibility

{training_context}

All effective settings: `training/config.yaml`. {training_text}
The cached validation loss uses batches of four clips. Because upstream SIGReg scales with batch size, raw training and validation loss magnitudes are not directly comparable; the frozen velocity evaluation is the downstream comparison.
Checkpoint strict reload/inference and optimizer/scheduler state verification: {reload_text}
Periodic full-state checkpoints every 100 optimizer steps, with saved random encoder initialization for the control. Native upstream checkpoint compatibility is retained. The upstream sampler still has no mid-epoch resume cursor; a resumed run restores weights/optimizer/step, but not exact sample order.

## Frozen-feature ball-motion task

{results}

## Commands, limitations and next experiment

These current CLI examples are not historical test records; archived commands remain in `provenance/legacy_reports/`. The planned run is read-only. For evaluation and report reruns, use fresh output directories; both commands refuse existing output:

```bash
.venv/bin/ssl-levjepa run --dataset {dataset.name} --output outputs/reproduction_plan --plan
.venv/bin/ssl-levjepa evaluate --dataset {dataset.name} --run {experiment.relative_to(ROOT)}/training --output {experiment.relative_to(ROOT)}/evaluation-rerun
.venv/bin/ssl-levjepa report --dataset {dataset.name} --run {experiment.relative_to(ROOT)} --output {experiment.relative_to(ROOT)}/reports-rerun
```

The small idealized native-simulation pilot validates the data-to-representation-to-evaluation experiment path. It cannot establish real-world transfer, physical-model accuracy or motion-guided masking novelty. The measured AI rollout repeatability limitation is documented in DATASET_REPORT.md. Exact fixed dataset hashes, clips, features, predictions and split masks are retained locally. The original toy logs/checkpoints are preserved.

Next experiment: enlarge the independent-session dataset and repeat this frozen evaluation with three fresh training seeds, holding the test sessions fixed. Require a consistent advantage over the random-encoder and constant controls before attributing useful motion representations to this short training. Keep uniform retention until that training is established.
"""
    baseline_path=output/"BASELINE_REPORT.md"
    baseline_path.write_text(training_report, encoding="utf-8")
    return output/"DATASET_REPORT.md", baseline_path


def write_v2(experiment: Path):
    """Report measured expanded baselines and paired session-level uncertainty."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .experiment import checksum, write_json
    from .probes import REGIMES, load_cohort

    experiment = resolve_path(experiment)
    destination = experiment/"reports/BASELINE_V2_REPORT.md"
    if destination.exists():
        raise FileExistsError(destination)
    declaration = read(experiment/"protocol.json")
    seeds = declaration["seeds"]
    dataset = resolve_path(declaration["dataset"])
    meta,audit = read(dataset/"metadata.json"),read(dataset/"audit.json")
    diversity = read(experiment/"diagnostics/dataset_diversity.json")
    folder,clips,target,masks = load_cohort(experiment/"evaluation/cohort")
    cohort = read(folder/"protocol.json")
    crop = read(experiment/"diagnostics/crop_visibility.json")
    temporal = read(experiment/"diagnostics/temporal.json")
    resolution = read(experiment/"diagnostics/resolution.json")
    source = read(experiment/"provenance/source_manifest.json")
    detector = read(folder/"visual_detector_metrics.json")
    probes = ["global_readout_ridge","mean_patch_pool_ridge","spatial_patch_ridge","spatial_pointwise","spatial_conv"]
    arms = ("random","trained","supervised")
    models,features,training_rows = {},{},[]
    gpu_hours = training_hours = 0.0
    for seed in seeds:
        base = experiment/"seeds"/str(seed)
        summary,verified = read(base/"training/summary.json"),read(base/"checkpoint_verified.json")
        assert summary["final_step"]==declaration["steps"]==verified["global_step"]
        assert verified["sha256"]==checksum(base/"training/checkpoints/final.ckpt")
        training_hours += summary["gpu_hours"]; gpu_hours += summary["gpu_hours"]
        training_rows.append(f"| {seed} | {summary['completed_steps']} | {summary['samples_seen']} | {summary['decoded_clips_per_sec']:.2f} | {summary['wall_sec']/60:.2f} | {summary['gpu_hours']:.4f} | {summary['peak_reserved_bytes']/1024**3:.3f} |")
        models[seed],features[seed] = {},{}
        for arm in arms:
            run = base/"evaluation"/arm
            models[seed][arm] = read(run/"probes/metrics.json")
            evidence = read(run/"features/extraction.json"); features[seed][arm] = evidence
            assert evidence["cohort_sha256"]==cohort["cohort_sha256"]
            gpu_hours += evidence.get("gpu_hours",evidence["wall_sec"]/3600)
            for name in ("spatial_pointwise","spatial_conv"):
                fit = read(run/"probes"/name/"fit.json")
                assert not fit["test_used_in_selection"]
                gpu_hours += fit["gpu_hours"]
            if arm=="supervised":
                gpu_hours += read(run/"training/fit.json")["gpu_hours"]
    gpu_hours += sum(v["extraction"]["gpu_hours"] for v in temporal.values())

    def mean_sd(values, digits=4):
        return f"{np.mean(values):.{digits}f} ± {np.std(values,ddof=1):.{digits}f}"

    def values(arm,probe,field="velocity_rmse_mps",split="test",regime=None):
        return [models[s][arm][probe][split]["motion"][regime][field] if regime
                else models[s][arm][probe][split]["aggregate"][field] for s in seeds]

    comparison_rows = [f"| {probe} | {arm} | {mean_sd(values(arm,probe))} | {mean_sd(values(arm,probe,'localization_rmse_px'),2)} | {mean_sd(values(arm,probe,'within_8px'),3)} |"
                       for probe in probes for arm in arms]
    speed = np.linalg.norm(target[:,:2],axis=1)
    distribution_rows = []
    for split,m in masks.items():
        counts = [int((m&(speed>=lo)&(speed<hi)).sum()) for lo,hi in REGIMES.values()]
        distribution_rows.append(f"| {split} | {m.sum()} | {' | '.join(str(n) for n in counts)} | {np.median(speed[m]):.3f} | {np.quantile(speed[m],.95):.3f} |")
    regime_rows = []
    for regime in REGIMES:
        n = detector["test"]["motion"][regime]["n"]
        entries = [mean_sd(values(arm,"spatial_conv",regime=regime)) for arm in arms] if n else ["unavailable"]*3
        rgb = f"{detector['test']['motion'][regime]['velocity_rmse_mps']:.4f}" if n else "unavailable"
        regime_rows.append(f"| {regime} | {n} | {rgb} | {' | '.join(entries)} |")

    # Bootstrap whole test sessions, preserving every seed's paired random control.
    test_ids = np.flatnonzero(masks["test"])
    sessions = sorted({clips[i]["match_id"] for i in test_ids})
    errors = {}
    for arm in ("random","trained"):
        predictions = np.stack([np.load(experiment/"seeds"/str(s)/"evaluation"/arm/"probes/spatial_conv/predictions.npz")["prediction"][:,:2] for s in seeds])
        errors[arm] = ((predictions-target[None,:,:2])**2).mean(2)
    rng = np.random.default_rng(20261117)
    bootstrap = []
    for _ in range(2000):
        chosen = rng.choice(sessions,len(sessions),replace=True)
        ids = np.concatenate([test_ids[[clips[i]["match_id"]==name for i in test_ids]] for name in chosen])
        bootstrap.append(float(np.sqrt(errors["trained"][:,ids].mean(1)).mean()-np.sqrt(errors["random"][:,ids].mean(1)).mean()))
    paired = np.array(values("trained","spatial_conv"))-np.array(values("random","spatial_conv"))
    interval = np.quantile(bootstrap,[.025,.975])
    consistent = bool((paired<0).all() and interval[1]<0)
    uncertainty = {"primary":"spatial_conv","seedwise_trained_minus_random_rmse_mps":paired.tolist(),
                   "mean_difference":float(paired.mean()),"session_bootstrap_95_percentile_interval":interval.tolist(),
                   "replicates":2000,"sessions":sessions,"consistent_under_declared_rule":consistent,
                   "rule":"All paired seed deltas negative and session-bootstrap 95% interval below zero; descriptive, not a multiplicity-adjusted causal claim."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination.parent/"paired_uncertainty.json",uncertainty)
    session_rows = []
    for name in sessions:
        difference = [models[s]["trained"]["spatial_conv"]["test"]["sessions"][name]["velocity_rmse_mps"]-models[s]["random"]["spatial_conv"]["test"]["sessions"][name]["velocity_rmse_mps"] for s in seeds]
        session_rows.append(f"| {name} | {mean_sd(difference)} |")
    collapse_rows = []
    for seed in seeds:
        for arm in arms:
            v = features[seed][arm]["readout_train"]
            collapse_rows.append(f"| {seed} | {arm} | {v['mean_dimension_std']:.6g} | {v['effective_rank']:.2f} | {v['participation_rank']:.2f} | {v['mean_paired_cosine']:.8f} |")
    temporal_rows = [f"| {arm} | {probe} | {models[seeds[0]][arm][probe]['test']['aggregate']['velocity_rmse_mps']:.4f} | {temporal[arm]['metrics'][probe]['test']['aggregate']['velocity_rmse_mps']:.4f} |"
                     for arm in ("random","trained") for probe in ("global_readout_ridge","spatial_conv")]
    resolution_rows = [f"| {size} | {v['both_frames_detected_rate']:.3f} | {v['metrics']['test']['aggregate']['velocity_rmse_mps']:.4f} | {v['metrics']['test']['aggregate']['localization_rmse_px']:.3f} |" for size,v in resolution.items()]
    scenario_rows = [f"| {name} | {v['episodes']} | {v['turning_episodes']} | {v['native_kick_timestamps']} | {v['barrier_contact_frames']} | {v['near_robot_turns']} | {v['same_team_contact_transfers']} |" for name,v in diversity["coverage"].items()]
    decision = ("The primary spatial-conv comparison passes the declared consistency rule on this simulated cohort. Proceed to a controlled retention comparison while retaining both controls. This does not establish real-camera transfer."
                if consistent else "The primary spatial-conv comparison does not pass the declared consistency rule. Isolated winning heads or a favorable seed mean do not establish a reliable pretraining benefit. Keep motion-aware retention deferred and investigate representation/objective alignment before scaling again.")
    initialization = read(experiment/"provenance/initialization_diagnostics.json")
    initialization_text = (f"Additional initialization audit: within-session ball X/Y standard deviations average `{initialization['mean_within_session_ball_xy_std_m']}`m; between-session mean-position standard deviations are `{initialization['between_session_mean_ball_xy_std_m']}`m; one-way coordinate ICC is `{initialization['one_way_random_effects_icc_xy']}`. Sequential Java Random seeds correlate early draws, especially ball X. Fresh snapshots therefore do not imply independent coordinate draws. Session splits and clustered uncertainty are essential; hash/mix episode seeds before a future cohort. No current split or recording was changed."
                           if initialization else "Initialization clustering is controlled by session-level splitting; fresh snapshots are not assumed to be independent statistical samples.")
    report = f"""# LeVJEPA visual baseline v2

Generated {datetime.now().astimezone().isoformat()}. Experiment: `{experiment.relative_to(ROOT)}`.

## Finding

{decision}

The original BASELINE_REPORT.md and DATASET_REPORT.md were reviewed first. Their 576-clip pilot used a single learned 192-dimensional readout rather than spatial probes. Exploratory pilot comparisons informed the v2 protocol before expanded downstream evaluation. A renderer-aware RGB centroid/finite-difference control achieved about 0.2720 m/s test RMSE on that original pilot, confirming useful visible motion information. The toy run remains retired and preserved.

## Dataset, diversity and reproducibility

Native Sumatra AI-vs-AI source `{meta['sumatra_source_sha']}`, release archive SHA256 `{meta['sumatra_release_zip_sha256']}`, recorder SHA256 `{meta['recorder_sha256']}`. No dynamics are implemented here. Compressed 100 Hz streams preserve timestamps, selected applied actions, ball state, robot poses, scenario, session, episode and seed. Five randomized starting affordances are balanced within sessions; native AIs choose all subsequent actions. A passing start does not guarantee a completed pass.

80 JVM sessions × 20 independently initialized episodes; fixed session split 64/8/8, giving **10,240/1,280/1,280 non-overlapping clips**. Each clip has 16 RGB frames near 30 FPS with actual 30/40 ms intervals retained. It spans about 0.5s, versus about 1s in the old 15 FPS pilot. Cohort size, temporal span and training duration changed together, so v1-to-v2 differences cannot be attributed to one factor. The original validation/test sets remain untouched and excluded from pretraining.

CPU renderer: 256×256 RGB, 3× supersampling, JPEG quality 95, minimum ball-marker radius 1.5px. Frozen ViT input: deterministic 224px resize and ImageNet normalization. Minimum marker diameter is only 2.625px at 224px versus 16px patches. Localization predicts the projected RGB center in original 256px coordinates, including the renderer's height cue; velocity predicts true simulator vx/vy at the last frame. These are distinct targets, stored separately from RGB.

Audits checked {audit['physics_ticks']:,} native ticks, {audit['rgb_frames']:,} RGB frames and {audit['clips_checked']:,} clip hashes/targets. Cross-split episode/session/seed overlap and identical RGB clips: zero. Within-split identical clips: {audit['within_split_duplicate_rgb']}. Repeated quantized full sampled trajectories/initial states: {diversity['duplicate_1cm_quantized_trajectories']} / {diversity['duplicate_1cm_quantized_first_states']}. Positions round to 1cm and orientations to .01rad for this screen; it does not prove complete behavioral independence.

| Start family | Episodes | Turning episodes | Native kick timestamps | Barrier-contact frames | Turns near robots | Same-team contact transfers |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

Contact transfers and near-robot turns are descriptive proxies, not tactical pass-success or exhaustive solver collision labels. Evidence: `diagnostics/dataset_diversity.json`.

{initialization_text}

Same-seed replay: {audit['rollout_repeat']['compared_ticks']} ticks, identical={audit['rollout_repeat']['identical']}, first divergence tick={audit['rollout_repeat']['first_different_tick']}, maximum ball divergence={audit['rollout_repeat']['max_ball_position_difference_m']:.4f}m. `SimulationHelper` resets simulator/AI/game-controller state, but inspected upstream `DribblingKickState` calls unseeded `Math.random()` and AI/skill execution uses worker threads. These are uncontrolled factors, not a proven explanation of the first divergent tick. Sumatra source was not patched. Generation continued with hashed archived states: fixed-record rendering and model inputs are reproducible; fresh rollouts are not bitwise reproducible from the initialization seed alone.

## Probes, visual controls and leakage checks

All arms use one frozen eligible cohort, SHA256 `{cohort['cohort_sha256']}`. Eligible counts: `{cohort['clips']}`; visibility exclusions: `{cohort['excluded']}`. The saved visibility flag gates downstream clips; pretraining still sees all training clips, including out-of-canvas-ball cases. Neither target values nor simulator state enter an encoder. Probes receive RGB features and observed last-frame interval. Normalization uses training data only; validation selects alpha/epoch; test predictions follow selection. Pretraining uses separate train/validation Lance stores; test never enters the optimizer. Upstream decoding/sampling/view checks are in `provenance/preflight_*.json`.

1. Learned global readout: 192 values compressing space/time.
2. Mean patch pooling: average all 16×196 dense outputs to 192 values.
3. Spatial patch Ridge: last two causal 14×14 grids; 2×2 pooling and fixed label-independent 192→8 projection preserve a 7×7 layout with manageable linear-probe size. This is a compressed spatial probe, not full-dimensional flattening.
4. Shared spatial heads use all 192 channels: last-frame features plus timestamp-normalized differences feed a 384→32 pointwise MLP; the conv head adds a 32-channel 3×3 layer. Heatmap/subpatch offsets localize the ball, then velocity is read at the **predicted** cell. Ground-truth cells supervise training only. Patch normalization is fit on training samples only.

Ridge alpha grid `{cohort['ridge_alpha_grid']}`. Spatial heads: AdamW lr .001, weight decay .001, 20 epochs, batch 128, identical probe seed 20261117. Validation selection minimizes joint standardized velocity/localization MSE. The original pilot's initial unnormalized heads remain as exploratory evidence; normalization was fixed before v2 evaluation.

Random controls load each run's exact saved initial encoder. The supervised visual control is a four-stage stride-2 CNN (3→32→64→96→192, GroupNorm/GELU), trained on the last two observed RGB frames with training-only labels, selected on validation, then frozen and passed through the **same five probes**. Its supervised end-to-end predictions are retained too. It demonstrates task availability; it is not an architecture- or temporal-budget-matched supervised ViT. The RGB color/centroid control uses fixed renderer-aware color/scale and timestamps, never ground-truth position as input; it is not a real-camera detector.

Mixed-precision ViT compute still returns FP32 residual tokens after concatenation with the FP32 readout. All expanded spatial caches retain FP32, avoiding quantization of contracted features. The exploratory pilot caches used FP16; that was corrected before expanded extraction/probing. The measured precision check and explicit source amendment are retained under `provenance/cache_precision_check.json` and `provenance/source_amendments.json`. Neither amendment changes pretraining parameters, masks or data.

## Unchanged pretraining and measured cost

Upstream `{source['upstream_commit']}` remains unmodified. ViT-Tiny: width 192, depth 12, three heads, patch 16. Uniform IID .10 retention; upstream rounding keeps 20/196 global and 4/36 local tokens per frame-equivalent (89.80% / 88.89% dropped). No motion-aware selection. Original loss/projector/AdamW, lr 4e-4, weight decay .04, bf16, batch 64. WSD warmup 300, stable through step 2,000, decay through 3,000. Fresh starts; no toy/pilot checkpoint resumed. Each seed sees 192,000 clip presentations, about 18.75 nominal passes over the non-overlapping manifest, although upstream samples all valid temporal windows.

| Seed | Optimizer steps | Clip presentations | Measured clips/s | Training wall min | GPU wall hours | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(training_rows)}

Training GPU-stage wall hours: **{training_hours:.4f}**. Including frozen extraction, spatial heads, supervised controls and temporal diagnostics: **{gpu_hours:.4f}**. These elapsed GPU-stage hours include stalls/validation, exclude CPU-only Ridge and simulator generation, and are not utilization-weighted SM hours. Per-stage 0.5s GPU telemetry and per-step timings/losses are saved under `logs/` and training directories. Strict reload, finite inference, optimizer and scheduler state checks passed at step 3,000 for every seed. Initial encoder hashes are distinct. Source snapshots/hashes are in `provenance/source/`, configs/checkpoints/predictions are retained, and uv.lock pins dependencies. Regenerable full patch caches are removed only after successful probes; global/projected features and fitted heads remain.

## Target distribution and results

Regimes were fixed before recording: stationary <.05 m/s, slow [.05,.5), moving [.5,2), fast ≥2.

| Split | Eligible clips | Stationary | Slow | Moving | Fast | Median speed | P95 speed |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(distribution_rows)}

Values below are mean ± sample SD across three encoder seeds. Velocity RMSE is component-wise m/s; localization RMSE is component-wise original-256px error; within-8px uses Euclidean center error. Every probe also saves per-session/scenario/regime metrics and predictions.

| Frozen probe | Encoder arm | Test velocity RMSE | Localization RMSE px | Within 8px |
|---|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

RGB centroid/finite difference: **{detector['test']['aggregate']['velocity_rmse_mps']:.4f} m/s** and **{detector['test']['aggregate']['localization_rmse_px']:.3f}px** test RMSE; both-frame detection rate {cohort['detector_both_frames_rate']:.4f} over the full cohort. Finite differences estimate interval-average velocity, so impacts/kicks and vertical height cues can disagree with the instantaneous target.

Primary spatial-conv velocity result by regime:

| Regime | Test clips | RGB finite difference | Random | LeVJEPA | Supervised CNN |
|---|---:|---:|---:|---:|---:|
{chr(10).join(regime_rows)}

Trained-minus-paired-random seed differences: `{paired.tolist()}`. Mean {paired.mean():.6f} m/s; session-cluster bootstrap 95% percentile interval **[{interval[0]:.6f}, {interval[1]:.6f}]**, 2,000 replicates over {len(sessions)} held-out sessions. Negative favors training. The declared consistency rule requires all paired deltas and the interval to be below zero. Bootstrap conditions on these trained seeds and this renderer; eight sessions remain limited.

| Test session | Mean ± SD trained-minus-random RMSE |
|---|---:|
{chr(10).join(session_rows)}

## Representation and objective diagnostics

| Seed | Arm | Mean channel std | Effective rank /192 | Participation rank | Paired cosine |
|---|---|---:|---:|---:|---:|
{chr(10).join(collapse_rows)}

Low raw variance alone does not prove collapse: Ridge rescales channels and the common field dominates RGB. Compare covariance rank, spatial variance and task performance. Global pooling removes explicit coordinates; the matched table tests whether important task information survives. A localization-only spatial improvement does not prove learned velocity. Projected patch spatial/temporal variation is retained in `diagnostics/patch_variance.json`; readout contraction alone cannot establish total encoder collapse.

Actual upstream crop sampling puts the last-frame ball inside {crop['global']['last_frame_ball_inside_crop_rate']:.3f} of global crops and {crop['local']['last_frame_ball_inside_crop_rate']:.3f} of individual local crops. Under the fixed IID masks, estimated retained-patch overlap is {crop['global']['last_frame_ball_has_retained_patch_probability']:.3f} globally and {crop['local']['last_frame_ball_has_retained_patch_probability']:.3f} per local view. These are last-frame bounding-box probabilities, not whole-clip visibility. Local grayscale probability is {crop['local_gray_probability']:.2f}. Matching a global readout to crops that omit the ball can favor background/robot invariance; this is a plausible objective mismatch, not a proven cause. No crop/objective change was introduced here.

Upstream also samples horizontal flips independently for global and local views, each with default probability .5. A global/local pair therefore has opposite orientation with probability .5. An invariant learned readout can discard signed horizontal velocity or absolute location even when spatial tokens retain them. A controlled no-flip/crop-alignment follow-up should precede attributing a failure to token selection; the current baseline kept the original augmentation objective.

Seed {seeds[0]} inference intervention, applying saved probes without fitting:

| Arm | Probe | Ordered RMSE | Repeat last image 16 times |
|---|---|---:|---:|
{chr(10).join(temporal_rows)}

This retains the last visible location and removes observed motion, but also shifts feature distributions. It is a diagnostic rather than a retrained temporal baseline.

Resolution sensitivity of the fixed RGB detector, resizing cached 224px RGB with the same thresholds:

| Size | Both frames detected | Velocity RMSE | Localization RMSE px |
|---|---:|---:|---:|
{chr(10).join(resolution_rows)}

These are detector measurements, not a trained ViT resolution ablation. Higher model resolution changes token count and compute and needs separate controls.

## Recommendation and limitations

Keep dataset assignments and the probe cohort fixed. If the uniform baseline fails the consistency rule, test crop/objective alignment and resolution before another scale increase, still using uniform selection. Use training/validation for development and a fresh final test cohort for claims after repeated test-driven exploration.

The first token-retention ablation should isolate temporal coverage: **uniform IID clip-level dropping versus uniform per-frame dropping**, both retaining exactly 320 global and 64 tokens per local clip, with the same crops, three seeds, optimizer budget and probes. This requires no labels and tests uneven frame coverage before attributing gains to a motion score. Gate motion-aware retention on useful frozen features. Any later RGB-motion-scored comparison needs equal token counts, a shuffled-score control, regime-specific metrics and measured overhead.

Limits: idealized physics and short starts, not full matches; colored top-down graphics; incomplete action serialization; non-bitwise native rollouts; three seeds/eight test sessions; compressed spatial Ridge; CNN/ViT architecture and context differences; intervention distribution shifts. No real-world transfer, tactical competence or motion-aware novelty is claimed.

```bash
ssl-levjepa dataset expand --output outputs/datasets/sumatra_baseline_v2
ssl-levjepa baseline-v2 --output outputs/NEW_DATED_FOLDER --plan
ssl-levjepa baseline-v2 --output outputs/NEW_DATED_FOLDER
```

Expansion resumes only completed hashed sessions; partial files and existing outputs are preserved. Fixed archived traces reproduce rendering, not fresh AI rollout identity. Original reports/checkpoints/logs remain intact. Evidence: `reports/paired_uncertainty.json`, all `seeds/*/evaluation/*/probes/metrics.json` and predictions, `diagnostics/`, and dataset audit/protocol. All scientific outputs remain local and Git-ignored.
"""
    destination.write_text(report,encoding="utf-8")
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for arm,color in [("random","#777777"),("trained","#2166ac"),("supervised","#b2182b")]:
        means = [np.mean(values(arm,p)) for p in probes]
        deviations = [np.std(values(arm,p),ddof=1) for p in probes]
        offset = {"random":-.22,"trained":0,"supervised":.22}[arm]
        axes[0].bar(np.arange(len(probes))+offset,means,.22,yerr=deviations,label=arm,color=color,capsize=2)
    axes[0].set_xticks(np.arange(len(probes)),["Readout","Mean pool","Spatial Ridge","Pointwise","Conv"],rotation=20)
    axes[0].set_ylabel("Test velocity RMSE (m/s)");axes[0].legend(fontsize=8)
    for arm in arms:
        y = [np.mean(values(arm,"spatial_conv",regime=r)) if detector["test"]["motion"][r]["n"] else np.nan for r in REGIMES]
        axes[1].plot(list(REGIMES),y,marker="o",label=arm)
    axes[1].set_ylabel("Spatial-conv RMSE (m/s)");axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(destination.parent/"baseline_v2_metrics.png",dpi=180);plt.close(fig)
    write_json(destination.parent/"summary.json",{"training_gpu_wall_hours":training_hours,"all_measured_gpu_stage_wall_hours":gpu_hours,
               "uncertainty":uncertainty,"report_sha256":checksum(destination)})
    return destination
