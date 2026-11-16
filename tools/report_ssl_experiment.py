"""Write local dataset/baseline reports from measured artifacts only."""
import argparse
from datetime import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,default=ROOT/"data/ssl_sumatra_pilot_v2")
    ap.add_argument("--experiment",type=Path,required=True)
    args=ap.parse_args()
    dataset=args.dataset.resolve(); experiment=args.experiment.resolve()
    meta=read(dataset/"metadata.json"); audit=read(dataset/"audit.json")
    environment=read(dataset/"train/preflight.json",{})
    stats=meta["stats"]; repeat=audit["rollout_repeat"]
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

Lance columns: `episode_idx`, `step_idx`, `timestamp_ns`, `frame`, `h`, `w`, `label=-1`. Ground-truth probe labels remain in the separate clip manifest/raw recordings. Episode index files are built by upstream LeVJEPA's builder.

Passed: every 10 ms tick interval; finite recorded states; 12 robots/sample; exactly 128 samples/episode; {meta['render_and_label_alignment_checks']}; all {audit['clip_labels_checked']} clip-target timestamps and ball-velocity labels checked against source states; no temporal boundary crossing; zero cross-split episode/session overlap and zero identical RGB clips crossing splits. Actual upstream dataloader/view checks are saved under each split's `preflight.json`.

Preview: `{dataset.relative_to(ROOT)}/preview.png`; animated 16-frame example: `{dataset.relative_to(ROOT)}/clip16.gif`.

## Reproducibility limits

The repeat test compared {repeat['compared_ticks']} ticks with the same initialized seed. Identical state/action record values: **{repeat['identical_record_values']}**. First different tick: {repeat['first_different_tick']}; maximum subsequent ball-position divergence: {repeat['max_ball_position_difference_m']:.4f} m. The initialization seed does not control all native AI randomness/timing or game-controller state. Do not claim bitwise rollout reproducibility.

The fixed archived recordings, hashes, split manifest and deterministic renderer make this dataset and baseline input reproducible. Regenerating a new AI rollout from seed alone remains an integration limitation. The early draft with an incorrectly named action-velocity unit was stopped, kept as debug output only, and excluded entirely from this pilot.

## Tested commands and next experiment

```bash
bash scripts/setup_ssl_simulation.sh
bash scripts/generate_pilot.sh
.venv/bin/python tools/audit_ssl_dataset.py --dataset data/ssl_sumatra_pilot_v2
bash scripts/run_ssl_experiment.sh
```

The original 1200-step toy checkpoint is unchanged: `{audit['original_toy_checkpoint_sha256']}`. Toy training was not continued.

Next dataset action: control Sumatra AI/game-controller randomness and state, then collect a larger cohort of independent matches with fixed held-out sessions. For stronger physical fidelity, build a pinned modern ER-Force simulator and export its internal `world::SimulatorState`, while retaining Sumatra as the AI. Never replace missing simulator state with invented trajectories.
"""
    (ROOT/"DATASET_REPORT.md").write_text(data_report)
    bench=read(experiment/"batch_benchmark/results.json",[])
    table=[]
    for r in bench:
        if r["status"]=="passed":
            table.append(f"| {r['batch_size']} | {r['samples_per_sec']:.2f} | {r['mean_step_sec']*1000:.2f} | {r['peak_allocated_bytes']/1024**3:.3f} / {r['peak_reserved_bytes']/1024**3:.3f} | {r['gpu_util_mean_percent']:.1f} | {r['device_peak_used_mib']:.0f} |")
        else:
            table.append(f"| {r['batch_size']} | {r['status']} | — | — | — | — |")
    summary=read(experiment/"baseline/summary.json")
    reload=read(experiment/"baseline/checkpoints/final.verified.json")
    probe=read(experiment/"probe/metrics.json")
    training_text=(f"Completed {summary['completed_steps']} optimizer steps from step {summary['initial_step']} to {summary['final_step']}, "
        f"with effective batch {summary['samples_seen']//summary['completed_steps']} and {summary['samples_seen']} clip presentations. "
        f"Exit reason: {summary['status']}. Mean/final training loss: {summary['mean_loss']:.6f} / {summary['final_loss']:.6f}; "
        f"non-finite losses: {summary['nonfinite_count']}. Training wall time including worker startup: {summary['wall_sec']:.2f} s "
        f"(~{summary['gpu_hours']:.4f} active training GPU-hours). After {summary['warmup_steps_excluded']} warmup steps, "
        f"{summary['measured_steps']} steps measured {summary['decoded_clips_per_sec']:.2f} clips/sec, "
        f"{summary['mean_compute_sec']*1000:.2f} ms compute and {summary['mean_data_wait_sec']*1000:.2f} ms blocking input wait per step. "
        f"Peak reserved memory: {summary['peak_reserved_bytes']/1024**3:.3f} GiB. Exact values: `baseline/summary.json`."
        if summary else "Pending; no completed baseline result is claimed.")
    reload_text=(f"Passed strict encoder reload and finite {reload['inference_shape']} inference. "
        f"Global step, all {reload['optimizer_state_entries']} AdamW state entries and scheduler are at step {reload['global_step']}. "
        f"Checkpoint SHA256: `{reload['sha256']}`. Evidence: `baseline/checkpoints/final.verified.json`."
        if reload else "Pending; no checkpoint reload is claimed.")
    results="Pending. No probe result is claimed."
    if probe:
        probe_table="\n".join(f"| {name} | {v['validation']['rmse_mps']:.5f} | {v['test']['rmse_mps']:.5f} | {v['test']['mae_mps']:.5f} |" for name,v in probe["results"].items())
        results=f"""Feature extraction: {probe['clips']}; excluded: {probe['excluded']}.
All 196 patch tokens/frame, deterministic 224px resize and ImageNet normalization, encoder eval mode, bf16, frozen 192-dimensional readout. Labels are simulator vx/vy at the last input frame. StandardScaler and Ridge fit training clips only; alpha in [0.1,1,10,100,1000] selected by validation RMSE; the test split is evaluated after selection. Neither state nor action labels enter pretraining.

| Frozen representation / baseline | Validation RMSE m/s | Test RMSE m/s | Test MAE m/s |
|---|---:|---:|---:|
{probe_table}

Per-session test metrics, alpha search and target distributions are in `{experiment.relative_to(ROOT)}/probe/metrics.json`. The random model uses the exact saved pretraining initialization and identical clips/preprocessing. The constant predictor uses the training-label mean. There are only two test sessions and one training seed; these point estimates do not establish a statistically reliable research improvement.
"""
        paired={s:probe["results"]["trained"]["test_per_session"][s]["rmse_mps"]-
                  probe["results"]["random_initialization"]["test_per_session"][s]["rmse_mps"]
                  for s in probe["results"]["trained"]["test_per_session"]}
        results+=f"\nTrained-minus-random RMSE by held-out session (negative is better): `{json.dumps(paired)}`. The average gain is not consistent across both sessions.\n"
    baseline_report=f"""# LeVJEPA simulator pilot baseline

Report generated {datetime.now().astimezone().isoformat()}. Experiment: `{experiment.relative_to(ROOT)}`.

## Batch-size benchmark on RTX 3060 12 GB

Measured software/device: {environment.get('device')}, PyTorch {environment.get('torch')}, CUDA wheel runtime {environment.get('torch_cuda')}, driver {environment.get('driver')}, bf16 supported {environment.get('bf16_supported')}. Dependencies are pinned in `uv.lock`; per-run environment files retain the actual installed versions.

Fresh disposable runs on simulator RGB only; no toy checkpoint or toy data used. Same ViT-Tiny architecture and unmodified upstream loss/optimizer, bf16, uniform 10% IID retention, 16 frames, 224px global + four 96px local views. Each size: 20 warmup optimizer steps + 100 timed optimizer steps, CUDA synchronization at boundaries. Samples/sec measures decoded training clip presentations end to end, including input wait/transfer and logging. Peak allocator memory includes warmup; GPU utilization/device memory sampled every 0.5s and filtered to the timed interval. Device memory/utilization also include the Windows desktop.

| Micro-batch | Samples/sec | Mean step ms | Allocated / reserved GiB | GPU util % | Device peak MiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table) if table else '| pending | — | — | — | — | — |'}

Per-step, GPU and failure logs: `{experiment.relative_to(ROOT)}/batch_benchmark/`. Recommendation uses highest measured throughput while leaving the measured memory safety margin; it is saved separately.
Upstream SIGReg's default statistic scales with batch size, so loss magnitudes across these micro-batches are not accuracy comparisons.

## Baseline training and compatibility

Fresh initialization, seed 42; no continuation of the infrastructure-only toy checkpoint. Upstream `{meta.get('levjepa_sha','a748e0fbaa0ec548cad283978b818779708a3bd7')}` unchanged. Encoder: 192 width, 12 blocks, 3 heads, patch 16. One global and four local views; uniform token retention (20 global and 4 local tokens/frame-equivalent); no motion-aware token selection, compile or optional evaluation EMA. Unmodified invariance + 0.06*SIGReg, upstream AdamW lr 4e-4/betas (0.9,0.95)/weight decay .04 and norm clipping 1. WSD warmup 30, stable until step 200, decay over final 100 of the 300-step pilot. Training sees only the 48 training episodes; 12 validation episodes are in a separate store. Test episodes never enter pretraining or hyperparameter selection.

All effective settings: `baseline/config.yaml`. {training_text}
The cached validation loss uses batches of four clips. Because upstream SIGReg scales with batch size, raw training and validation loss magnitudes are not directly comparable; the frozen velocity probe is the downstream comparison.
Checkpoint strict reload/inference and optimizer/scheduler state verification: {reload_text}
Periodic full-state checkpoints every 100 optimizer steps, with saved random encoder initialization for the control. Native upstream checkpoint compatibility is retained. The upstream sampler still has no mid-epoch resume cursor; a resumed run restores weights/optimizer/step, but not exact sample order.

## Frozen-feature ball-motion task

{results}

## Commands, limitations and next experiment

```bash
bash scripts/run_ssl_experiment.sh
# The finisher can use completed benchmarks when no baseline directory exists:
bash scripts/finish_ssl_experiment.sh {experiment.relative_to(ROOT)}
.venv/bin/python tools/report_ssl_experiment.py --experiment {experiment.relative_to(ROOT)}
```

The small idealized native-simulation pilot validates the data-to-representation-to-probe experiment path. It cannot establish real-world transfer, physical-model accuracy or motion-guided masking novelty. The measured AI rollout repeatability limitation is documented in DATASET_REPORT.md. Exact fixed dataset hashes, clips, features, predictions and split masks are retained locally. The original toy logs/checkpoints are preserved.

Next experiment: enlarge the independent-session dataset and repeat this frozen probe with three fresh training seeds, holding the test sessions fixed. Require a consistent advantage over the random-encoder and constant controls before attributing useful motion representations to this short baseline. Keep uniform retention until that baseline is established.
"""
    (ROOT/"BASELINE_REPORT.md").write_text(baseline_report)
    print(ROOT/"DATASET_REPORT.md"); print(ROOT/"BASELINE_REPORT.md")


if __name__=="__main__":
    main()
