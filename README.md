# SSL × LeVJEPA

A local RTX 3060 12 GB pipeline for RoboCup Small Size League simulator video datasets and representation-learning baselines.

The completed 1,200-step hand-written toy run is successful infrastructure validation. Toy training is now retired; the runner rejects stores tagged as toy data. The native SSL simulator pilot below is the active experiment. Neither pilot establishes a publication-level research conclusion.

The official [LeVJEPA](https://github.com/MLO-lab/LeVJEPA) checkout is a Git submodule pinned to `a748e0fbaa0ec548cad283978b818779708a3bd7`. Its v2 encoder, Lance loader, augmentations, invariance + SIGReg loss, optimizer factory and scheduler are used without source changes. The wrapper adds measurements and isolated checkpoint output. Lightning is called directly because upstream Manager can otherwise add a saver in the directory of a resume input.

## Historical infrastructure setup

Linux/WSL2, NVIDIA CUDA GPU, Python 3.12, Git, curl and tmux are required. All Python dependencies and uv stay inside this project. No global Python or CUDA changes are needed. These were the original bootstrap commands; use the native simulator workflow below for new experiments.

```bash
git clone --recurse-submodules https://github.com/AhmadSaeedZaidi/ssl-levjepa.git
cd ssl-levjepa
bash scripts/setup.sh
bash scripts/inspect.sh
bash scripts/preflight.sh
bash scripts/profile_3060.sh
# Only after all measured gates pass:
tmux new-session -d -s ssl-levjepa bash scripts/run_overnight.sh
```

The preflight builds a 4 × 32 pilot, then 160 × 96 deterministic JPEG RGB frames in Lance with seed 42. It checks upstream decoding, clip boundaries/order and normalized view shapes. The training validation runs one optimizer step, resumes for 50 more, then resumes for 25 warmup + 200 measured steps. The overnight stage resumes to an absolute total of 1,200 steps. Each invocation has an eight-hour wall limit; failures preserve nonzero exit codes. No stage silently restarts or overwrites an existing output directory.

`configs/ssl_toy_3060_12gb.yaml` configures a 192-wide, 12-block, three-head encoder, 16-frame clips, one 224px global view, four 96px local views, 10% IID token retention, batch 8 and bf16 mixed precision (fp16/scaler fallback if unsupported). The unmodified upstream recipe uses AdamW and invariance + 0.06 × SIGReg. Optional evaluation EMA and compilation are disabled. Sixteen toy episodes are held out.

## Outputs and resume

`data/`, `runs/`, `logs/`, `preview.png` and `RUN_REPORT.md` are local and Git-ignored. The public repository contains source, configuration, a dependency lock and this README. Checkpoints are saved every 100 optimizer steps; two recent periodic checkpoints plus last/final are retained. JSONL metrics record every step. GPU temperature, utilization and power are sampled every 30 seconds.

```bash
# Use a NEW output directory and an explicit existing checkpoint:
bash scripts/run_stage.sh runs/resumed_run 1200 --resume runs/previous_run/checkpoints/final.ckpt
.venv/bin/python tools/verify_checkpoint.py runs/resumed_run/checkpoints/final.ckpt
.venv/bin/python tools/report.py --run runs/resumed_run
tmux attach -t ssl-levjepa
```

Resume restores model, optimizer, scheduler and step count. Upstream's sampler has no saved mid-epoch cursor, so resumed sample order is not bitwise identical to uninterrupted training. Throughput is measured end to end after warmup, with CUDA synchronization, separately logged blocking input wait and training compute. No throughput numbers are claimed here before local measurements.

The next experiment should use actual, reproducible SSL gameplay trajectories rendered to RGB, separate state labels and held-out match splits. Establish uniform retention before comparing motion-guided retention at the same token count.

## Native SSL simulator pilot

The completed toy run is retained as infrastructure validation. Further experiments use actual Sumatra 2025 native simulator states and both built-in AIs. A Java adapter uses blocking 10 ms steps, records robot/ball state and selected applied actions, and initializes varied scenario snapshots. Physics is provided by Sumatra's idealized native models; this is not Bullet/ODE or real camera footage. Upstream source remains unmodified.

```bash
bash scripts/setup_ssl_simulation.sh
bash scripts/generate_pilot.sh
bash scripts/run_ssl_experiment.sh
```

The pinned public release and source hashes are in `simulation/sources.json`. The pilot uses 72 episodes across 12 independent sessions, with an 8/2/2 session split. Raw state/action streams are recorded at 100 Hz and sampled near 15 FPS. The deterministic CPU renderer creates 256px Lance stores and separate 16-frame downstream-label manifests. The ball uses a documented 1.5px minimum display radius. Frame bytes, timestamps, labels, split overlap and duplicate clips are audited before training.

Java 21 with the `jdk.compiler` module is required for the adapter; compilation uses the released Sumatra JARs without a Gradle build. Simulator downloads, generated configuration and classes stay under `.sim/`.

The experiment benchmarks batches 8/16/32/64 with 20 warmup + 100 timed optimizer steps, chooses the fastest size with memory headroom, then trains a fresh 300-step uniform-retention baseline. A frozen readout/Ridge probe predicts ball vx/vy, compared with the exact random encoder initialization and a training-label mean control. Scaling and probe fitting use training clips only; validation selects regularization; test sessions remain held out.

Fixed archived trajectories render reproducibly. Repeating a native AI rollout with the same initialization seed is **not bitwise reproducible**: not all AI/game-controller randomness is controlled. The local `DATASET_REPORT.md` and `BASELINE_REPORT.md` record this limitation, measured results and provenance. These reports and all experiment data/logs/checkpoints remain Git-ignored. The pilot is not sufficient evidence of real-world transfer or a research improvement.
