# SSL × LeVJEPA

A local RTX 3060 12 GB pipeline bootstrap for future motion-aware RoboCup Small Size League video representation experiments.

**Toy data only; no empirical research conclusion yet.** The CPU renderer produces simplified moving discs, not SSL simulation or real gameplay. Training loss on this dataset cannot validate the research hypothesis.

The official [LeVJEPA](https://github.com/MLO-lab/LeVJEPA) checkout is a Git submodule pinned to `a748e0fbaa0ec548cad283978b818779708a3bd7`. Its v2 encoder, Lance loader, augmentations, invariance + SIGReg loss, optimizer factory and scheduler are used without source changes. The wrapper adds measurements and isolated checkpoint output. Lightning is called directly because upstream Manager can otherwise add a saver in the directory of a resume input.

## Run locally

Linux/WSL2, NVIDIA CUDA GPU, Python 3.12, Git, curl and tmux are required. All Python dependencies and uv stay inside this project. No global Python or CUDA changes are needed.

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
