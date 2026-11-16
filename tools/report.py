"""Generate a local measured run report and loss plot; never a research result."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path)
    args = ap.parse_args()
    gate_path = ROOT / "logs/validation_passed.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    paths = []
    if "prefix" in gate:
        paths.extend(ROOT / (gate["prefix"]+suffix) for suffix in ["_one", "_smoke50", "_profile"])
    if args.run:
        paths.append(args.run.resolve())
    summaries, rows, gpu = [], [], []
    for path in paths:
        if (path / "summary.json").exists():
            summaries.append((path, read_json(path / "summary.json")))
        if (path / "metrics.jsonl").exists():
            rows.extend(json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines())
        if (path / "gpu.csv").exists():
            with (path / "gpu.csv").open() as stream:
                gpu.extend(dict((k.strip(), v.strip()) for k, v in r.items()) for r in csv.DictReader(stream))
    def values(key):
        vals = []
        for row in gpu:
            try:
                vals.append(float(row[key].split()[0]))
            except (KeyError, ValueError):
                continue
        return vals
    temps, utils, powers = values("temperature.gpu"), values("utilization.gpu [%]"), values("power.draw [W]")
    losses = [r["loss"] for r in rows]
    sha = subprocess.check_output(["git", "-C", str(ROOT / "upstream/LeVJEPA"), "rev-parse", "HEAD"], text=True).strip()
    pre = read_json(ROOT / "data/toy_ssl/preflight.json")
    bench_path = ROOT / "logs/benchmark.json"
    bench = read_json(bench_path) if bench_path.exists() else {}
    latest = paths[-1] if paths else ROOT / "runs"
    verified_path = latest / "checkpoints/final.verified.json"
    verified = read_json(verified_path) if verified_path.exists() else None
    verify_result = (f"passed at optimizer/scheduler step {verified['global_step']}; "
                     f"{verified['optimizer_state_entries']} AdamW state entries; strict encoder reload; finite {verified['inference_shape']} readout"
                     if verified else "pending")
    model = read_json(latest / "model.json") if (latest / "model.json").exists() else {}
    status = summaries[-1][1]["status"] if summaries else "not completed"
    resume_note = ("The 1200-step target is complete; extending it requires an explicit new step limit and schedule decision."
                   if rows and rows[-1]["optimizer_step"] >= 1200 else
                   "The resume command continues to the absolute 1200-step target.")
    table = "\n".join(f"| {p.name} | {s['initial_step']} -> {s['final_step']} | {s['wall_sec']:.2f} | {s['mean_loss']:.6f} | {s['status']} |" for p, s in summaries)
    resume = f"bash scripts/run_stage.sh runs/resume_$(date +%Y%m%d_%H%M%S) 1200 --resume {latest.relative_to(ROOT)}/checkpoints/final.ckpt"
    report = f"""# Local LeVJEPA toy bootstrap

Generated: {datetime.now().astimezone().isoformat()}

**Toy data only; no empirical research conclusion yet.** This run does not validate the research hypothesis.

## Environment

Windows host, Ubuntu 24.04 WSL2; isolated project `.venv`, local uv, dependencies pinned in `uv.lock`.
Upstream: https://github.com/MLO-lab/LeVJEPA at `{sha}`; checkout unmodified.
Device: {pre['device']}, {pre['vram_bytes']/1024**3:.2f} GiB, driver {pre['driver']}.
PyTorch {pre['torch']}, wheel CUDA {pre['torch_cuda']}, bf16 supported: {pre['bf16_supported']}.
Driver CUDA capability is recorded by nvidia-smi in `logs/environment.txt`; it differs from the wheel runtime.
Full versions and installation changes: `logs/requirements-frozen.txt`, `logs/setup.log`.

## Data and configuration

Pilot: 4 episodes x 32 frames. Full store: 160 episodes x 96 frames = 15,360 JPEG RGB frames at 256 x 256, 15 FPS, seed 42.
Metadata: `dataset_kind=TOY_SMOKE_TEST_NOT_ROBOCUP_PHYSICS`. CPU Pillow rendering, contiguous episodes, monotonically increasing frame indices.
Actual upstream sampler/decoder and GPU view builder passed; each of the first sampled clip's 16 frames matched direct decode without crossing an episode.
Preview: `preview.png`. Tensor shapes/dtypes/range: `data/toy_ssl/preflight.json`.

Encoder: upstream v2 ViT-Tiny size, width 192, depth 12, heads 3, patch 16, attention window 16.
Measured parameter count: {model.get('encoder_parameters')} encoder, {model.get('total_parameters')} including projector.
One 224px global view, four 96px local views, 16 frames, batch 8, no gradient accumulation, bf16 mixed.
IID uniform retention at 10% rounded up: 20/196 global tokens and 4/36 local tokens per frame-equivalent.
Unmodified upstream invariance + 0.06*SIGReg, projector 2048 -> 256, AdamW lr 4e-4, betas (0.9,0.95), weight decay 0.04, norm clipping 1.
WSD warmup 50 steps, stable through step 1000, decay over final 200 of 1200. Compile and optional evaluation EMA disabled.
16 fixed held-out toy episodes; no scientific evaluation metric. All resolved settings are stored per run in `config.yaml`.
Local CSV/JSON logs only, W&B disabled. Launcher caps each invocation at 8h actual wall time; total target is absolute optimizer step 1200.

## Measured validation

| Stage | Optimizer steps | Training wall seconds | Mean loss | Exit |
|---|---:|---:|---:|---|
{table}

Profile: {bench.get('measured_steps')} timed training steps after {bench.get('warmup_steps_excluded')} warmup steps.
Steps/sec: {bench.get('steps_per_sec')}; decoded clips/sec: {bench.get('decoded_clips_per_sec')}.
Mean compute seconds/step: {bench.get('mean_compute_sec')}; mean blocking dataloader wait seconds/step: {bench.get('mean_data_wait_sec')}.
End-to-end throughput includes transfer, checkpointing and validation overhead. CUDA synchronizes at each step boundary.
These are training clip presentations, not unique videos or evaluation throughput; overlapping windows can repeat frames.
Peak allocated bytes: {max((r['peak_allocated_bytes'] for r in rows), default=0)}; peak reserved bytes: {max((r['peak_reserved_bytes'] for r in rows), default=0)}.
Safety limit: 11 GiB reserved; desktop usage is also present in nvidia-smi telemetry.

## Current result

Completed logged optimizer steps across stages: {len(rows)}; final absolute step: {rows[-1]['optimizer_step'] if rows else 0}.
Mean loss: {sum(losses)/len(losses) if losses else None}; final loss: {losses[-1] if losses else None}.
Non-finite logged losses: {sum(not math.isfinite(x) for x in losses)}. Exceptions, if any, remain in stage stdout logs.
Approximate active training GPU-hours (training wall time): {sum(s['wall_sec'] for _, s in summaries)/3600:.5f}.
GPU telemetry every 30 seconds, including startup: {len(gpu)} samples.
Average GPU utilization: {sum(utils)/len(utils) if utils else None}%; temperature range: {min(temps) if temps else None} to {max(temps) if temps else None} C; mean power: {sum(powers)/len(powers) if powers else None} W.
Status: **{status}**. Latest run: `{latest.relative_to(ROOT)}`.
Latest full-state checkpoint: `{latest.relative_to(ROOT)}/checkpoints/final.ckpt`.
Periodic checkpoint policy: every 100 optimizer steps, retain two latest periodic checkpoints plus last/final; no best model is claimed.
Checkpoint reload evidence: each resumed stage's `resume_evidence.json` records restored optimizer entries and min/max AdamW state steps matching its nonzero restored global step.
Final checkpoint strict reload/inference evidence: `{verified_path.relative_to(ROOT)}`; result: {verify_result}.

## Commands

```bash
bash scripts/setup.sh
bash scripts/preflight.sh
bash scripts/profile_3060.sh
tmux new-session -d -s ssl-levjepa bash scripts/run_overnight.sh
# Resume into a NEW output directory, retaining absolute step limit:
{resume}
# Read and run inference from the final checkpoint:
.venv/bin/python tools/verify_checkpoint.py {latest.relative_to(ROOT)}/checkpoints/final.ckpt
```

Exact tested invocation paths and timestamps are in `logs/`; benchmark JSON: `logs/benchmark.json`.
{resume_note}

## Issues and next action

- Upstream Manager may create a saver in the resume input directory; this wrapper uses the same upstream Module/data/loss/optimizer with Lightning directly to preserve previous checkpoints.
- Lightning restores optimizer, scheduler and step state. Upstream sampler has epoch-seeded sampling but no resumable mid-epoch cursor; resumption is not bitwise identical to uninterrupted sample order.
- Final checkpoints of short stages advance Lightning's epoch counter. The long stage uses unlimited epochs with an absolute 1200-step cap and an explicit unchanged 1200-step learning-rate schedule.
- WSL on the Windows-mounted project filesystem; measured input wait determines whether this is a bottleneck.
- PyTorch warns that upstream RMSNorm uses a mixed-dtype fallback rather than its fused implementation; the upstream computation is preserved.
- Toy loss curves are pipeline checks and cannot establish useful representations or motion-guided-token novelty.

Best next action: replace the toy generator with reproducible, timestamped actual SSL simulator trajectories rendered to RGB, with held-out match splits and separate state labels; establish the uniform-retention baseline before changing token selection.
"""
    (ROOT / "RUN_REPORT.md").write_text(report)
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.plot([r["optimizer_step"] for r in rows], losses, linewidth=.8)
        axis.set(xlabel="Optimizer step", ylabel="Upstream LeVJEPA loss",
                 title="Toy pipeline check only — no empirical research conclusion")
        axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(ROOT / "logs/loss.png", dpi=140)
        plt.close(fig)
    print(ROOT / "RUN_REPORT.md")


if __name__ == "__main__":
    main()
