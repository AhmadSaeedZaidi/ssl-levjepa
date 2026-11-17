# SSL × LeVJEPA

RoboCup Small Size League simulator video datasets and LeVJEPA baselines on an RTX 3060 12 GB. Sumatra supplies the simulation and team AIs; Python records experiments, renders video and runs the pinned upstream model.

```text
ssl-levjepa/
├── src/ssl_levjepa/           # One Python package and CLI
│   ├── simulator.py          # Runtime setup and Java recorder integration
│   ├── dataset.py            # CPU rendering, Lance datasets and validation
│   ├── training.py           # Upstream recipe, measurements and checkpoint checks
│   ├── benchmark.py          # Isolated batch-size measurements
│   ├── evaluation.py         # Frozen-feature ball-velocity probe
│   ├── reporting.py          # Reports from measured artifacts
│   ├── experiment.py         # Experiment orchestration and output bundles
│   ├── layout.py             # Paths, including historical-path resolution
│   └── cli.py                # All commands below
├── simulation/
│   ├── sumatra/java/SumatraRecorder.java
│   └── sources.json          # Simulator source/release pins
├── configs/sumatra_pilot.yaml
├── tests/                   # CLI and migration regression checks
├── upstream/LeVJEPA/         # Unmodified, pinned Git submodule
├── outputs/                 # Local only; all scientific artifacts
│   ├── datasets/sumatra_pilot_v2/
│   │   ├── trajectories/    # Original simulator state/action recordings
│   │   ├── train/           # Lance store and episode index
│   │   ├── validation/
│   │   ├── test/
│   │   ├── previews/
│   │   └── …                # Clip manifest, metadata and audit
│   ├── 2026-11-16_101113_sumatra_baseline/
│   │   ├── config.yaml      # Effective configuration
│   │   ├── dataset.json     # Shared dataset reference and manifest hashes
│   │   ├── benchmark/
│   │   ├── training/        # Checkpoints, metrics and saved initialization
│   │   ├── evaluation/      # Features, predictions and probe metrics
│   │   ├── logs/
│   │   ├── reports/         # DATASET_REPORT.md and BASELINE_REPORT.md
│   │   ├── provenance/      # Verification and original migration evidence
│   │   └── status.json
│   ├── archive/             # Preserved toy and simulator-debug artifacts
│   ├── latest.json
│   └── migration.json       # Old/new paths and artifact SHA-256 inventory
├── .cache/                  # Simulator downloads/classes and environment cache
├── bootstrap.sh             # Environment setup only
├── pyproject.toml
└── uv.lock
```

Datasets are shared across experiments. Each new experiment gets a dated directory containing its configuration, results, logs and reports. Existing stage outputs are preserved; reruns need a new destination. Historical saved paths resolve through the local migration manifest without editing old configurations or checkpoints. `outputs/`, `.cache/`, `.tools/` and `.venv/` are Git-ignored.

## Setup

Use Linux/WSL2 with Python 3.12, Git, curl, CUDA access and Java 21 including `jdk.compiler`. All commands run from the repository root. Python dependencies and simulator assets stay inside the project.

```bash
git clone --recurse-submodules https://github.com/AhmadSaeedZaidi/ssl-levjepa.git
cd ssl-levjepa
bash bootstrap.sh
source .venv/bin/activate
ssl-levjepa --help
```

The Java recorder compiles against the released Sumatra JARs; no Gradle build or source patch is needed. The Python package coordinates the native simulator and uses PyTorch/CUDA for model computation. Simulation dynamics remain in Sumatra.

## Dataset and experiment commands

Setup downloads the pinned simulator. Dataset building records new AI-vs-AI trajectories, renders deterministic CPU RGB clips, and audits session splits and label alignment. It refuses an existing dataset destination.

```bash
ssl-levjepa simulator setup
ssl-levjepa dataset build --output outputs/datasets/sumatra_pilot_v2
ssl-levjepa dataset audit sumatra_pilot_v2
ssl-levjepa dataset validate outputs/datasets/sumatra_pilot_v2/train

# Inspect stages and paths without writing or running them:
ssl-levjepa run --dataset sumatra_pilot_v2 --plan

# Start a fresh benchmark + baseline + frozen evaluation in a dated bundle:
ssl-levjepa run --dataset sumatra_pilot_v2
ssl-levjepa info
```

Individual stages use the same CLI:

```bash
ssl-levjepa benchmark --dataset sumatra_pilot_v2 --output outputs/batch_comparison/benchmark
ssl-levjepa train --config configs/sumatra_pilot.yaml --output outputs/my_baseline/training
ssl-levjepa checkpoint verify outputs/my_baseline/training/checkpoints/final.ckpt
ssl-levjepa evaluate --dataset sumatra_pilot_v2 --run outputs/my_baseline/training --output outputs/my_baseline/evaluation
```

The full `run` command writes dataset references and reports automatically. Use `report --run <experiment>` to generate reports in a complete bundle; use `--output <new-directory>` to generate another copy. `train --resume <checkpoint>` restores model, optimizer, scheduler and step into a **new** training directory; upstream does not save its mid-epoch sampling cursor, so resumed sample order is not exact. Audit, validation and checkpoint checks write evidence only when `--output` is supplied.

## Existing pilot

The completed 1,200-step toy run is successful infrastructure validation and is archived. Further toy training is rejected. The active pilot contains 72 short AI-vs-AI episodes across 12 JVM sessions, split 8/2/2 sessions into train/validation/test. The deterministic renderer produces 576 non-overlapping 16-frame RGB clips. Simulator ground truth is kept separately for downstream labels.

The existing experiment measured batches 8/16/32/64, then trained a fresh 300-step uniform-retention baseline and evaluated frozen ball-velocity features against saved random-encoder and constant controls. Actual measurements and limitations are in the local experiment's `reports/`; the restructure preserves the original evidence under `provenance/`.

Sumatra's native physics is idealized and the scenarios are short controlled starts, not complete refereed matches or camera footage. Repeating AI rollouts from the same initialization seed is not bitwise deterministic. Archived trajectories and rendered inputs are reproducible. One training seed and two held-out test sessions do not establish a reliable research improvement or real-world transfer.

Upstream [LeVJEPA](https://github.com/MLO-lab/LeVJEPA) remains unmodified at `a748e0fbaa0ec548cad283978b818779708a3bd7`. Its encoder, Lance loader, augmentations, invariance + SIGReg loss, optimizer and scheduler are used directly. No motion-aware token selection is introduced. Lightning is called directly to keep new outputs away from a resume input's directory.

```bash
python -m unittest discover -s tests -v
```
