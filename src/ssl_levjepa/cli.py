"""One command surface for simulator data and representation experiments."""
import argparse
import json
from pathlib import Path

from . import __version__
from .layout import DEFAULT_CONFIG, dataset_path, experiment_dataset, resolve_path


def parser():
    command = argparse.ArgumentParser(prog="ssl-levjepa", description=__doc__)
    command.add_argument("--version", action="version", version=__version__)
    groups = command.add_subparsers(dest="command", required=True)

    sim = groups.add_parser("simulator", help="Prepare and operate native simulators").add_subparsers(dest="action", required=True)
    sim.add_parser("setup", help="Prepare the checksum-pinned Sumatra runtime")
    record = sim.add_parser("record", help="Record one session using the Java adapter")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--session", type=int, default=0)
    record.add_argument("--episodes", type=int, default=6)
    record.add_argument("--samples", type=int, default=128)
    record.add_argument("--seed", type=int, default=20261116)

    data = groups.add_parser("dataset", help="Build, render and audit simulator datasets").add_subparsers(dest="action", required=True)
    build = data.add_parser("build", help="Record the pilot, then render and audit it")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--output", type=Path)
    expand = data.add_parser("expand", help="Record and render the fixed expanded baseline cohort; resume at completed sessions")
    expand.add_argument("--output", type=Path, required=True)
    render = data.add_parser("render", help="Render existing recorded trajectories")
    render.add_argument("dataset")
    for name, help_text in [("audit", "Check hashes, split isolation and label alignment"), ("validate", "Test a split with the actual upstream dataloader")]:
        check = data.add_parser(name, help=help_text)
        check.add_argument("path")
        check.add_argument("--output", type=Path, help="Write new evidence here; existing files are preserved")

    run = groups.add_parser("run", help="Run the complete baseline experiment into one output bundle")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--dataset")
    run.add_argument("--name", default="sumatra_baseline")
    run.add_argument("--output", type=Path, help="New experiment directory")
    run.add_argument("--plan", action="store_true", help="Print stages and paths without running or writing anything")
    expanded = groups.add_parser("baseline-v2",help="Three-seed expanded baseline and identical frozen probes")
    expanded.add_argument("--config",type=Path,default=DEFAULT_CONFIG.parent/"baseline_v2.yaml")
    expanded.add_argument("--dataset",default="sumatra_baseline_v2")
    expanded.add_argument("--output",type=Path,required=True)
    expanded.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    expanded.add_argument("--steps",type=int,default=3000)
    expanded.add_argument("--plan",action="store_true")

    ablations = groups.add_parser("ablate-v2", help="Run matched augmentation and per-frame masking on an existing baseline")
    ablations.add_argument("--baseline", type=Path, required=True)
    ablations.add_argument("--output", type=Path, required=True)
    ablations.add_argument("--plan", action="store_true")

    train = groups.add_parser("train", help="Train with the unmodified upstream recipe")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--output", type=Path, required=True, help="New training stage directory")
    train.add_argument("--steps", type=int)
    train.add_argument("--warmup", type=int, default=0)
    train.add_argument("--resume", type=Path)
    train.add_argument("--dataset", type=Path, help="Override the training split")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--benchmark", action="store_true", help="Disposable optimizer timing, no checkpoints")
    train.add_argument("--no-holdout", action="store_true")

    checkpoint = groups.add_parser("checkpoint", help="Inspect saved training state").add_subparsers(dest="action", required=True)
    verify = checkpoint.add_parser("verify", help="Strictly reload and run one inference clip")
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--dataset", type=Path, help="Optional replacement split for compatibility checks")
    verify.add_argument("--output", type=Path, help="Write new evidence; never modifies the checkpoint")

    benchmark = groups.add_parser("benchmark", help="Measure batch sizes on simulator RGB")
    benchmark.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--output", type=Path, required=True, help="New benchmark stage directory")
    benchmark.add_argument("--log-dir", type=Path)
    benchmark.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32, 64])
    benchmark.add_argument("--steps", type=int, default=120)
    benchmark.add_argument("--warmup", type=int, default=20)

    evaluate = groups.add_parser("evaluate", help="Extract frozen features and fit the ball-motion probe")
    evaluate.add_argument("--run", type=Path, required=True, help="Experiment or training directory")
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--output", type=Path, help="New evaluation directory")
    evaluate.add_argument("--batch-size", type=int, default=2)

    probe = groups.add_parser("probe", help="Spatial frozen probes and visual-only controls").add_subparsers(dest="action",required=True)
    prepare = probe.add_parser("prepare",help="Freeze labels/cohort and cache two observed RGB frames")
    prepare.add_argument("--dataset",required=True)
    prepare.add_argument("--output",type=Path,required=True)
    extract = probe.add_parser("extract",help="Extract readout, pooled and full frozen patch features")
    extract.add_argument("--dataset",required=True)
    extract.add_argument("--run",type=Path,required=True)
    extract.add_argument("--output",type=Path,required=True)
    extract.add_argument("--cohort",type=Path,required=True)
    extract.add_argument("--weights",choices=["trained","random"],default="trained")
    extract.add_argument("--batch-size",type=int,default=32)
    extract.add_argument("--temporal-mode",choices=["ordered","repeat_last"],default="ordered")
    fit = probe.add_parser("fit",help="Fit identical global and spatial heads to frozen representations")
    fit.add_argument("--cohort",type=Path,required=True)
    fit.add_argument("--features",type=Path,required=True)
    fit.add_argument("--output",type=Path,required=True)
    fit.add_argument("--discard-cache",action="store_true",help="Remove only the new regenerable full patch cache after successful probes")
    supervised = probe.add_parser("supervised",help="Train a small RGB-only supervised CNN visual control")
    supervised.add_argument("--cohort",type=Path,required=True)
    supervised.add_argument("--output",type=Path,required=True)
    supervised.add_argument("--seed",type=int,default=42)
    diagnose = probe.add_parser("diagnose",help="Collapse, crop, temporal and visual-resolution diagnostics")
    diagnose.add_argument("--dataset",required=True)
    diagnose.add_argument("--run",type=Path,required=True)
    report_v2 = groups.add_parser("report-v2",help="Compile the expanded baseline evidence")
    report_v2.add_argument("--run",type=Path,required=True)

    report = groups.add_parser("report", help="Generate reports inside an experiment bundle")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--dataset")
    report.add_argument("--output", type=Path)
    info = groups.add_parser("info", help="Show the latest or a selected experiment without loading ML libraries")
    info.add_argument("--run", type=Path)
    return command


def split_path(value):
    path = dataset_path(value)
    return path / "train" if (path / "train/train.lance").exists() else path


def dispatch(argv=None):
    args = parser().parse_args(argv)
    if args.command == "simulator":
        from . import simulator
        if args.action == "setup":
            return simulator.setup()
        return simulator.record(resolve_path(args.output), match_index=args.session, episodes=args.episodes, samples=args.samples, seed=args.seed)
    if args.command == "dataset":
        if args.action == "expand":
            from .dataset import expand
            return expand(resolve_path(args.output))
        if args.action == "build":
            from .experiment import build_dataset
            return build_dataset(resolve_path(args.config), args.output)
        from . import dataset
        if args.action == "render":
            return dataset.build(dataset_path(args.dataset))
        output = resolve_path(args.output) if args.output else None
        if args.action == "audit":
            return dataset.audit(dataset_path(args.path), output=output)
        return dataset.validate(resolve_path(args.path), output=output)
    if args.command == "run":
        from .experiment import run
        return run(config=resolve_path(args.config), dataset=dataset_path(args.dataset), name=args.name, output=args.output, plan=args.plan)
    if args.command == "baseline-v2":
        from .experiment import baseline_v2
        return baseline_v2(resolve_path(args.config),dataset_path(args.dataset),resolve_path(args.output),seeds=args.seeds,steps=args.steps,plan=args.plan)
    if args.command == "ablate-v2":
        from .ablations import run
        return run(args.baseline, args.output, args.plan)
    if args.command == "train":
        from .training import train
        return train(config=resolve_path(args.config), output=resolve_path(args.output), steps=args.steps, warmup=args.warmup,
                     resume=resolve_path(args.resume) if args.resume else None,
                     dataset=split_path(args.dataset) if args.dataset else None,
                     batch_size=args.batch_size, benchmark=args.benchmark, no_holdout=args.no_holdout)
    if args.command == "checkpoint":
        from .training import verify
        return verify(resolve_path(args.checkpoint), output=resolve_path(args.output) if args.output else None,
                      dataset=split_path(args.dataset) if args.dataset else None)
    if args.command == "benchmark":
        from .benchmark import run
        return run(config=resolve_path(args.config), output=resolve_path(args.output), batch_sizes=args.batches,
                   steps=args.steps, warmup=args.warmup, log_dir=resolve_path(args.log_dir) if args.log_dir else None,
                   dataset=split_path(args.dataset) if args.dataset else None)
    if args.command == "evaluate":
        from .evaluation import evaluate
        location = resolve_path(args.run)
        experiment = location if (location / "training").exists() else location.parent
        training = experiment / "training" if (experiment / "training").exists() else location
        source = dataset_path(args.dataset) if args.dataset else experiment_dataset(experiment)
        return evaluate(source, training, resolve_path(args.output) if args.output else experiment / "evaluation", batch_size=args.batch_size)
    if args.command == "report":
        from .reporting import write
        location = resolve_path(args.run)
        source = dataset_path(args.dataset) if args.dataset else experiment_dataset(location)
        return write(source, location, resolve_path(args.output) if args.output else None)
    if args.command == "probe":
        from . import probes
        if args.action=="prepare":
            return probes.prepare(dataset_path(args.dataset),resolve_path(args.output))
        if args.action=="extract":
            return probes.extract(dataset_path(args.dataset),resolve_path(args.run),resolve_path(args.output),resolve_path(args.cohort),weights=args.weights,batch_size=args.batch_size,temporal_mode=args.temporal_mode)
        if args.action=="fit":
            return probes.fit(resolve_path(args.cohort),resolve_path(args.features),resolve_path(args.output),remove_patch_cache=args.discard_cache)
        if args.action=="diagnose":
            return probes.diagnose(dataset_path(args.dataset),resolve_path(args.run))
        return probes.supervised(resolve_path(args.cohort),resolve_path(args.output),seed=args.seed)
    if args.command == "report-v2":
        from .reporting import write_v2
        return write_v2(resolve_path(args.run))
    if args.command == "info":
        from .experiment import info
        return info(args.run)


def main(argv=None):
    # Console-script launchers pass this return value to sys.exit().
    result = dispatch(argv)
    if result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0
