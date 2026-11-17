"""Experiment bundles, stage orchestration and their dataset references."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .layout import DEFAULT_CONFIG, DEFAULT_DATASET, OUTPUTS, ROOT, dataset_path, experiment_dataset, resolve_path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def paths(output=None, name="sumatra_baseline"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Experiment name must contain letters, numbers, underscores or hyphens")
    return resolve_path(output) if output else OUTPUTS / (datetime.now().strftime("%Y-%m-%d_%H%M%S_") + name)


def prepare_config(source, dataset, output, batch_size=None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(resolve_path(source))
    cfg.dataset.path = str(dataset / "train/train.lance")
    cfg.dataset.episodes_path = str(dataset / "train/train.episodes.npz")
    if cfg.get("validation_dataset"):
        cfg.validation_dataset.path = str(dataset / "validation/train.lance")
        cfg.validation_dataset.episodes_path = str(dataset / "validation/train.episodes.npz")
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    OmegaConf.save(cfg, output)
    return cfg


def stage_commands(experiment, dataset, config):
    executable = [sys.executable, "-m", "ssl_levjepa"]
    stages = []
    for split in ("train", "validation", "test"):
        stages.append(("validate_" + split, executable + ["dataset", "validate", str(dataset / split),
                       "--output", str(experiment / "provenance" / ("preflight_" + split + ".json"))]))
    stages.append(("benchmark", executable + ["benchmark", "--config", str(config),
                   "--output", str(experiment / "benchmark"), "--log-dir", str(experiment / "logs/benchmark")]))
    stages.append(("training", executable + ["train", "--config", str(config), "--output", str(experiment / "training"), "--warmup", "20"]))
    stages.append(("checkpoint", executable + ["checkpoint", "verify", str(experiment / "training/checkpoints/final.ckpt"),
                   "--output", str(experiment / "provenance/checkpoint_verified.json")]))
    stages.append(("evaluation", executable + ["evaluate", "--run", str(experiment), "--output", str(experiment / "evaluation")]))
    stages.append(("report", executable + ["report", "--run", str(experiment)]))
    return stages


def execute(command, log, timeout=1800, env=None):
    """Preserve child exit codes and allow graceful shutdown before forced termination."""
    with Path(log).open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt", env=env)
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGINT)
            else:
                child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(config=DEFAULT_CONFIG, dataset=DEFAULT_DATASET, name="sumatra_baseline", output=None, plan=False):
    config = resolve_path(config)
    dataset = dataset_path(dataset)
    destination = paths(output, name)
    commands = stage_commands(destination, dataset, destination / "config.yaml")
    result = {"experiment": str(destination), "dataset": str(dataset), "stages": [{"stage": s, "command": c} for s, c in commands]}
    if plan:
        return result
    audit = json.loads((dataset / "audit.json").read_text())
    if not audit.get("passed"):
        raise ValueError("Dataset audit has not passed")
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "reports", "provenance"):
        (destination / directory).mkdir()
    source_config = destination / "provenance/source_config.yaml"
    source_config.write_bytes(config.read_bytes())
    prepare_config(config, dataset, destination / "config.yaml")
    write_json(destination / "dataset.json", {
        "id": dataset.name, "path": os.path.relpath(dataset, destination).replace(os.sep, "/"),
        "metadata_sha256": checksum(dataset / "metadata.json"), "clips_sha256": checksum(dataset / "clips.jsonl"),
    })
    state = {"created_at": datetime.now().astimezone().isoformat(), "state": "running", "stages": {}}
    for stage, command in commands:
        state["stages"][stage] = "running"
        write_json(destination / "status.json", state)
        try:
            execute(command, destination / "logs" / (stage + ".log"))
            if stage == "benchmark":
                selected = json.loads((destination / "benchmark/recommendation.json").read_text())["batch_size"]
                prepare_config(source_config, dataset, destination / "config.yaml", selected)
        except BaseException as error:
            state.update(state="failed", failed_stage=stage, error=str(error))
            write_json(destination / "status.json", state)
            raise
        state["stages"][stage] = "complete"
    state["state"] = "complete"
    write_json(destination / "status.json", state)
    write_json(OUTPUTS / "latest.json", {"experiment": destination.name, "path": os.path.relpath(destination, ROOT)})
    return destination


def build_dataset(config=DEFAULT_CONFIG, output=None):
    from . import dataset as data
    from . import simulator
    # Generation currently reproduces the documented 12-session pilot protocol.
    target = dataset_path(output)
    if target.exists():
        raise FileExistsError(f"Dataset already exists and will be preserved: {target}")
    if not resolve_path(config).is_file():
        raise FileNotFoundError(config)
    simulator.generate(target)
    data.build(target)
    data.audit(target, output=target / "audit.json")
    return target


def info(run=None):
    if run is None:
        latest = json.loads((OUTPUTS / "latest.json").read_text(encoding="utf-8-sig"))
        run = latest["path"]
    experiment = resolve_path(run)
    result = {"experiment": str(experiment), "dataset": str(experiment_dataset(experiment)),
              "reports": str(experiment / "reports"), "checkpoint": str(experiment / "training/checkpoints/final.ckpt")}
    protocol_path = experiment/"protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("version")==2 and "seeds" in protocol:
            result.pop("checkpoint")
            result["report"] = str(experiment/"reports/BASELINE_V2_REPORT.md")
            result["checkpoints"] = {str(seed):str(experiment/"seeds"/str(seed)/"training/checkpoints/final.ckpt") for seed in protocol["seeds"]}
            result["training"] = {}
            for seed in protocol["seeds"]:
                summary_path = experiment/"seeds"/str(seed)/"training/summary.json"
                if summary_path.exists():
                    result["training"][str(seed)] = json.loads(summary_path.read_text())
    status = experiment / "status.json"
    if status.exists():
        result["status"] = json.loads(status.read_text())
    summary = experiment / "training/summary.json"
    if summary.exists():
        result["training"] = json.loads(summary.read_text())
    return result


def baseline_v2(config, dataset, output, seeds=(42,43,44), steps=3000, plan=False):
    """Run the declared expanded cohort with paired random controls and supervised visual controls."""
    import shutil
    import threading
    import time
    from .benchmark import _telemetry
    from omegaconf import OmegaConf
    dataset,output,config = dataset_path(dataset),resolve_path(output),resolve_path(config)
    declaration = {"version":2,"dataset":str(dataset),"seeds":list(seeds),"steps":steps,"micro_batch":64,
                   "uniform_keep_fraction":.1,"frozen_extraction_batch":32,"source_config_sha256":checksum(config),
                   "probe_policy":"Same cohort, train-only normalization, alpha/epoch selected on validation, held-out session test.",
                   "primary_probe":"spatial_conv; test component-wise velocity RMSE, paired trained versus saved random initialization",
                   "secondary_probes":["global_readout_ridge","mean_patch_pool_ridge","spatial_patch_ridge","spatial_pointwise"],
                   "planned_diagnostics":["readout covariance/cosine/rank","patch spatial and temporal variance","crop ball visibility","repeat-last-frame temporal control","RGB resolution color-detector check"],
                   "token_retention_ablation":"No motion-aware retention in this experiment"}
    if len(set(seeds))!=len(seeds) or not seeds or steps<=0:
        raise ValueError("Use distinct seeds and a positive step budget")
    executable = [sys.executable,"-m","ssl_levjepa"]
    cohort = output/"evaluation/cohort"
    stages = []
    for split in ("train","validation","test"):
        stages.append(("validate_"+split,executable+["dataset","validate",str(dataset/split),"--output",str(output/f"provenance/preflight_{split}.json")]))
    stages.append(("prepare_probes",executable+["probe","prepare","--dataset",str(dataset),"--output",str(cohort)]))
    for seed in seeds:
        base = output/"seeds"/str(seed)
        stages.append((f"train_{seed}",executable+["train","--config",str(base/"config.yaml"),"--output",str(base/"training"),"--warmup","20"]))
        stages.append((f"verify_{seed}",executable+["checkpoint","verify",str(base/"training/checkpoints/final.ckpt"),"--output",str(base/"checkpoint_verified.json")]))
    for seed in seeds:
        base = output/"seeds"/str(seed)
        for weights in ("random","trained"):
            folder = base/"evaluation"/weights
            stages.append((f"extract_{weights}_{seed}",executable+["probe","extract","--dataset",str(dataset),"--run",str(base/"training"),
                           "--cohort",str(cohort),"--weights",weights,"--batch-size","32","--output",str(folder/"features")]))
            stages.append((f"probe_{weights}_{seed}",executable+["probe","fit","--cohort",str(cohort),"--features",str(folder/"features"),
                           "--output",str(folder/"probes"),"--discard-cache"]))
        folder = base/"evaluation/supervised"
        stages.append((f"supervised_{seed}",executable+["probe","supervised","--cohort",str(cohort),"--seed",str(seed),"--output",str(folder)]))
        stages.append((f"probe_supervised_{seed}",executable+["probe","fit","--cohort",str(cohort),"--features",str(folder/"features"),
                       "--output",str(folder/"probes"),"--discard-cache"]))
    stages.append(("diagnostics",executable+["probe","diagnose","--dataset",str(dataset),"--run",str(output)]))
    stages.append(("report",executable+["report-v2","--run",str(output)]))
    if plan:
        return {"protocol":declaration,"stages":[{"stage":name,"command":command} for name,command in stages]}
    audit = json.loads((dataset/"audit.json").read_text())
    assert audit["passed"] and audit["split_counts_and_motion"]["train"]["clips"]>=10000
    upstream_state = subprocess.check_output(["git","-C",str(ROOT/"upstream/LeVJEPA"),"status","--porcelain"],text=True)
    assert not upstream_state.strip(), "Upstream LeVJEPA must remain unmodified"
    if output.exists() and not (output/"protocol.json").exists():
        assert {p.name for p in output.iterdir()}.issubset({"pilot_diagnostics","study_plan.json"}), "Destination belongs to another experiment"
    output.mkdir(parents=True,exist_ok=True)
    for name in ("logs","reports","provenance"):
        (output/name).mkdir(exist_ok=True)
    protocol = output/"protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text())==declaration, "Resume protocol differs"
    else:
        write_json(protocol,declaration)
        prepare_config(config,dataset,output/"config.yaml",64)
        write_json(output/"dataset.json",{"id":dataset.name,"path":os.path.relpath(dataset,output),
                   "metadata_sha256":checksum(dataset/"metadata.json"),"clips_sha256":checksum(dataset/"clips.jsonl"),
                   "protocol_sha256":checksum(dataset/"protocol.json")})
        snapshot = output/"provenance/source"
        for source in (ROOT/"src/ssl_levjepa").glob("*.py"):
            destination = snapshot/"src/ssl_levjepa"/source.name
            destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,destination)
        write_json(output/"provenance/source_manifest.json",{
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "files":{str(p.relative_to(ROOT)):checksum(p) for p in (ROOT/"src/ssl_levjepa").glob("*.py")},
            "upstream_commit":subprocess.check_output(["git","-C",str(ROOT/"upstream/LeVJEPA"),"rev-parse","HEAD"],text=True).strip()})
        (output/"provenance/working_changes.diff").write_bytes(subprocess.check_output(["git","diff","HEAD","--binary"],cwd=ROOT))
        for seed in seeds:
            base = output/"seeds"/str(seed); base.mkdir(parents=True)
            cfg = prepare_config(config,dataset,base/"config.yaml",64)
            cfg.seed = seed; cfg.trainer.max_steps = steps; cfg.schedule_total_steps = steps
            OmegaConf.save(cfg,base/"config.yaml")
    env = os.environ.copy()
    env["SSL_LEVJEPA_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(output/"provenance/source/src")
    state_path = output/"status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"state":"running","stages":{}}
    for name,command in stages:
        if state["stages"].get(name,{}).get("state")=="complete":
            continue
        state.update(state="running",current_stage=name)
        stage = {"state":"running","command":command,"started_at":datetime.now().astimezone().isoformat()}
        state["stages"][name] = stage; write_json(state_path,state)
        print(f"Starting baseline v2 stage: {name}",flush=True)
        stop = threading.Event(); telemetry = []
        thread = threading.Thread(target=_telemetry,args=(stop,telemetry,output/"logs"/(name+"_gpu.jsonl")),daemon=True)
        thread.start(); started = time.perf_counter()
        try:
            execute(command,output/"logs"/(name+".log"),timeout=7200,env=env)
            if name.startswith("train_"):
                seed = int(name.removeprefix("train_"))
                summary = json.loads((output/"seeds"/str(seed)/"training/summary.json").read_text())
                assert summary["final_step"]==steps and summary["initial_step"]==0 and summary["nonfinite_count"]==0
        except BaseException as error:
            stage.update(state="failed",error=str(error)); state.update(state="failed",failed_stage=name)
            write_json(state_path,state)
            raise
        finally:
            stop.set();thread.join(timeout=10)
        stage.update(state="complete",wall_sec=time.perf_counter()-started,
                     finished_at=datetime.now().astimezone().isoformat(),gpu_telemetry_samples=len(telemetry))
        write_json(state_path,state)
    initial_hashes = [checksum(output/"seeds"/str(seed)/"training/initial_encoder.pt") for seed in seeds]
    assert len(set(initial_hashes))==len(seeds), "Training seeds reused the same encoder initialization"
    state.update(state="complete",current_stage=None);write_json(state_path,state)
    write_json(OUTPUTS/"latest.json",{"experiment":output.name,"path":os.path.relpath(output,ROOT)})
    return output
