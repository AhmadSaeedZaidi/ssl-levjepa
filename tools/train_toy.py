"""Instrument the unmodified upstream data, model, loss and optimizer recipe."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/LeVJEPA"
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from omegaconf import OmegaConf
import stable_pretraining as spt
import torch

import train as upstream
from data import (Episodes, WindowedLanceDataset, VariableLengthBatchSampler,
                  ViewBuilder, identity_collate, holdout_episodes, holdout_batches, CachedBatches)
from model import Encoder, Projector
from loss import SIGReg


class TimedLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        iterator = super().__iter__()
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            batch["data_wait_sec"] = time.perf_counter() - started
            yield batch


class Measurements(pl.Callback):
    def __init__(self, directory, warmup, limit_gb=11):
        self.directory = directory
        self.warmup = warmup
        self.limit = limit_gb * 1024**3
        self.rows = []
        self.grad_norm = None

    def on_train_start(self, trainer, module):
        self.started = time.perf_counter()
        self.initial_step = trainer.global_step
        self.initial_weight = next(module.encoder.parameters()).detach().clone()
        torch.cuda.reset_peak_memory_stats()
        evidence = {"global_step": trainer.global_step, "optimizers": []}
        for optimizer in trainer.optimizers:
            steps = [float(state["step"]) for state in optimizer.state.values() if "step" in state]
            evidence["optimizers"].append({"state_entries": len(optimizer.state),
                                         "min_step": min(steps) if steps else 0,
                                         "max_step": max(steps) if steps else 0})
        (self.directory / "resume_evidence.json").write_text(json.dumps(evidence, indent=2))

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        torch.cuda.synchronize()
        self.batch_started = time.perf_counter()

    def on_before_backward(self, trainer, module, loss):
        self.loss = float(loss.detach())
        if not math.isfinite(self.loss):
            raise FloatingPointError(f"Non-finite loss at optimizer step {trainer.global_step}")

    def on_before_optimizer_step(self, trainer, module, optimizer):
        norms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
        self.grad_norm = float(torch.stack(norms).norm())
        if not math.isfinite(self.grad_norm) or self.grad_norm <= 0:
            raise FloatingPointError(f"Invalid gradient norm {self.grad_norm}")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.perf_counter()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if reserved > self.limit:
            raise RuntimeError(f"Reserved CUDA memory {reserved} exceeds safety threshold {self.limit}")
        if trainer.global_step == self.initial_step + 1:
            if torch.equal(self.initial_weight, next(module.encoder.parameters()).detach()):
                raise RuntimeError("First optimizer step did not change encoder weights")
            if self.grad_norm is None:
                raise RuntimeError("Optimizer hook did not record gradients")
        row = {"optimizer_step": trainer.global_step, "loss": self.loss,
               "gradient_norm": self.grad_norm, "wall_elapsed_sec": now-self.started,
               "compute_sec": now-self.batch_started, "data_wait_sec": batch["data_wait_sec"],
               "clips": int(batch["clip"].shape[0]), "peak_allocated_bytes": allocated,
               "peak_reserved_bytes": reserved}
        self.rows.append(row)
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        if len(self.rows) % 10 == 0 or len(self.rows) == 1:
            print(json.dumps(row), flush=True)

    def summarize(self, trainer, status):
        elapsed = time.perf_counter()-self.started
        measured = self.rows[self.warmup:]
        # End-to-end interval includes input wait, transfer, checkpoint/validation and compute.
        duration = (measured[-1]["wall_elapsed_sec"] - self.rows[self.warmup-1]["wall_elapsed_sec"]
                    if measured and self.warmup else elapsed)
        losses = [row["loss"] for row in self.rows]
        summary = {"status": status, "initial_step": self.initial_step,
                   "final_step": trainer.global_step, "completed_steps": len(self.rows),
                   "wall_sec": elapsed, "gpu_hours": elapsed/3600,
                   "warmup_steps_excluded": self.warmup, "measured_steps": len(measured),
                   "measured_sec": duration, "samples_seen": sum(r["clips"] for r in self.rows),
                   "steps_per_sec": len(measured)/duration if duration else None,
                   "decoded_clips_per_sec": sum(r["clips"] for r in measured)/duration if duration else None,
                   "mean_loss": float(np.mean(losses)) if losses else None,
                   "final_loss": losses[-1] if losses else None,
                   "loss_min": min(losses) if losses else None, "loss_max": max(losses) if losses else None,
                   "nonfinite_count": sum(not math.isfinite(v) for v in losses),
                   "mean_compute_sec": float(np.mean([r["compute_sec"] for r in measured])) if measured else None,
                   "mean_data_wait_sec": float(np.mean([r["data_wait_sec"] for r in measured])) if measured else None,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary), flush=True)


def load_config(path):
    base = OmegaConf.load(UPSTREAM / "configs/config.yaml")
    del base["defaults"]
    base.model = OmegaConf.load(UPSTREAM / "configs/model/vitb.yaml")
    base.dataset = OmegaConf.load(UPSTREAM / "configs/dataset/single.yaml")
    return OmegaConf.merge(base, OmegaConf.load(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/ssl_toy_3060_12gb.yaml")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200, help="absolute final optimizer step")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    cfg.trainer.max_steps = args.steps
    if args.dataset:
        cfg.dataset.path = str(args.dataset.resolve() / "train.lance")
        cfg.dataset.episodes_path = str(args.dataset.resolve() / "train.episodes.npz")
    if args.no_holdout:
        cfg.holdout = None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this measured baseline")
    if cfg.trainer.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        cfg.trainer.precision = "16-mixed"
    torch.set_num_threads(4)
    pl.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg), flush=True)
    OmegaConf.save(cfg, args.run_dir / "config.yaml")
    sha = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    (args.run_dir / "upstream_sha.txt").write_text(sha+"\n")
    episodes = Episodes([OmegaConf.to_container(cfg.dataset)])
    dataset = WindowedLanceDataset([s["path"] for s in episodes.sources], episodes.offsets,
                                   cfg.data.height, cfg.data.width, cfg.dataset.column)
    holdout = cfg.holdout
    held = holdout_episodes(episodes, holdout.videos, 16, cfg.seed) if holdout else None
    sampler = VariableLengthBatchSampler(episodes, cfg.data.batch_size,
                {int(k): v for k, v in cfg.data.probs.items()}, cfg.data.batches_per_epoch,
                seed=cfg.seed, exclude=held)
    loader = TimedLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate,
                         num_workers=cfg.data.num_workers, pin_memory=cfg.data.num_workers > 0,
                         persistent_workers=cfg.data.persistent_workers,
                         multiprocessing_context="forkserver" if cfg.data.num_workers else None)
    if holdout:
        val_loader = torch.utils.data.DataLoader(CachedBatches(copy.copy(dataset)),
                         collate_fn=identity_collate, num_workers=0,
                         batch_sampler=holdout_batches(episodes, held, 16, cfg.data.batch_size, 0, 1))
        data = spt.data.DataModule(train=loader, val=val_loader)
    else:
        data = spt.data.DataModule(train=loader)
    module = spt.Module(forward=upstream.forward, view_builder=ViewBuilder(**cfg.views),
                        encoder=Encoder(**cfg.model),
                        projector=Projector(cfg.model.residual_dim, cfg.projector.hidden, cfg.projector.out),
                        sigreg=SIGReg(), keep_global=upstream.keep_tokens(cfg.masking.keep_global, cfg.views.global_size, cfg.model.patch_size),
                        keep_local=upstream.keep_tokens(cfg.masking.keep_local, cfg.views.local_size, cfg.model.patch_size),
                        iid=cfg.masking.iid, lam=cfg.loss.lam,
                        optim=upstream.build_optim(cfg, cfg.schedule_total_steps))
    params = {"encoder_parameters": sum(p.numel() for p in module.encoder.parameters()),
              "total_parameters": sum(p.numel() for p in module.parameters()),
              "keep_global": module.keep_global, "keep_local": module.keep_local,
              "upstream_sha": sha}
    (args.run_dir / "model.json").write_text(json.dumps(params, indent=2))
    monitor = Measurements(args.run_dir, args.warmup)
    checkpoint = ModelCheckpoint(dirpath=args.run_dir / "checkpoints", filename="step-{step:06d}",
                                 auto_insert_metric_name=False,
                                 **OmegaConf.to_container(cfg.checkpoint))
    trainer = pl.Trainer(**cfg.trainer, default_root_dir=str(args.run_dir),
                         logger=CSVLogger(str(args.run_dir), name="csv"),
                         callbacks=[monitor, checkpoint],
                         **({"val_check_interval": holdout.every_n_steps} if holdout else {}))
    trainer.gradient_clip_val_ = float(cfg.gradient_clip_val)
    trainer.gradient_clip_algorithm_ = "norm"
    status = "error"
    try:
        # Manager adds an automatic saver beside a resume input. Use Lightning directly
        # so a new run never writes into a pre-existing checkpoint's directory.
        trainer.fit(module, datamodule=data,
                    ckpt_path=str(args.resume.resolve()) if args.resume else None, weights_only=False)
        trainer.save_checkpoint(args.run_dir / "checkpoints/final.ckpt")
        status = "reached_steps" if trainer.global_step >= args.steps else "time_or_interrupt"
    finally:
        if hasattr(monitor, "started"):
            monitor.summarize(trainer, status)


if __name__ == "__main__":
    main()
