"""Strictly reload a local trusted checkpoint and infer from a real decoded clip."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/LeVJEPA"))
from omegaconf import OmegaConf
import numpy as np
import torch
from data import Episodes, WindowedLanceDataset, ViewBuilder
from model import Encoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path, help="trusted local training checkpoint")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.checkpoint.parent.parent / "config.yaml")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = Encoder(**cfg.model)
    state = {k.removeprefix("encoder."): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.cuda().eval()
    episodes = Episodes([OmegaConf.to_container(cfg.dataset)])
    rows = episodes.windows(np.array([0]), episodes.starts[:1], 16)
    dataset = WindowedLanceDataset([cfg.dataset.path], episodes.offsets, cfg.data.height, cfg.data.width)
    batch = dataset.__getitems__(rows)
    view_builder = ViewBuilder(**cfg.views).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        global_view, _ = view_builder(batch["clip"].cuda())
        readout, _, _ = encoder(global_view, keep_per_frame=20, iid=cfg.masking.iid)
    assert torch.isfinite(readout).all()
    steps = [float(s["step"]) for s in checkpoint["optimizer_states"][0]["state"].values() if "step" in s]
    assert min(steps) == max(steps) == checkpoint["global_step"]
    digest = hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest()
    evidence = {"checkpoint": str(args.checkpoint), "sha256": digest,
                "global_step": checkpoint["global_step"], "optimizer_state_entries": len(steps),
                "optimizer_step_min": min(steps), "optimizer_step_max": max(steps),
                "scheduler_last_epoch": checkpoint["lr_schedulers"][0]["last_epoch"],
                "strict_encoder_reload": True, "inference_shape": list(readout.shape),
                "inference_finite": True}
    args.checkpoint.with_suffix(".verified.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
