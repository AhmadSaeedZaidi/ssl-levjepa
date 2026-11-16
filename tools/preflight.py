"""Check CUDA and the actual upstream Lance sampler, decoder and view builder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/LeVJEPA"))
import lance
import numpy as np
import torch
from data import Episodes, WindowedLanceDataset, VariableLengthBatchSampler, ViewBuilder, decode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.cuda.get_device_properties(0)
    if "3060" not in device.name or device.total_memory < 11*1024**3:
        raise RuntimeError(f"Unexpected device: {device}")
    path = str(args.dataset / "train.lance")
    index = str(args.dataset / "train.episodes.npz")
    episodes = Episodes([{"path": path, "episodes_path": index}])
    sampler = VariableLengthBatchSampler(episodes, 8, {16: 1.0}, 1, seed=42)
    rows = next(iter(sampler))
    ds = lance.dataset(path)
    attrs = ds.take(rows.flatten().tolist(), columns=["episode_idx", "step_idx"]).to_pydict()
    ids = np.array(attrs["episode_idx"]).reshape(8, 16)
    steps = np.array(attrs["step_idx"]).reshape(8, 16)
    assert np.all(ids == ids[:, :1]) and np.all(np.diff(steps, axis=1) == 1)
    dataset = WindowedLanceDataset([path], episodes.offsets, 256, 256)
    batch = dataset.__getitems__(rows)
    assert batch["clip"].shape == (8, 16, 256, 256, 3) and batch["clip"].dtype == torch.uint8
    for t, row in enumerate(rows[0]):
        blob = ds.take([int(row)], columns=["frame"]).column(0)[0].as_py()
        assert np.array_equal(batch["clip"][0, t].numpy(), decode(blob, 256, 256))
    builder = ViewBuilder(num_local=4, global_scale=(.8, 1), local_scale=(.02, .4)).cuda()
    g, local = builder(batch["clip"].cuda())
    assert g.shape == (8, 16, 3, 224, 224) and local.shape == (4, 8, 16, 3, 96, 96)
    assert torch.isfinite(g).all() and torch.isfinite(local).all()
    # ViewBuilder's upstream ImageNet normalization maps [0,1] into these channel limits.
    assert float(g.min()) >= -2.118 and float(g.max()) <= 2.641
    result = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "device": device.name, "vram_bytes": device.total_memory,
              "bf16_supported": torch.cuda.is_bf16_supported(),
              "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
              "dataset": str(args.dataset), "episodes": len(episodes.starts),
              "rows": ds.count_rows(), "clip_shape": list(batch["clip"].shape),
              "clip_dtype": str(batch["clip"].dtype), "global_shape": list(g.shape),
              "local_shape": list(local.shape), "normalized_dtype": str(g.dtype),
              "normalized_range": [float(g.min()), float(g.max())],
              "episode_boundary_and_temporal_order": "passed", "all_16_frames_match_decoder": True}
    (args.dataset / "preflight.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
