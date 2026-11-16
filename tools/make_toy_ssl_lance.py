"""CPU-only deterministic toy footage. This is not a RoboCup simulator."""
import argparse
import io
import json
import runpy
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
KIND = "TOY_SMOKE_TEST_NOT_ROBOCUP_PHYSICS"


def render(positions, velocity, phase, step, size):
    image = Image.new("RGB", (size, size), (26, 108, 59))
    draw = ImageDraw.Draw(image)
    scale = size / 256
    def box(values):
        return tuple(round(x * scale) for x in values)
    white = (220, 235, 220)
    draw.rectangle(box((12, 28, 244, 228)), outline=white, width=2)
    draw.line(box((128, 28, 128, 228)), fill=white, width=1)
    draw.ellipse(box((102, 102, 154, 154)), outline=white, width=1)
    draw.rectangle(box((12, 91, 43, 165)), outline=white)
    draw.rectangle(box((213, 91, 244, 165)), outline=white)
    draw.rectangle(box((5, 108, 12, 148)), outline=white)
    draw.rectangle(box((244, 108, 251, 148)), outline=white)
    # Reflected trajectories preserve temporal continuity. No physical contacts are simulated.
    raw = positions + step * velocity + 3 * np.sin(phase + step * .09)
    xy = 23 + (1 - np.abs((raw / 210) % 2 - 1)) * 210
    for index, (x, y) in enumerate(xy):
        r = 3.5 if index == 12 else 7
        color = (255, 120, 18) if index == 12 else ((35, 105, 255) if index < 6 else (255, 219, 35))
        draw.ellipse(box((x-r, y-r, x+r, y+r)), fill=color, outline=(10, 22, 20))
        if index != 12:
            angle = phase[index, 0] + step * .04
            draw.line(box((x, y, x+5*np.cos(angle), y+5*np.sin(angle))), fill=(245, 245, 245), width=2)
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    if min(args.episodes, args.frames, args.size, args.fps) <= 0:
        ap.error("counts, size and FPS must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(generator_version=1, seed=args.seed, episodes=args.episodes,
                    frames_per_episode=args.frames, fps=args.fps, height=args.size,
                    width=args.size, dataset_kind=KIND, frame_count=args.episodes*args.frames)
    schema = pa.schema([
        ("episode_idx", pa.int64()), ("step_idx", pa.int64()), ("frame", pa.binary()),
        ("h", pa.int32()), ("w", pa.int32()), ("label", pa.int64()),
    ], metadata={"toy_ssl": json.dumps(metadata)})
    preview = Image.new("RGB", (4*args.size, 2*(args.size+24)), (22, 25, 30))
    preview_draw = ImageDraw.Draw(preview)

    def batches():
        for episode in range(args.episodes):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode]))
            position = rng.uniform(0, 210, (13, 2))
            velocity = rng.uniform(-1.7, 1.7, (13, 2))
            velocity[-1] *= 2.2
            phase = rng.uniform(0, 2*np.pi, (13, 2))
            blobs = []
            preview_steps = np.linspace(0, args.frames-1, 4, dtype=int).tolist()
            for step in range(args.frames):
                image = render(position, velocity, phase, step, args.size)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90, subsampling=0)
                blobs.append(buffer.getvalue())
                if episode < 2 and step in preview_steps:
                    x, y = preview_steps.index(step)*args.size, episode*(args.size+24)
                    preview.paste(Image.open(io.BytesIO(blobs[-1])), (x, y))
                    preview_draw.text((x+6, y+args.size+5), f"TOY ep={episode} frame={step}", fill="white")
            yield pa.RecordBatch.from_arrays([
                pa.array([episode]*args.frames, type=pa.int64()),
                pa.array(range(args.frames), type=pa.int64()), pa.array(blobs, type=pa.binary()),
                pa.array([args.size]*args.frames, type=pa.int32()),
                pa.array([args.size]*args.frames, type=pa.int32()),
                pa.array([-1]*args.frames, type=pa.int64()),
            ], schema=schema)

    store = args.output / "train.lance"
    lance.write_dataset(pa.RecordBatchReader.from_batches(schema, batches()), str(store), mode="create")
    build = runpy.run_path(str(ROOT / "upstream/LeVJEPA/scripts/build_episodes.py"))["build"]
    build(str(store))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    preview.save(args.output / "preview.png")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
