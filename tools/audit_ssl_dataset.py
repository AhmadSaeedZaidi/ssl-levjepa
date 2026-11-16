"""Audit state/clip labels, session splits, rollout repeatability and preserved checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import lance
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,default=ROOT/"data/ssl_sumatra_pilot_v2")
    args=ap.parse_args()
    metadata=json.loads((args.dataset/"metadata.json").read_text())
    assert metadata["dataset_kind"]=="SUMATRA_NATIVE_AI_VS_AI_IDEALIZED_PHYSICS_NOT_REAL_RGB"
    samples={}; reference=[]; tick_count=0; seeds=set()
    for path in sorted((args.dataset/"trajectories").glob("session_*.jsonl")):
        with path.open("rb") as binary:
            assert hashlib.file_digest(binary,"sha256").hexdigest()==metadata["trajectory_sha256"][path.name]
        with path.open() as stream:
            for line in stream:
                row=json.loads(line); tick_count+=1
                if row["episode_id"]==0: reference.append(row)
                if row["sample_idx"]>=0:
                    samples[(row["episode_id"],row["sample_idx"])]=row; seeds.add(row["seed"])
    assert len(samples)==9216 and len(seeds)==72 and tick_count==72*848
    clips=[json.loads(line) for line in (args.dataset/"clips.jsonl").read_text().splitlines()]
    stores={s:lance.dataset(str(args.dataset/s/"train.lance")) for s in ["train","validation","test"]}
    episode_sets={s:set() for s in stores}; session_sets={s:set() for s in stores}; hashes={}; duplicates=[]
    for clip in clips:
        keys=stores[clip["split"]].take(list(range(clip["start_row"],clip["start_row"]+16)),columns=["episode_idx","step_idx","timestamp_ns","frame"]).to_pydict()
        assert set(keys["episode_idx"])=={clip["episode_id"]}
        assert np.all(np.diff(keys["step_idx"])==1)
        assert keys["timestamp_ns"][-1]==clip["target_timestamp_ns"]
        actual=samples[(clip["episode_id"],keys["step_idx"][-1])]
        assert actual["ball"]["velocity_mps"][:2]==clip["target_vxy_mps"]
        assert hashlib.sha256(b"".join(keys["frame"])).hexdigest()==clip["rgb_sha256"]
        episode_sets[clip["split"]].add(clip["episode_id"]); session_sets[clip["split"]].add(clip["match_id"])
        if clip["rgb_sha256"] in hashes and hashes[clip["rgb_sha256"]]!=clip["split"]:
            duplicates.append(clip["rgb_sha256"])
        hashes[clip["rgb_sha256"]]=clip["split"]
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert episode_sets[a].isdisjoint(episode_sets[b]) and session_sets[a].isdisjoint(session_sets[b])
    assert not duplicates, "Exact RGB clip duplicated across splits"
    replay_path=args.dataset/"replay_check/session_000.jsonl"
    replay=[json.loads(line) for line in replay_path.read_text().splitlines()]
    assert len(reference)==len(replay)==848
    differences=[i for i,(a,b) in enumerate(zip(reference,replay)) if a!=b]
    delta=np.array([a["ball"]["position_m"] for a in reference])-np.array([b["ball"]["position_m"] for b in replay])
    old_checkpoint=ROOT/"runs/toy_vit_tiny_20261116_045716/checkpoints/final.ckpt"
    with old_checkpoint.open("rb") as stream:
        old_hash=hashlib.file_digest(stream,"sha256").hexdigest()
    assert old_hash=="dfbb3a94d1b8c5e0b6df0035a94c3a1e4901eebf576e1e418363d25c5988e75f"
    result={"passed":True,"physics_ticks":tick_count,"rgb_frames":len(samples),"clip_labels_checked":len(clips),
            "cross_split_episode_overlap":0,"cross_split_session_overlap":0,"cross_split_duplicate_clips":len(duplicates),
            "rollout_repeat":{"compared_ticks":848,"identical_record_values":not differences,
                              "first_different_tick":differences[0]+1 if differences else None,
                              "max_ball_position_difference_m":float(np.linalg.norm(delta,axis=1).max())},
            "original_toy_checkpoint_sha256":old_hash,"original_checkpoint_preserved":True}
    (args.dataset/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
