"""Build, audit, and validate the recorded simulator dataset."""
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import subprocess

from .layout import CACHE, ROOT, UPSTREAM, TOY_CHECKPOINT, enable_upstream, resolve_path

enable_upstream()

import lance
import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw

KIND = "SUMATRA_NATIVE_AI_VS_AI_IDEALIZED_PHYSICS_NOT_REAL_RGB"

def render(row, size=256):
    supersample = 3
    canvas = Image.new("RGB", (size*supersample, size*supersample), (24, 104, 55))
    draw = ImageDraw.Draw(canvas)
    length, width = row["field_size_m"]
    scale = (size-24)/length
    def point(x, y, z=0):
        return ((size/2+x*scale)*supersample, (size/2-y*scale-z*scale*.15)*supersample)
    def rectangle(x0,y0,x1,y1, **kwargs):
        a,b=point(x0,y1), point(x1,y0)
        draw.rectangle((*a,*b), **kwargs)
    def circle(x,y,r, **kwargs):
        a,b=point(x-r,y+r),point(x+r,y-r)
        draw.ellipse((*a,*b), **kwargs)
    white=(226,236,222)
    rectangle(-length/2,-width/2,length/2,width/2,outline=white,width=2*supersample)
    draw.line([point(0,-width/2),point(0,width/2)],fill=white,width=supersample)
    circle(0,0,.5,outline=white,width=supersample)
    for side in [-1,1]:
        x=side*length/2
        rectangle(min(x,x-side*1),-1,max(x,x-side*1),1,outline=white,width=supersample)
        rectangle(min(x,x+side*.18),-.5,max(x,x+side*.18),.5,outline=white,width=supersample)
    for robot in sorted(row["robots"], key=lambda b:(b["team"],b["id"])):
        x,y,heading=robot["pose_m_rad"]
        circle(x,y,.09,fill=(35,100,250) if robot["team"]=="BLUE" else (255,218,32),outline=(8,20,20),width=supersample)
        draw.line([point(x+.065*math.cos(heading+.6),y+.065*math.sin(heading+.6)),
                   point(x+.065*math.cos(heading-.6),y+.065*math.sin(heading-.6))],fill="white",width=supersample)
    x,y,z=row["ball"]["position_m"]
    # 1.5px marker deliberately preserves visibility at 256px. It is not a calibrated camera image.
    radius=max(.0215*scale,1.5)*supersample
    px,py=point(x,y,z)
    draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill=(255,113,12))
    return canvas.resize((size,size),Image.Resampling.LANCZOS)


def encoded(row):
    buffer=io.BytesIO()
    render(row).save(buffer,format="JPEG",quality=95,subsampling=0)
    return buffer.getvalue()


def visible(row):
    length=row["field_size_m"][0]
    scale=232/length
    x,y,z=row["ball"]["position_m"]
    return abs(x*scale)<124 and abs((y+z*.15)*scale)<124


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()



def build(dataset: Path):
    """Render recorded trajectories into fresh derived Lance stores."""
    dataset=resolve_path(dataset)
    if (dataset/"protocol.json").exists():
        return render_expanded(dataset)
    if not dataset.is_dir() or not (dataset/"trajectories").is_dir():
        raise FileNotFoundError(f"Expected an existing trajectory folder: {dataset}")
    derived=[dataset/name for name in ("train","validation","test","clips.jsonl","metadata.json","previews")]
    existing=[str(p) for p in derived if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite derived dataset artifacts: "+", ".join(existing))
    files=sorted((dataset/"trajectories").glob("session_*.jsonl"))
    assert len(files)==12, f"Expected 12 complete sessions, got {len(files)}"
    split_sessions={"train":list(range(8)),"validation":[8,9],"test":[10,11]}
    frames={k:[] for k in split_sessions}
    stats={"episodes":0,"physics_ticks":0,"contact_frames":0,"kick_request_frames":0,"turning_episodes":0,"max_ball_z_m":0}
    all_episodes={}
    hashes={}
    for session,path in enumerate(files):
        raw=[json.loads(line) for line in path.read_text().splitlines()]
        stats["physics_ticks"]+=len(raw)
        for e in range(6):
            ticks=[r for r in raw if r["episode_id"]==session*6+e]
            assert ticks and np.all(np.diff([r["timestamp_ns"] for r in ticks])==10_000_000)
            assert all(r["tick_idx"]==i+1 for i,r in enumerate(ticks))
            assert all("target_velocity_local_mmps_radps" in b["action_applied_last_10ms"] for r in ticks for b in r["robots"])
        rows=[r for r in raw if r["sample_idx"]>=0]
        assert len(rows)==6*128
        hashes[path.name]=digest(path)
        split=next(k for k,v in split_sessions.items() if session in v)
        for e in range(6):
            episode=rows[e*128:(e+1)*128]
            assert {r["episode_id"] for r in episode}=={session*6+e}
            assert [r["sample_idx"] for r in episode]==list(range(128))
            ts=np.array([r["timestamp_ns"] for r in episode])
            assert set(np.diff(ts)).issubset({60_000_000,70_000_000})
            vel=np.array([r["ball"]["velocity_mps"] for r in episode])
            positions=np.array([r["ball"]["position_m"] for r in episode])
            assert np.isfinite(vel).all() and np.isfinite(positions).all()
            assert all(len(r["robots"])==12 for r in episode)
            for r in episode:
                for robot in r["robots"]:
                    assert np.isfinite(robot["pose_m_rad"]).all() and np.isfinite(robot["velocity_mps_radps"]).all()
            speeds=np.linalg.norm(vel[:,:2],axis=1)
            directions=vel[:,:2]/np.maximum(speeds[:,None],1e-9)
            changes=(directions[1:]*directions[:-1]).sum(axis=1)
            turns=(changes<math.cos(.2)) & (speeds[1:]>.15) & (speeds[:-1]>.15)
            stats["turning_episodes"]+=int(turns.any())
            stats["episodes"]+=1
            stats["contact_frames"]+=sum(any(b["ball_contact"] for b in r["robots"]) for r in episode)
            stats["kick_request_frames"]+=sum(any(b["action_applied_last_10ms"]["kick_speed_mps"]>0 for b in r["robots"]) for r in episode)
            stats["max_ball_z_m"]=max(stats["max_ball_z_m"],float(positions[:,2].max()))
            all_episodes[session*6+e]=episode
            frames[split].append(episode)
    assert stats["contact_frames"]>10 and stats["turning_episodes"]>5, "Pilot lacks varied interactions"
    metadata={"dataset_kind":KIND,"generator_seed":20261116,"render_version":1,"size":[256,256],
              "fps_nominal":15,"source_tick_ns":10_000_000,"source_physics_substep_ns":1_000_000,
              "sample_intervals_ns":[60_000_000,70_000_000],"split_sessions":split_sessions,
              "ball_marker_min_radius_px":1.5,"trajectory_sha256":hashes,"stats":stats,
              "sumatra_source_sha":subprocess.check_output(["git","-C",str(CACHE/"Sumatra"),"rev-parse","HEAD"],text=True).strip(),
              "sumatra_release_zip_sha256":digest(CACHE/"sumatra.zip")}
    metadata["recorder_sha256"]=digest(ROOT/"simulation/sumatra/java/SumatraRecorder.java")
    metadata["runtime_moduli_sha256"]=digest(CACHE/"sumatra-dist/sumatra/config/moduli/ssl_record.xml")
    metadata["java_version"]=subprocess.run(["java","-version"],capture_output=True,text=True).stderr.strip()
    metadata["split_overlap_episodes"]=0
    metadata["action_units"]={"target_position_mm_rad":"mm,mm,rad","target_velocity_local_mmps_radps":"mm/s,mm/s,rad/s","kick_speed_mps":"m/s","dribbler_rpm":"rpm"}
    schema=pa.schema([("episode_idx",pa.int64()),("step_idx",pa.int64()),("timestamp_ns",pa.int64()),
                      ("frame",pa.binary()),("h",pa.int32()),("w",pa.int32()),("label",pa.int64())],
                      metadata={"ssl_dataset":json.dumps(metadata)})
    clips=[]
    build=runpy.run_path(str(UPSTREAM/"scripts/build_episodes.py"))["build"]
    for split,episodes in frames.items():
        store_dir=dataset/split
        store_dir.mkdir(exist_ok=False)
        def batches():
            row_start=0
            for episode in episodes:
                blobs=[encoded(r) for r in episode]
                n=len(episode)
                for start in range(0,n-15,16):
                    end=episode[start+15]
                    clips.append({"split":split,"episode_id":end["episode_id"],"match_id":end["match_id"],
                                  "start_row":row_start+start,"length":16,"target_timestamp_ns":end["timestamp_ns"],
                                  "target_vxy_mps":end["ball"]["velocity_mps"][:2],
                                  "rgb_sha256":hashlib.sha256(b"".join(blobs[start:start+16])).hexdigest(),
                                  "visible_frames":sum(visible(r) for r in episode[start:start+16])})
                yield pa.RecordBatch.from_arrays([pa.array([r["episode_id"] for r in episode],type=pa.int64()),
                    pa.array(range(n),type=pa.int64()),pa.array([r["timestamp_ns"] for r in episode],type=pa.int64()),
                    pa.array(blobs,type=pa.binary()),pa.array([256]*n,type=pa.int32()),pa.array([256]*n,type=pa.int32()),
                    pa.array([-1]*n,type=pa.int64())],schema=schema)
                row_start+=n
        store=store_dir/"train.lance"
        lance.write_dataset(pa.RecordBatchReader.from_batches(schema,batches()),str(store),mode="create")
        build(str(store))
        # Re-encode selected frames to test deterministic rendering and exact row/label alignment.
        ds=lance.dataset(str(store))
        for e,episode in enumerate(episodes):
            for t in [0,63,127]:
                result=ds.take([e*128+t]).to_pylist()[0]
                assert result["episode_idx"]==episode[t]["episode_id"] and result["timestamp_ns"]==episode[t]["timestamp_ns"]
                assert result["frame"]==encoded(episode[t])
    (dataset/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in clips))
    metadata["split_counts"]={k:{"episodes":len(v),"frames":len(v)*128,"clips":sum(c["split"]==k for c in clips),
                                  "visible_clips":sum(c["split"]==k and c["visible_frames"]==16 for c in clips)} for k,v in frames.items()}
    metadata["render_and_label_alignment_checks"]="216 exact stored-byte, episode and timestamp checks passed"
    (dataset/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    sheet=Image.new("RGB",(4*256,2*280),(20,24,29)); draw=ImageDraw.Draw(sheet)
    for i,e in enumerate([0,2]):
        episode=all_episodes[e]
        for j,t in enumerate([0,31,63,95]):
            frame=Image.open(io.BytesIO(encoded(episode[t])))
            sheet.paste(frame,(j*256,i*280)); draw.text((j*256+5,i*280+260),f"ep {e}, t={episode[t]['timestamp_ns']/1e9:.2f}s",fill="white")
    previews=dataset/"previews"
    previews.mkdir()
    sheet.save(previews/"preview.png")
    sequence=[render(r) for r in all_episodes[0][:16]]
    sequence[0].save(previews/"clip16.gif",save_all=True,append_images=sequence[1:],duration=67,loop=0)
    return metadata


def audit(dataset: Path, output: Path | None = None) -> dict:
    """Audit stored dataset integrity without writing unless output is explicit."""
    dataset=resolve_path(dataset)
    if (dataset/"protocol.json").exists():
        return audit_expanded(dataset, output)
    if output is not None:
        output=resolve_path(output)
        if output.exists():
            raise FileExistsError(f"Evidence already exists and will be preserved: {output}")
    import lance
    import numpy as np
    metadata=json.loads((dataset/"metadata.json").read_text())
    assert metadata["dataset_kind"]=="SUMATRA_NATIVE_AI_VS_AI_IDEALIZED_PHYSICS_NOT_REAL_RGB"
    samples={}; reference=[]; tick_count=0; seeds=set()
    for path in sorted((dataset/"trajectories").glob("session_*.jsonl")):
        with path.open("rb") as binary:
            assert hashlib.file_digest(binary,"sha256").hexdigest()==metadata["trajectory_sha256"][path.name]
        with path.open() as stream:
            for line in stream:
                row=json.loads(line); tick_count+=1
                if row["episode_id"]==0: reference.append(row)
                if row["sample_idx"]>=0:
                    samples[(row["episode_id"],row["sample_idx"])]=row; seeds.add(row["seed"])
    assert len(samples)==9216 and len(seeds)==72 and tick_count==72*848
    clips=[json.loads(line) for line in (dataset/"clips.jsonl").read_text().splitlines()]
    stores={s:lance.dataset(str(dataset/s/"train.lance")) for s in ["train","validation","test"]}
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
    replay_path=dataset/"replay_check/session_000.jsonl"
    replay=[json.loads(line) for line in replay_path.read_text().splitlines()]
    assert len(reference)==len(replay)==848
    differences=[i for i,(a,b) in enumerate(zip(reference,replay)) if a!=b]
    delta=np.array([a["ball"]["position_m"] for a in reference])-np.array([b["ball"]["position_m"] for b in replay])
    old_hash=None
    original_checkpoint_preserved=None
    if TOY_CHECKPOINT.exists():
        with TOY_CHECKPOINT.open("rb") as stream:
            old_hash=hashlib.file_digest(stream,"sha256").hexdigest()
        assert old_hash=="dfbb3a94d1b8c5e0b6df0035a94c3a1e4901eebf576e1e418363d25c5988e75f"
        original_checkpoint_preserved=True
    result={"passed":True,"physics_ticks":tick_count,"rgb_frames":len(samples),"clip_labels_checked":len(clips),
            "cross_split_episode_overlap":0,"cross_split_session_overlap":0,"cross_split_duplicate_clips":len(duplicates),
            "rollout_repeat":{"compared_ticks":848,"identical_record_values":not differences,
                              "first_different_tick":differences[0]+1 if differences else None,
                              "max_ball_position_difference_m":float(np.linalg.norm(delta,axis=1).max())},
            "original_toy_checkpoint_sha256":old_hash,"original_checkpoint_preserved":original_checkpoint_preserved}
    if output is not None:
        output=Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def _recorded_rows(path):
    import gzip
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def expand(dataset: Path):
    """Record a preregistered 10,240-training-clip cohort, with resumable session boundaries."""
    from datetime import datetime, timezone
    import random
    import shutil
    import time
    from . import simulator
    from .experiment import write_json
    dataset = resolve_path(dataset)
    dataset.mkdir(parents=True, exist_ok=True)
    protocol_path = dataset/"protocol.json"
    if not protocol_path.exists():
        if any(dataset.iterdir()):
            raise FileExistsError(f"Nonempty destination without a protocol: {dataset}")
        order = list(range(80))
        random.Random(20261117).shuffle(order)
        split_sessions = {"train": sorted(order[:64]), "validation": sorted(order[64:72]), "test": sorted(order[72:])}
        write_json(protocol_path, {
            "version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "base_seed": 20261117,
            "sessions": 80, "episodes_per_session": 20, "frames_per_episode": 128, "fps": 30,
            "clip_length": 16, "clip_stride": 16, "split_sessions": split_sessions,
            "scenario_families": ["kick", "interception", "pursuit", "passing", "collision"],
            "scenario_interpretation": "Randomized starting affordances; both native AIs decide all subsequent actions. A passing start does not guarantee a completed pass.",
            "regimes_mps": {"stationary": [0, .05], "slow": [.05, .5], "moving": [.5, 2], "fast": [2, None]},
            "heldout_policy": "Session assignments fixed before recording. Existing pilot validation/test remain untouched and excluded from this cohort.",
            "recorder_sha256": digest(simulator.JAVA_SOURCE),
        })
    protocol = json.loads(protocol_path.read_text())
    assert protocol["version"] == 2 and digest(simulator.JAVA_SOURCE) == protocol["recorder_sha256"]
    trajectories = dataset/"trajectories"
    trajectories.mkdir(exist_ok=True)
    logs = dataset/"logs"
    logs.mkdir(exist_ok=True)
    recording = dataset/"recording.json"
    completed = json.loads(recording.read_text()) if recording.exists() else {}
    simulator._compile()
    for session in range(protocol["sessions"]):
        name = f"v2_session_{session:03d}"
        path = trajectories/(name+".jsonl.gz")
        if name in completed:
            assert path.exists() and digest(path) == completed[name]["sha256"]
            continue
        if path.exists():
            raise FileExistsError(f"Incomplete session preserved; inspect it before resuming: {path}")
        if shutil.disk_usage(dataset).free < 6*1024**3:
            raise RuntimeError("Less than 6 GiB free; completed sessions preserved")
        started = time.perf_counter()
        with (logs/(name+".log")).open("x") as log:
            simulator._record_one(path, session, protocol["episodes_per_session"], protocol["frames_per_episode"],
                                  protocol["base_seed"], timeout=600, log_stream=log,
                                  expanded=True, fps=protocol["fps"], compile=False)
        completed[name] = {"sha256": digest(path), "bytes": path.stat().st_size, "wall_sec": time.perf_counter()-started}
        write_json(recording, completed)
        print(f"Recorded {session+1}/{protocol['sessions']} sessions: {completed[name]['wall_sec']:.1f}s, {path.stat().st_size/1e6:.1f} MB compressed", flush=True)
    replay = dataset/"replay_check"
    replay.mkdir(exist_ok=True)
    if not (replay/"completed.json").exists():
        with (logs/"replay.log").open("x") as log:
            simulator._record_one(replay/"v2_session_000.jsonl.gz", 0, 1, protocol["frames_per_episode"],
                                  protocol["base_seed"], timeout=180, log_stream=log,
                                  expanded=True, fps=protocol["fps"], compile=False)
        write_json(replay/"completed.json", {"sha256": digest(replay/"v2_session_000.jsonl.gz")})
    if not (dataset/"metadata.json").exists():
        render_expanded(dataset)
    if not (dataset/"audit.json").exists():
        audit_expanded(dataset, dataset/"audit.json")
    return dataset


def render_expanded(dataset: Path):
    """Stream compressed simulator sessions into the same upstream-compatible Lance schema."""
    from concurrent.futures import ProcessPoolExecutor
    from collections import Counter
    from .experiment import write_json
    protocol = json.loads((dataset/"protocol.json").read_text())
    for name in ("train", "validation", "test", "clips.jsonl"):
        if (dataset/name).exists():
            raise FileExistsError(f"Derived output exists: {dataset/name}")
    samples = protocol["frames_per_episode"]
    metadata = {"version": 2, "dataset_kind": KIND, "render_version": 1, "size": [256,256],
                "fps_nominal": protocol["fps"], "protocol_sha256": digest(dataset/"protocol.json"),
                "recorder_sha256": protocol["recorder_sha256"], "split_sessions": protocol["split_sessions"],
                "trajectory_sha256": {}, "split_counts": {}, "scenario_counts": {},
                "stats": {"physics_ticks":0, "contact_frames":0, "kick_request_frames":0},
                "sumatra_source_sha": subprocess.check_output(["git","-C",str(CACHE/"Sumatra"),"rev-parse","HEAD"],text=True).strip(),
                "sumatra_release_zip_sha256": digest(CACHE/"sumatra.zip"),
                "runtime_moduli_sha256": digest(CACHE/"sumatra-dist/sumatra/config/moduli/ssl_record.xml")}
    schema = pa.schema([("episode_idx",pa.int64()),("step_idx",pa.int64()),("timestamp_ns",pa.int64()),
                        ("frame",pa.binary()),("h",pa.int32()),("w",pa.int32()),("label",pa.int64())],
                       metadata={"ssl_dataset": json.dumps({"dataset_kind":KIND,"version":2})})
    index_builder = runpy.run_path(str(UPSTREAM/"scripts/build_episodes.py"))["build"]
    clips = []
    previews = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for split, sessions in protocol["split_sessions"].items():
            folder = dataset/split
            folder.mkdir()
            counts = Counter()
            def batches():
                start_row = 0
                for session in sessions:
                    source = dataset/"trajectories"/f"v2_session_{session:03d}.jsonl.gz"
                    metadata["trajectory_sha256"][source.name] = digest(source)
                    rows = []
                    previous = {}
                    for row in _recorded_rows(source):
                        ep = row["episode_id"]
                        assert row["timestamp_ns"]-previous.get(ep,row["timestamp_ns"]-10_000_000) == 10_000_000
                        previous[ep] = row["timestamp_ns"]
                        metadata["stats"]["physics_ticks"] += 1
                        if row["sample_idx"] >= 0:
                            rows.append(row)
                    assert len(rows) == protocol["episodes_per_session"]*samples
                    for e in range(protocol["episodes_per_session"]):
                        episode = rows[e*samples:(e+1)*samples]
                        assert {r["episode_id"] for r in episode} == {session*1_000_000+e}
                        assert [r["sample_idx"] for r in episode] == list(range(samples))
                        blobs = list(pool.map(encoded, episode, chunksize=8))
                        family = episode[0]["scenario_family"]
                        counts[family] += 1
                        for r in episode:
                            assert len(r["robots"]) == 12 and np.isfinite(r["ball"]["position_m"]+r["ball"]["velocity_mps"]).all()
                            metadata["stats"]["contact_frames"] += int(any(b["ball_contact"] for b in r["robots"]))
                            metadata["stats"]["kick_request_frames"] += int(any(b["action_applied_last_10ms"]["kick_speed_mps"]>0 for b in r["robots"]))
                        for start in range(0,samples-15,16):
                            last, previous_row = episode[start+15], episode[start+14]
                            x,y,z = last["ball"]["position_m"]
                            scale = 232/last["field_size_m"][0]
                            clips.append({"split":split,"episode_id":last["episode_id"],"match_id":last["match_id"],
                                          "scenario_family":family,"seed":last["seed"],"start_row":start_row+start,"length":16,
                                          "target_timestamp_ns":last["timestamp_ns"],"target_vxy_mps":last["ball"]["velocity_mps"][:2],
                                          "target_xy_m":[x,y],"target_image_xy_px":[128+x*scale,128-(y+.15*z)*scale],
                                          "target_z_m":z,"target_acceleration_mps2":last["ball"]["acceleration_mps2"],
                                          "dt_last_sec":(last["timestamp_ns"]-previous_row["timestamp_ns"])/1e9,
                                          "visible_frames":sum(visible(r) for r in episode[start:start+16]),
                                          "visible_last_two":all(visible(r) for r in episode[start+14:start+16]),
                                          "rgb_sha256":hashlib.sha256(b"".join(blobs[start:start+16])).hexdigest()})
                        if len(previews)<5 and family not in [r["scenario_family"] for r in previews]:
                            previews.append(episode[0])
                        yield pa.RecordBatch.from_arrays([pa.array([r["episode_id"] for r in episode],type=pa.int64()),
                              pa.array(range(samples),type=pa.int64()),pa.array([r["timestamp_ns"] for r in episode],type=pa.int64()),
                              pa.array(blobs,type=pa.binary()),pa.array([256]*samples,type=pa.int32()),pa.array([256]*samples,type=pa.int32()),
                              pa.array([-1]*samples,type=pa.int64())],schema=schema)
                        start_row += samples
                    print(f"Rendered {split} session {session}", flush=True)
            store = folder/"train.lance"
            lance.write_dataset(pa.RecordBatchReader.from_batches(schema,batches()),str(store),mode="create",max_rows_per_file=16384)
            index_builder(str(store))
            n = len(sessions)*protocol["episodes_per_session"]
            metadata["split_counts"][split] = {"episodes":n,"frames":n*samples,"clips":n*samples//16}
            metadata["scenario_counts"][split] = dict(counts)
    (dataset/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in clips))
    write_json(dataset/"metadata.json", metadata)
    folder = dataset/"previews"
    folder.mkdir()
    sheet = Image.new("RGB",(256*5,280))
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(previews):
        sheet.paste(render(row),(i*256,0)); draw.text((i*256+8,260),row["scenario_family"],fill="white")
    sheet.save(folder/"scenarios.png")
    return metadata


def audit_expanded(dataset: Path, output: Path | None = None):
    """Audit all recorded targets, immutable split assignments and rendered clip hashes."""
    from collections import Counter
    from .experiment import write_json
    dataset = resolve_path(dataset)
    output = resolve_path(output) if output is not None else None
    if output is not None and output.exists():
        raise FileExistsError(output)
    protocol = json.loads((dataset/"protocol.json").read_text())
    meta = json.loads((dataset/"metadata.json").read_text())
    assert digest(dataset/"protocol.json") == meta["protocol_sha256"]
    clips = [json.loads(line) for line in (dataset/"clips.jsonl").read_text().splitlines()]
    samples = {}; first_states = {}; ticks = 0
    for filename, checksum in meta["trajectory_sha256"].items():
        source = dataset/"trajectories"/filename
        assert digest(source) == checksum
        previous = {}
        for row in _recorded_rows(source):
            ep = row["episode_id"]
            assert row["timestamp_ns"]-previous.get(ep,row["timestamp_ns"]-10_000_000) == 10_000_000
            previous[ep] = row["timestamp_ns"]
            assert len(row["robots"]) == 12
            ticks += 1
            if row["sample_idx"] >= 0:
                samples[(ep,row["sample_idx"])] = (row["timestamp_ns"],row["ball"])
                if row["sample_idx"] == 0:
                    first_states[ep] = hashlib.sha256(json.dumps({"ball":row["ball"],"robots":[[b["team"],b["id"],b["pose_m_rad"]] for b in row["robots"]]},sort_keys=True).encode()).hexdigest()
    stores = {s:lance.dataset(str(dataset/s/"train.lance")) for s in protocol["split_sessions"]}
    episodes = {s:set() for s in stores}; sessions = {s:set() for s in stores}; seeds = {s:set() for s in stores}
    hashes = {}; duplicates = 0; render_checks = 0
    for clip in clips:
        split = clip["split"]
        values = stores[split].take(list(range(clip["start_row"],clip["start_row"]+16))).to_pydict()
        assert set(values["episode_idx"]) == {clip["episode_id"]} and np.all(np.diff(values["step_idx"])==1)
        assert set(values["label"]) == {-1}
        source_t,ball = samples[(clip["episode_id"],values["step_idx"][-1])]
        assert source_t == clip["target_timestamp_ns"] == values["timestamp_ns"][-1]
        assert ball["velocity_mps"][:2] == clip["target_vxy_mps"] and ball["position_m"][:2] == clip["target_xy_m"]
        assert hashlib.sha256(b"".join(values["frame"])).hexdigest() == clip["rgb_sha256"]
        assert clip["match_id"] == f"v2_session_{clip['episode_id']//1_000_000:03d}"
        assert clip["episode_id"]//1_000_000 in protocol["split_sessions"][split]
        if clip["rgb_sha256"] in hashes:
            duplicates += 1
            assert hashes[clip["rgb_sha256"]] == split, "Cross-split duplicate RGB"
        hashes[clip["rgb_sha256"]] = split
        episodes[split].add(clip["episode_id"]); sessions[split].add(clip["match_id"]); seeds[split].add(clip["seed"])
        if values["step_idx"][0] == 0:
            raw_ball = samples[(clip["episode_id"],0)][1]
            # Reconstruct only the checks from source records below; never synthesize dynamics.
            assert np.isfinite(raw_ball["position_m"]).all()
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert episodes[a].isdisjoint(episodes[b]) and sessions[a].isdisjoint(sessions[b]) and seeds[a].isdisjoint(seeds[b])
    state_duplicates = len(first_states)-len(set(first_states.values()))
    assert state_duplicates == 0, "Repeated initialization state"
    replay = list(_recorded_rows(dataset/"replay_check/v2_session_000.jsonl.gz"))
    reference = []
    for row in _recorded_rows(dataset/"trajectories/v2_session_000.jsonl.gz"):
        if row["episode_id"] != 0:
            break
        reference.append(row)
    assert len(replay) == len(reference)
    differences = [i for i,(a,b) in enumerate(zip(reference,replay)) if a != b]
    delta = np.array([a["ball"]["position_m"] for a in reference])-np.array([b["ball"]["position_m"] for b in replay])
    distribution = {}
    for split in stores:
        group = [c for c in clips if c["split"] == split]
        speed = np.linalg.norm([c["target_vxy_mps"] for c in group],axis=1)
        distribution[split] = {"clips":len(group),"sessions":len(sessions[split]),"episodes":len(episodes[split]),
                               "speed_quantiles_mps":np.quantile(speed,[0,.1,.25,.5,.75,.9,.99,1]).tolist(),
                               "stationary":int((speed<.05).sum()),"slow":int(((speed>=.05)&(speed<.5)).sum()),
                               "moving":int(((speed>=.5)&(speed<2)).sum()),"fast":int((speed>=2).sum()),
                               "scenarios":dict(Counter(c["scenario_family"] for c in group)),
                               "visible_clips":sum(c["visible_frames"]==16 for c in group)}
    result = {"passed":True,"physics_ticks":ticks,"rgb_frames":len(samples),"clips_checked":len(clips),
              "split_counts_and_motion":distribution,"cross_split_episode_session_seed_overlap":0,
              "cross_split_duplicate_rgb":0,"within_split_duplicate_rgb":duplicates,"duplicate_initial_states":state_duplicates,
              "rollout_repeat":{"compared_ticks":len(replay),"identical":not differences,
                                "first_different_tick":differences[0]+1 if differences else None,
                                "max_ball_position_difference_m":float(np.linalg.norm(delta,axis=1).max())}}
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,result)
    return result


def initialization_diagnostics(dataset: Path, output: Path):
    """Measure initialization clustering from archived first states, without rerunning simulation."""
    import gzip
    from .experiment import write_json
    dataset,output = resolve_path(dataset),resolve_path(output)
    if output.exists():
        raise FileExistsError(output)
    sessions = []; seeds = []; minimum_robot_distance = []
    for path in sorted((dataset/"trajectories").glob("*.jsonl.gz")):
        starts = []
        with gzip.open(path,"rb") as stream:
            for line in stream:
                if b'"sample_idx":0,' not in line:
                    continue
                row = json.loads(line)
                starts.append(row["ball"]["position_m"][:2]); seeds.append(row["seed"])
                xy = np.array([r["pose_m_rad"][:2] for r in row["robots"]])
                distances = np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
                np.fill_diagonal(distances,np.inf)
                minimum_robot_distance.append(float(distances.min()))
        assert len(starts)==20
        sessions.append(starts)
    values = np.array(sessions)
    n,k,_ = values.shape
    mean = values.mean(1)
    within = ((values-mean[:,None])**2).sum((0,1))/(n*(k-1))
    between = k*((mean-mean.mean(0))**2).sum(0)/(n-1)
    icc = (between-within)/(between+(k-1)*within)
    result = {"sessions":n,"episodes":n*k,"unique_initialization_seeds":len(set(seeds)),
              "mean_within_session_ball_xy_std_m":values.std(1,ddof=1).mean(0).tolist(),
              "between_session_mean_ball_xy_std_m":mean.std(0,ddof=1).tolist(),
              "one_way_random_effects_icc_xy":icc.tolist(),
              "ball_start_xy_min_m":values.reshape(-1,2).min(0).tolist(),
              "ball_start_xy_max_m":values.reshape(-1,2).max(0).tolist(),
              "minimum_robot_center_separation_first_tick_m":min(minimum_robot_distance),
              "interpretation":"Unique fresh snapshots do not imply statistically independent coordinates. Sequential java.util.Random seeds may correlate early draws. Session-level splitting/uncertainty handles clustering; mix per-episode seeds before a future cohort. Measurements use actual first recorded states, not generated trajectories."}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
    return result


def validate(split: Path, output: Path | None = None) -> dict:
    """Check CUDA, sampler, decoder and view builder for one split."""
    split=resolve_path(split)
    if output is not None:
        output=resolve_path(output)
        if output.exists():
            raise FileExistsError(f"Evidence already exists and will be preserved: {output}")
    from .layout import enable_upstream
    enable_upstream()
    import lance
    import numpy as np
    import torch
    from data import Episodes, WindowedLanceDataset, VariableLengthBatchSampler, ViewBuilder, decode
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.cuda.get_device_properties(0)
    if "3060" not in device.name or device.total_memory < 11*1024**3:
        raise RuntimeError(f"Unexpected device: {device}")
    path = str(split / "train.lance")
    index = str(split / "train.episodes.npz")
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
              "dataset": str(split), "episodes": len(episodes.starts),
              "rows": ds.count_rows(), "clip_shape": list(batch["clip"].shape),
              "clip_dtype": str(batch["clip"].dtype), "global_shape": list(g.shape),
              "local_shape": list(local.shape), "normalized_dtype": str(g.dtype),
              "normalized_range": [float(g.min()), float(g.max())],
              "episode_boundary_and_temporal_order": "passed", "all_16_frames_match_decoder": True}
    if output is not None:
        output=Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+"\n")
    return result
