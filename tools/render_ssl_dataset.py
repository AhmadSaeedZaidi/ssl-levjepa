"""Render recorded Sumatra state only. This module contains no trajectory dynamics."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import subprocess

import lance
import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    args=ap.parse_args()
    files=sorted((args.input/"trajectories").glob("session_*.jsonl"))
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
              "sumatra_source_sha":subprocess.check_output(["git","-C",str(ROOT/".sim/Sumatra"),"rev-parse","HEAD"],text=True).strip(),
              "sumatra_release_zip_sha256":digest(ROOT/".sim/sumatra.zip")}
    metadata["recorder_sha256"]=digest(ROOT/"simulation/SumatraRecorder.java")
    metadata["runtime_moduli_sha256"]=digest(ROOT/".sim/sumatra-dist/sumatra/config/moduli/ssl_record.xml")
    metadata["java_version"]=subprocess.run(["java","-version"],capture_output=True,text=True).stderr.strip()
    metadata["split_overlap_episodes"]=0
    metadata["action_units"]={"target_position_mm_rad":"mm,mm,rad","target_velocity_local_mmps_radps":"mm/s,mm/s,rad/s","kick_speed_mps":"m/s","dribbler_rpm":"rpm"}
    schema=pa.schema([("episode_idx",pa.int64()),("step_idx",pa.int64()),("timestamp_ns",pa.int64()),
                      ("frame",pa.binary()),("h",pa.int32()),("w",pa.int32()),("label",pa.int64())],
                      metadata={"ssl_dataset":json.dumps(metadata)})
    clips=[]
    build=runpy.run_path(str(ROOT/"upstream/LeVJEPA/scripts/build_episodes.py"))["build"]
    for split,episodes in frames.items():
        store_dir=args.input/split
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
    (args.input/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in clips))
    metadata["split_counts"]={k:{"episodes":len(v),"frames":len(v)*128,"clips":sum(c["split"]==k for c in clips),
                                  "visible_clips":sum(c["split"]==k and c["visible_frames"]==16 for c in clips)} for k,v in frames.items()}
    metadata["render_and_label_alignment_checks"]="216 exact stored-byte, episode and timestamp checks passed"
    (args.input/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    sheet=Image.new("RGB",(4*256,2*280),(20,24,29)); draw=ImageDraw.Draw(sheet)
    for i,e in enumerate([0,2]):
        episode=all_episodes[e]
        for j,t in enumerate([0,31,63,95]):
            frame=Image.open(io.BytesIO(encoded(episode[t])))
            sheet.paste(frame,(j*256,i*280)); draw.text((j*256+5,i*280+260),f"ep {e}, t={episode[t]['timestamp_ns']/1e9:.2f}s",fill="white")
    sheet.save(args.input/"preview.png")
    sequence=[render(r) for r in all_episodes[0][:16]]
    sequence[0].save(args.input/"clip16.gif",save_all=True,append_images=sequence[1:],duration=67,loop=0)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    main()
