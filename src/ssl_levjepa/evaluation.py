"""Frozen readout to linear ball-velocity probe."""
from pathlib import Path
from .layout import resolve_path, enable_upstream

def evaluate(dataset: Path, run: Path, output: Path, batch_size: int = 2) -> dict:
    """Evaluate trained and saved random encoders on the fixed velocity probe."""
    dataset_root=resolve_path(dataset)
    run=resolve_path(run)
    output=resolve_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    import hashlib, json, os, time
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    enable_upstream()
    import lance
    import numpy as np
    from omegaconf import OmegaConf
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn.functional as F
    from data import WindowedLanceDataset
    from model import Encoder
    def metrics(target, prediction):
        return {"rmse_mps":float(np.sqrt(mean_squared_error(target,prediction))),
                "mae_mps":float(mean_absolute_error(target,prediction)),
                "r2":float(r2_score(target,prediction))}
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    cfg=OmegaConf.load(run/"config.yaml")
    checkpoint=torch.load(run/"checkpoints/final.ckpt",map_location="cpu",weights_only=False)
    initial=torch.load(run/"initial_encoder.pt",map_location="cpu",weights_only=True)
    trained={k.removeprefix("encoder."):v for k,v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    encoders={}
    for name,state in [("random_initialization",initial),("trained",trained)]:
        encoder=Encoder(**cfg.model)
        encoder.load_state_dict(state,strict=True)
        encoders[name]=encoder.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device="cuda").view(1,1,3,1,1)
    std=torch.tensor([.229,.224,.225],device="cuda").view(1,1,3,1,1)
    candidates=[json.loads(line) for line in (dataset_root/"clips.jsonl").read_text().splitlines()]
    clips=[]; excluded={"ball_outside_canvas":0,"duplicate_rgb_clip":0}; seen=set()
    for c in candidates:
        if c["visible_frames"]!=16:
            excluded["ball_outside_canvas"]+=1; continue
        if c["rgb_sha256"] in seen:
            excluded["duplicate_rgb_clip"]+=1; continue
        seen.add(c["rgb_sha256"]); clips.append(c)
    sessions={split:{c["match_id"] for c in clips if c["split"]==split} for split in ["train","validation","test"]}
    assert sessions["train"].isdisjoint(sessions["validation"]|sessions["test"])
    assert sessions["validation"].isdisjoint(sessions["test"])
    features={name:[] for name in encoders}; ordered=[]
    started=time.perf_counter()
    for split in ["train","validation","test"]:
        subset=[c for c in clips if c["split"]==split]
        store=str(dataset_root/split/"train.lance")
        table=lance.dataset(store)
        clip_dataset=WindowedLanceDataset([store],[0],256,256)
        assert len(subset)>=16, f"Too few probe clips in {split}"
        for offset in range(0,len(subset),batch_size):
            group=subset[offset:offset+batch_size]
            indices=np.array([np.arange(c["start_row"],c["start_row"]+16) for c in group])
            # Target is strictly the last input timestamp, never read as an encoder input channel.
            for c,rows in zip(group,indices):
                keys=table.take(rows.tolist(),columns=["episode_idx","timestamp_ns","label"]).to_pydict()
                assert set(keys["episode_idx"])=={c["episode_id"]} and keys["timestamp_ns"][-1]==c["target_timestamp_ns"]
                assert set(keys["label"])=={-1}
            clip=clip_dataset.__getitems__(indices)["clip"].cuda().permute(0,1,4,2,3).float()/255
            b,t,ch,h,w=clip.shape
            clip=F.interpolate(clip.reshape(b*t,ch,h,w),size=(224,224),mode="bilinear",align_corners=False,antialias=True).reshape(b,t,ch,224,224)
            clip=(clip-mean)/std
            with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
                for name,encoder in encoders.items():
                    readout,_,_=encoder(clip)  # dense tokens, no random masks during evaluation
                    assert torch.isfinite(readout).all()
                    features[name].append(readout.float().cpu().numpy())
            ordered.extend(group)
            if offset%64==0:
                print(f"{split}: {min(offset+len(group),len(subset))}/{len(subset)} clips",flush=True)
    features={k:np.concatenate(v) for k,v in features.items()}
    targets=np.array([c["target_vxy_mps"] for c in ordered],dtype=np.float64)
    masks={s:np.array([c["split"]==s for c in ordered]) for s in sessions}
    train,val,test=(masks[s] for s in ["train","validation","test"])
    assert all(float(np.var(targets[m],axis=0).sum())>1e-6 for m in masks.values()), "Degenerate velocity labels"
    result={"task":"regress ball vx,vy (m/s) at last of 16 input frames", "checkpoint_step":checkpoint["global_step"],
            "preprocessing":"deterministic bilinear 224px resize, ImageNet normalization; all 196 tokens/frame; encoder.eval; bf16",
            "feature_dim":192,"clips":{s:int(m.sum()) for s,m in masks.items()},"sessions":{s:sorted(v) for s,v in sessions.items()},
            "excluded":excluded,"seed":42,"extraction_sec":time.perf_counter()-started,"results":{}}
    result["target_std_mps"]={s:np.std(targets[m],axis=0).tolist() for s,m in masks.items()}
    constant=np.broadcast_to(targets[train].mean(axis=0),targets.shape)
    result["results"]["train_mean_constant"]={"validation":metrics(targets[val],constant[val]),"test":metrics(targets[test],constant[test])}
    predictions={"constant":constant}
    for name,x in features.items():
        grid=[]; best=None
        for alpha in [.1,1,10,100,1000]:
            probe=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            probe.fit(x[train],targets[train])
            score=metrics(targets[val],probe.predict(x[val]))
            grid.append({"alpha":alpha,**score})
            if best is None or score["rmse_mps"]<best[0]:
                best=(score["rmse_mps"],alpha,probe)
        prediction=best[2].predict(x)
        predictions[name]=prediction
        result["results"][name]={"alpha":best[1],"validation_grid":grid,
             "validation":metrics(targets[val],prediction[val]),"test":metrics(targets[test],prediction[test]),
             "test_per_session":{session:metrics(targets[sel],prediction[sel]) for session in sorted(sessions["test"])
               for sel in [test & np.array([c["match_id"]==session for c in ordered])]}}
    np.savez(output/"features_and_predictions.npz",targets=targets,**features,**{f"prediction_{k}":v for k,v in predictions.items()},**masks)
    (output/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in ordered))
    result["feature_sha256"]={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in features.items()}
    (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
