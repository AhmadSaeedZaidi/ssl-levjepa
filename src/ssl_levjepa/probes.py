"""Frozen spatial probes and visual controls, isolated from SSL pretraining."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .experiment import checksum, write_json
from .layout import enable_upstream, resolve_path

REGIMES = {"stationary": (0,.05), "slow": (.05,.5), "moving": (.5,2), "fast": (2,float("inf"))}


def metrics(target, prediction):
    if not len(target):
        return {"n":0,"velocity_rmse_mps":None,"velocity_epe_mps":None,"localization_rmse_px":None,"within_8px":None}
    delta = prediction-target
    return {"n":len(target),"velocity_rmse_mps":float(np.sqrt(np.mean(delta[:,:2]**2))),
            "velocity_epe_mps":float(np.linalg.norm(delta[:,:2],axis=1).mean()),
            "localization_rmse_px":float(np.sqrt(np.mean((delta[:,2:]*256)**2))),
            "within_8px":float((np.linalg.norm(delta[:,2:]*256,axis=1)<=8).mean())}


def grouped_metrics(target, prediction, clips):
    speed = np.linalg.norm(target[:,:2],axis=1)
    result = {}
    for split in ("train","validation","test"):
        mask = np.array([c["split"]==split for c in clips])
        result[split] = {"aggregate":metrics(target[mask],prediction[mask]),
                         "motion":{name:metrics(target[m],prediction[m]) for name,(lo,hi) in REGIMES.items()
                                   for m in [mask & (speed>=lo) & (speed<hi)]},
                         "sessions":{name:metrics(target[m],prediction[m]) for name in sorted({c["match_id"] for c in clips if c["split"]==split})
                                     for m in [mask & np.array([c["match_id"]==name for c in clips])]},
                         "scenario":{name:metrics(target[m],prediction[m]) for name in sorted({c.get("scenario_family","legacy") for c in clips})
                                     for m in [mask & np.array([c.get("scenario_family","legacy")==name for c in clips])]},
                         "airborne":metrics(target[m],prediction[m]) if (m:=mask & np.array([c.get("target_z_m",0)>.05 for c in clips])).any() else None}
    return result


def prepare(dataset, output):
    """Freeze the probe cohort once, and cache only two RGB frames for visual controls."""
    import lance
    from .dataset import _recorded_rows
    from PIL import Image
    import io
    dataset,output = resolve_path(dataset),resolve_path(output)
    output.mkdir(parents=True,exist_ok=False)
    candidates = [json.loads(line) for line in (dataset/"clips.jsonl").read_text().splitlines()]
    if candidates and "target_image_xy_px" not in candidates[0]:
        # Enrich a COPY of the legacy pilot manifest; original state, stores and labels stay untouched.
        lookup = {}
        for path in sorted((dataset/"trajectories").glob("session_*.jsonl")):
            for row in _recorded_rows(path):
                if row["sample_idx"]>=0:
                    lookup[(row["episode_id"],row["sample_idx"])] = (row["timestamp_ns"],row["ball"],row["scenario"],row["seed"])
        for c in candidates:
            last = (c["start_row"]%128)+15
            timestamp,ball,scenario,seed = lookup[(c["episode_id"],last)]
            previous = lookup[(c["episode_id"],last-1)]
            assert timestamp==c["target_timestamp_ns"] and ball["velocity_mps"][:2]==c["target_vxy_mps"]
            x,y,z = ball["position_m"]
            c.update(target_xy_m=[x,y],target_image_xy_px=[128+x*(232/9),128-(y+.15*z)*(232/9)],
                     target_z_m=z,dt_last_sec=(timestamp-previous[0])/1e9,scenario_family=f"legacy_{scenario}",
                     seed=seed)
    clips = [c for c in candidates if c.get("visible_last_two",c["visible_frames"]==16)]
    assert all("target_image_xy_px" in c for c in clips), "V2 labels required"
    for key in ("episode_id","match_id","seed"):
        sets = {s:{c[key] for c in clips if c["split"]==s} for s in ("train","validation","test")}
        assert sets["train"].isdisjoint(sets["validation"]|sets["test"]) and sets["validation"].isdisjoint(sets["test"])
    seen = {}
    for c in clips:
        assert c["rgb_sha256"] not in seen or seen[c["rgb_sha256"]]==c["split"]
        seen[c["rgb_sha256"]] = c["split"]
    target = np.array([c["target_vxy_mps"]+[p/256 for p in c["target_image_xy_px"]] for c in clips],dtype=np.float32)
    train = np.array([c["split"]=="train" for c in clips])
    rgb = np.lib.format.open_memmap(output/"rgb_pairs.npy",mode="w+",dtype=np.uint8,shape=(len(clips),2,3,224,224))
    prediction = np.tile(target[train].mean(0),(len(clips),1))
    detections = np.zeros((len(clips),2),dtype=bool)
    tables = {s:lance.dataset(str(dataset/s/"train.lance")) for s in ("train","validation","test")}
    for i,c in enumerate(clips):
        values = tables[c["split"]].take([c["start_row"]+14,c["start_row"]+15],columns=["frame","timestamp_ns","episode_idx"]).to_pylist()
        assert all(row["episode_idx"]==c["episode_id"] for row in values)
        assert values[-1]["timestamp_ns"]==c["target_timestamp_ns"]
        assert abs((values[-1]["timestamp_ns"]-values[0]["timestamp_ns"])/1e9-c["dt_last_sec"])<1e-9
        positions = []
        for j,row in enumerate(values):
            picture = Image.open(io.BytesIO(row["frame"])).convert("RGB")
            pixels = np.asarray(picture)
            # Renderer-aware color detector. No state, position label or learned test calibration is used.
            r,g,b = pixels[...,0].astype(float),pixels[...,1].astype(float),pixels[...,2].astype(float)
            mask = (r>170)&(g>45)&(g<180)&(b<100)&(r>1.3*g)
            ys,xs = np.where(mask)
            detected = len(xs)>=2
            detections[i,j] = detected
            positions.append(np.array([xs.mean()+.5,ys.mean()+.5]) if detected else None)
            rgb[i,j] = np.asarray(picture.resize((224,224),Image.Resampling.BILINEAR)).transpose(2,0,1)
        if positions[-1] is not None:
            prediction[i,2:] = positions[-1]/256
        if all(p is not None for p in positions):
            d = (positions[-1]-positions[0])/(c["dt_last_sec"]*(232/9))
            prediction[i,:2] = [d[0],-d[1]]
        if i%512==0:
            print(f"Prepared RGB/targets {i}/{len(clips)}",flush=True)
    rgb.flush()
    np.save(output/"targets.npy",target)
    (output/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in clips))
    np.savez(output/"visual_detector_predictions.npz",prediction=prediction,detected=detections)
    write_json(output/"visual_detector_metrics.json",grouped_metrics(target,prediction,clips))
    protocol = {"dataset":str(dataset),"dataset_manifest_sha256":checksum(dataset/"clips.jsonl"),
                "cohort_sha256":checksum(output/"clips.jsonl"),"clips":dict(Counter(c["split"] for c in clips)),
                "targets_sha256":checksum(output/"targets.npy"),
                "excluded":dict(Counter(c["split"] for c in candidates if not c.get("visible_last_two",c["visible_frames"]==16))),
                "targets":"last-frame vx,vy in m/s and projected RGB center x,y divided by 256",
                "regimes":{name:[lo,hi if np.isfinite(hi) else None] for name,(lo,hi) in REGIMES.items()},
                "feature_policy":"SSL gets only RGB; downstream probes also receive the recorded frame interval. No state input.",
                "selection":"Ridge alpha and probe epoch chosen on validation only; final test predictions after selection.",
                "head_epochs":20,"head_batch_size":128,"head_seed":20261117,
                "ridge_alpha_grid":[.1,1,10,100,1000,10000],
                "detector_both_frames_rate":float(detections.all(1).mean()),
                "color_detector":"Fixed orange RGB threshold, centroid, backward finite difference; known renderer scale 232px/9m; height cue not inverted"}
    write_json(output/"protocol.json",protocol)
    return protocol


def load_cohort(probe_data):
    folder = resolve_path(probe_data)
    protocol = json.loads((folder/"protocol.json").read_text())
    assert checksum(folder/"clips.jsonl")==protocol["cohort_sha256"], "Probe cohort changed"
    assert checksum(Path(protocol["dataset"])/"clips.jsonl")==protocol["dataset_manifest_sha256"], "Source dataset changed"
    if "targets_sha256" in protocol:
        assert checksum(folder/"targets.npy")==protocol["targets_sha256"], "Probe labels changed"
    clips = [json.loads(line) for line in (folder/"clips.jsonl").read_text().splitlines()]
    target = np.load(folder/"targets.npy")
    masks = {s:np.array([c["split"]==s for c in clips]) for s in ("train","validation","test")}
    return folder,clips,target,masks


def feature_diagnostics(values):
    x = np.asarray(values,dtype=np.float64)
    centered = x-x.mean(0)
    eig = np.maximum(np.linalg.eigvalsh(centered.T@centered/max(len(x)-1,1)),0)
    p = eig/(eig.sum()+1e-30)
    norm = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return {"mean_dimension_std":float(x.std(0).mean()),"min_dimension_std":float(x.std(0).min()),
            "covariance_trace":float(eig.sum()),"effective_rank":float(np.exp(-(p*np.log(np.maximum(p,1e-30))).sum())),
            "participation_rank":float(1/max((p*p).sum(),1e-30)),
            "mean_paired_cosine":float((norm[:len(norm)//2]*norm[-(len(norm)//2):]).sum(1).mean())}


def extract(dataset, run, output, probe_data, weights="trained", batch_size=32, temporal_mode="ordered"):
    """Cache dense frozen patch grids. The encoder is never altered or optimized here."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    enable_upstream()
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from model import Encoder
    from data import WindowedLanceDataset
    dataset,run,output = resolve_path(dataset),resolve_path(run),resolve_path(output)
    output.mkdir(parents=True,exist_ok=False)
    folder,clips,target,masks = load_cohort(probe_data)
    torch.set_num_threads(4); torch.manual_seed(20261117)
    cfg = OmegaConf.load(run/"config.yaml")
    encoder = Encoder(**cfg.model)
    source = run/"initial_encoder.pt" if weights=="random" else run/"checkpoints/final.ckpt"
    state = torch.load(source,map_location="cpu",weights_only=weights=="random")
    if weights!="random":
        state = {k.removeprefix("encoder."):v for k,v in state["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(state,strict=True)
    encoder = encoder.cuda().eval().requires_grad_(False)
    before = {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in encoder.state_dict().items()}
    patches = np.lib.format.open_memmap(output/"patches.npy",mode="w+",dtype=np.float32,shape=(len(clips),2,196,192))
    readouts = np.zeros((len(clips),192),dtype=np.float32)
    pooled = np.zeros_like(readouts)
    mean = torch.tensor([.485,.456,.406],device="cuda").view(1,1,3,1,1)
    std = torch.tensor([.229,.224,.225],device="cuda").view(1,1,3,1,1)
    stores = {s:WindowedLanceDataset([str(dataset/s/"train.lance")],[0],256,256) for s in masks}
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for split in masks:
        indices = np.flatnonzero(masks[split])
        for offset in range(0,len(indices),batch_size):
            ids = indices[offset:offset+batch_size]
            rows = np.array([np.arange(clips[i]["start_row"],clips[i]["start_row"]+16) for i in ids])
            video = stores[split].__getitems__(rows)["clip"].cuda().permute(0,1,4,2,3).float()/255
            b,t,c,h,w = video.shape
            video = F.interpolate(video.reshape(b*t,c,h,w),size=(224,224),mode="bilinear",align_corners=False,antialias=True).reshape(b,t,c,224,224)
            if temporal_mode=="repeat_last":
                video = video[:,-1:].expand(-1,16,-1,-1,-1)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                readout,grid,idx = encoder((video-mean)/std)
            assert grid.shape == (len(ids),16,196,192) and torch.isfinite(grid).all()
            assert torch.equal(idx[0,0],torch.arange(196,device=idx.device))
            readouts[ids] = readout.float().cpu().numpy()
            pooled[ids] = grid.float().mean((1,2)).cpu().numpy()
            patches[ids] = grid[:,-2:].float().cpu().numpy()
            if offset%128==0:
                print(f"{weights} dense frozen extraction {split}: {min(offset+len(ids),len(indices))}/{len(indices)}",flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    patches.flush()
    assert before == {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in encoder.state_dict().items()}
    np.savez(output/"global_features.npz",readout=readouts,mean_pool=pooled)
    result = {"weights":weights,"source":str(source),"source_sha256":checksum(source),"encoder_frozen_verified":True,
              "wall_sec":elapsed,"gpu_hours":elapsed/3600,"clips_per_sec":len(clips)/elapsed,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
              "readout_train":feature_diagnostics(readouts[masks["train"]]) if masks["train"].any() else None,
              "mean_pool_train":feature_diagnostics(pooled[masks["train"]]) if masks["train"].any() else None,
              "cohort_sha256":checksum(folder/"clips.jsonl"),"dense_tokens_per_frame":196,"input_frames":16,
              "patch_cache":"last two causal frame grids in FP32; preserve small residual variation after float32 readout-token concatenation",
              "encoder_output_dtype":str(grid.dtype),"temporal_mode":temporal_mode}
    write_json(output/"extraction.json",result)
    return result


def _head_module(kind="conv"):
    import torch.nn as nn
    layers = [nn.Conv2d(384,32,1),nn.GELU()]
    if kind=="conv":
        layers += [nn.Conv2d(32,32,3,padding=1),nn.GELU()]
    layers += [nn.Conv2d(32,5,1)]
    return nn.Sequential(*layers)


def _head_input(patches, ids, clips, device="cuda", normalization=None):
    import torch
    if isinstance(patches,torch.Tensor):
        pairs = patches[torch.as_tensor(ids,device=patches.device)].float()
    else:
        pairs = torch.from_numpy(np.array(patches[ids],dtype=np.float32)).to(device)
    if pairs.ndim==4:
        pairs = pairs.reshape(-1,2,14,14,192).permute(0,1,4,2,3)
    if normalization is not None:
        center,scale = normalization
        pairs = (pairs-center.reshape(1,1,-1,1,1))/scale.reshape(1,1,-1,1,1)
    dt = torch.tensor([clips[i]["dt_last_sec"]*30 for i in ids],device=device).reshape(-1,1,1,1)
    return torch.cat((pairs[:,1],(pairs[:,1]-pairs[:,0])/dt),dim=1)


def _head_decode(maps, mean, std):
    import torch
    b = maps.shape[0]
    cell = maps[:,0].flatten(1).argmax(1)
    values = maps.flatten(2).transpose(1,2)[torch.arange(b,device=maps.device),cell]
    xy = torch.stack((cell%14,cell//14),1)+.5+.5*values[:,1:3].tanh()
    return torch.cat((values[:,3:5]*std[:2]+mean[:2],xy/14),1)


def _head_loss(maps, target, mean, std):
    import torch
    import torch.nn.functional as F
    xy = target[:,2:]*14
    cells = xy.floor().long().clamp(0,13)
    index = cells[:,1]*14+cells[:,0]
    values = maps.flatten(2).transpose(1,2)[torch.arange(len(target),device=maps.device),index]
    offset = xy-(cells+.5)
    return (F.cross_entropy(maps[:,0].flatten(1),index)
            +2*F.smooth_l1_loss(.5*values[:,1:3].tanh(),offset)
            +F.mse_loss(values[:,3:5],(target[:,:2]-mean[:2])/std[:2]))


def _predict_head(head, patches, indices, clips, mean, std, encoder=None, batch_size=128, normalization=None):
    import torch
    result = []
    head.eval()
    if encoder is not None:
        encoder.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for offset in range(0,len(indices),batch_size):
            ids = indices[offset:offset+batch_size]
            features = _cnn_features(encoder,patches,ids) if encoder is not None else patches
            batch_clips = [clips[i] for i in ids] if encoder is not None else clips
            maps = head(_head_input(features,np.arange(len(ids)) if encoder is not None else ids,batch_clips,normalization=normalization))
            result.append(_head_decode(maps,mean,std).float().cpu().numpy())
    return np.concatenate(result)


def _fit_head(patches,target,clips,masks,output,kind,seed=20261117,encoder=None,epochs=20,batch_size=128):
    import torch
    torch.manual_seed(seed)
    head = _head_module(kind).cuda()
    mean = torch.tensor(target[masks["train"]].mean(0),device="cuda")
    std = torch.tensor(np.maximum(target[masks["train"]].std(0),1e-3),device="cuda")
    optimizer = torch.optim.AdamW(list(head.parameters())+(list(encoder.parameters()) if encoder is not None else []),lr=.001,weight_decay=.001)
    train_ids,val_ids = np.flatnonzero(masks["train"]),np.flatnonzero(masks["validation"])
    normalization = None
    if encoder is None:
        total = np.zeros(192,dtype=np.float64); squared = np.zeros_like(total); count = 0
        for start in range(0,len(train_ids),128):
            values = np.asarray(patches[train_ids[start:start+128]],dtype=np.float64).reshape(-1,192)
            total += values.sum(0); squared += (values*values).sum(0); count += len(values)
        center = total/count
        scale = np.sqrt(np.maximum(squared/count-center**2,1e-12))
        normalization = (torch.tensor(center,dtype=torch.float32,device="cuda"),torch.tensor(scale,dtype=torch.float32,device="cuda"))
    rng = np.random.default_rng(seed)
    best = None; history = []; started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        head.train()
        if encoder is not None:
            encoder.train()
        losses = []
        for offset in range(0,len(train_ids),batch_size):
            if offset==0:
                ids_order = rng.permutation(train_ids)
            ids = ids_order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                features = _cnn_features(encoder,patches,ids) if encoder is not None else patches
                batch_clips = [clips[i] for i in ids] if encoder is not None else clips
                x = _head_input(features,np.arange(len(ids)) if encoder is not None else ids,batch_clips,normalization=normalization)
                maps = head(x)
                y = torch.from_numpy(target[ids]).cuda()
                loss = _head_loss(maps,y,mean,std)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite spatial probe loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters())+(list(encoder.parameters()) if encoder is not None else []),1)
            optimizer.step(); losses.append(float(loss.detach()))
        prediction = _predict_head(head,patches,val_ids,clips,mean,std,encoder,batch_size,normalization)
        score = float(np.mean(((prediction-target[val_ids])/std.cpu().numpy())**2))
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"validation_standardized_mse":score})
        if best is None or score<best[0]:
            best = (score,copy.deepcopy(head.state_dict()),copy.deepcopy(encoder.state_dict()) if encoder is not None else None,epoch+1)
        print(f"{kind} epoch {epoch+1}/{epochs} validation standardized MSE {score:.5f}",flush=True)
    head.load_state_dict(best[1])
    if encoder is not None:
        encoder.load_state_dict(best[2]); encoder.requires_grad_(False)
    prediction = _predict_head(head,patches,np.arange(len(clips)),clips,mean,std,encoder,batch_size,normalization)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    torch.save({"head":head.state_dict(),"encoder":encoder.state_dict() if encoder is not None else None,
                "target_mean":mean.cpu(),"target_std":std.cpu(),"kind":kind,"selected_epoch":best[3],
                "feature_center":normalization[0].cpu() if normalization else None,
                "feature_scale":normalization[1].cpu() if normalization else None},output/"head.pt")
    evidence = {"history":history,"selected_epoch":best[3],"seed":seed,"wall_sec":elapsed,"gpu_hours":elapsed/3600,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "train_clips":len(train_ids),"test_used_in_selection":False}
    write_json(output/"fit.json",evidence)
    return prediction,evidence


def fit(probe_data, features, output, remove_patch_cache=False):
    """Compare readout, mean pooling, spatial linear features and two shared spatial heads."""
    import joblib
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    folder,clips,target,masks = load_cohort(probe_data)
    features,output = resolve_path(features),resolve_path(output)
    evidence = json.loads((features/"extraction.json").read_text())
    assert evidence["cohort_sha256"]==checksum(folder/"clips.jsonl"), "Features and labels use different cohorts"
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    global_features = np.load(features/"global_features.npz")
    patches = np.load(features/"patches.npy",mmap_mode="r")
    # A fixed label-independent projection controls linear-probe parameter count; shared heads use all 192 channels.
    projection = np.random.default_rng(20261117).normal(size=(192,8)).astype(np.float32)/np.sqrt(192)
    spatial = np.zeros((len(clips),2*7*7*8),dtype=np.float32)
    for start in range(0,len(clips),128):
        p = np.asarray(patches[start:start+128],dtype=np.float32).reshape(-1,2,7,2,7,2,192).mean((3,5))
        spatial[start:start+len(p)] = (p@projection).reshape(len(p),-1)
    np.savez_compressed(features/"spatial_linear_features.npz",features=spatial,projection=projection)
    candidates = {"global_readout_ridge":global_features["readout"],"mean_patch_pool_ridge":global_features["mean_pool"],"spatial_patch_ridge":spatial}
    dt = np.array([c["dt_last_sec"] for c in clips],dtype=np.float32)[:,None]
    ymean,ystd = target[masks["train"]].mean(0),np.maximum(target[masks["train"]].std(0),1e-3)
    results = {}
    constant = np.tile(ymean,(len(clips),1))
    results["constant"] = grouped_metrics(target,constant,clips)
    np.savez(output/"constant_predictions.npz",prediction=constant)
    with threadpool_limits(limits=4):
        for name,x in candidates.items():
            destination = output/name; destination.mkdir()
            x = np.concatenate((x,dt),1).astype(np.float64)
            best = None; grid = []
            for alpha in (.1,1,10,100,1000,10000):
                estimator = make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                estimator.fit(x[masks["train"]],(target[masks["train"]]-ymean)/ystd)
                pred = estimator.predict(x[masks["validation"]])*ystd+ymean
                score = float(np.mean(((pred-target[masks["validation"]])/ystd)**2))
                if not np.isfinite(score):
                    raise FloatingPointError(f"Non-finite validation score for {name}, alpha={alpha}")
                grid.append({"alpha":alpha,"validation_standardized_mse":score})
                if best is None or score<best[0]:
                    best = (score,estimator,alpha)
            prediction = best[1].predict(x)*ystd+ymean
            joblib.dump({"estimator":best[1],"target_mean":ymean,"target_std":ystd},destination/"probe.joblib")
            np.savez(destination/"predictions.npz",prediction=prediction)
            results[name] = grouped_metrics(target,prediction,clips)
            write_json(destination/"selection.json",{"alpha":best[2],"grid":grid,"test_used_in_selection":False})
            print(f"Completed {name}",flush=True)
    for kind in ("pointwise","conv"):
        name = "spatial_"+kind
        destination = output/name; destination.mkdir()
        prediction,_ = _fit_head(patches,target,clips,masks,destination,kind)
        np.savez(destination/"predictions.npz",prediction=prediction)
        results[name] = grouped_metrics(target,prediction,clips)
    write_json(output/"metrics.json",results)
    del patches
    if remove_patch_cache:
        (features/"patches.npy").unlink()
        write_json(features/"cache_policy.json",{"full_patch_cache":"Removed after successful probes; regenerate from saved encoder and fixed RGB", "retained":"global features, fixed-projection spatial features, heads, predictions and extraction evidence"})
    return results


def _cnn_module():
    import torch.nn as nn
    layers = []
    for a,b in [(3,32),(32,64),(64,96),(96,192)]:
        layers += [nn.Conv2d(a,b,3,stride=2,padding=1),nn.GroupNorm(8,b),nn.GELU()]
    return nn.Sequential(*layers)


def _cnn_features(encoder,rgb,ids):
    import torch
    value = torch.from_numpy(np.array(rgb[ids],dtype=np.float32)).cuda()/255
    features = encoder(value.reshape(-1,3,224,224)).reshape(len(ids),2,192,14,14)
    # Keep the graph for the explicitly supervised control only.
    return features


def supervised(probe_data, output, seed=42):
    """A small RGB-only supervised encoder, then the identical frozen probes."""
    import torch
    folder,clips,target,masks = load_cohort(probe_data)
    output = resolve_path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(seed)
    encoder = _cnn_module().cuda()
    rgb = np.load(folder/"rgb_pairs.npy",mmap_mode="r")
    training = output/"training"; training.mkdir()
    prediction,evidence = _fit_head(rgb,target,clips,masks,training,"conv",seed=seed,encoder=encoder,epochs=20,batch_size=64)
    np.savez(training/"predictions.npz",prediction=prediction)
    write_json(training/"metrics.json",grouped_metrics(target,prediction,clips))
    features = output/"features"; features.mkdir()
    patches = np.lib.format.open_memmap(features/"patches.npy",mode="w+",dtype=np.float32,shape=(len(clips),2,196,192))
    global_values = np.zeros((len(clips),192),dtype=np.float32)
    encoder.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for start in range(0,len(clips),64):
            ids = np.arange(start,min(start+64,len(clips)))
            p = _cnn_features(encoder,rgb,ids)
            global_values[ids] = p.float().mean((1,3,4)).cpu().numpy()
            patches[ids] = p.permute(0,1,3,4,2).reshape(len(ids),2,196,192).float().cpu().numpy()
    patches.flush(); del patches
    np.savez(features/"global_features.npz",readout=global_values,mean_pool=global_values)
    write_json(features/"extraction.json",{"source":"small supervised CNN; two observed RGB frames, no state input", "seed":seed,
               "cohort_sha256":checksum(folder/"clips.jsonl"),
               "wall_sec":time.perf_counter()-started,"readout_train":feature_diagnostics(global_values[masks["train"]]),
               "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"encoder_frozen_after_supervision":True})
    return evidence


def _predict_saved(feature_folder, fitted_folder, original_features, cohort, output):
    """Apply saved probes without refitting, including their original train-only normalization."""
    import joblib
    import torch
    folder,clips,target,masks = load_cohort(cohort)
    output.mkdir(parents=True,exist_ok=False)
    global_values = np.load(feature_folder/"global_features.npz")
    patches = np.load(feature_folder/"patches.npy",mmap_mode="r")
    projection = np.load(original_features/"spatial_linear_features.npz")["projection"]
    spatial = np.empty((len(clips),784),dtype=np.float32)
    for start in range(0,len(clips),128):
        value = np.asarray(patches[start:start+128],dtype=np.float32).reshape(-1,2,7,2,7,2,192).mean((3,5))
        spatial[start:start+len(value)] = (value@projection).reshape(len(value),-1)
    dt = np.array([c["dt_last_sec"] for c in clips],dtype=np.float32)[:,None]
    result = {}
    for name,x in {"global_readout_ridge":global_values["readout"],"mean_patch_pool_ridge":global_values["mean_pool"],"spatial_patch_ridge":spatial}.items():
        probe = joblib.load(fitted_folder/name/"probe.joblib")
        prediction = probe["estimator"].predict(np.concatenate((x,dt),1).astype(np.float64))*probe["target_std"]+probe["target_mean"]
        result[name] = grouped_metrics(target,prediction,clips)
        np.savez(output/(name+"_predictions.npz"),prediction=prediction)
    for kind in ("pointwise","conv"):
        state = torch.load(fitted_folder/("spatial_"+kind)/"head.pt",map_location="cpu",weights_only=True)
        head = _head_module(kind).cuda().eval().requires_grad_(False)
        head.load_state_dict(state["head"])
        mean,std = state["target_mean"].cuda(),state["target_std"].cuda()
        normalization = (state["feature_center"].cuda(),state["feature_scale"].cuda())
        prediction = _predict_head(head,patches,np.arange(len(clips)),clips,mean,std,normalization=normalization)
        result["spatial_"+kind] = grouped_metrics(target,prediction,clips)
        np.savez(output/("spatial_"+kind+"_predictions.npz"),prediction=prediction)
    write_json(output/"metrics.json",result)
    return result


def diagnose(dataset, experiment):
    """Predeclared readout/crop/temporal/resolution diagnostics; no encoder optimization."""
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    enable_upstream()
    from data import ViewBuilder
    dataset,experiment = resolve_path(dataset),resolve_path(experiment)
    output = experiment/"diagnostics"
    output.mkdir(parents=True,exist_ok=False)
    folder,clips,target,masks = load_cohort(experiment/"evaluation/cohort")
    from .dataset import _recorded_rows
    coverage = {}; signatures = {}; first_signatures = {}; tick_count = 0
    for path in sorted((dataset/"trajectories").glob("*.jsonl.gz")):
        episode_samples = {}
        for row in _recorded_rows(path):
            tick_count += 1
            assert np.isfinite(row["ball"]["position_m"]+row["ball"]["velocity_mps"]+row["ball"]["acceleration_mps2"]).all()
            if row["sample_idx"]>=0:
                episode_samples.setdefault(row["episode_id"],[]).append(row)
        for episode,rows in episode_samples.items():
            family = rows[0]["scenario_family"]
            stats = coverage.setdefault(family,{"episodes":0,"turning_episodes":0,"barrier_contact_frames":0,"kick_request_frames":0,"native_kick_timestamps":0,"near_robot_turns":0,"same_team_contact_transfers":0})
            stats["episodes"] += 1
            velocity = np.array([r["ball"]["velocity_mps"][:2] for r in rows])
            speed = np.linalg.norm(velocity,axis=1)
            direction = velocity/np.maximum(speed[:,None],1e-9)
            turns = ((direction[1:]*direction[:-1]).sum(1)<np.cos(.2))&(speed[:-1]>.15)&(speed[1:]>.15)
            stats["turning_episodes"] += int(turns.any())
            kick_times = set(); last_contact = None
            states = []
            for j,row in enumerate(rows):
                bots = sorted(row["robots"],key=lambda b:(b["team"],b["id"]))
                positions = [b["pose_m_rad"] for b in bots]
                assert np.isfinite(positions).all()
                states.append(np.round(np.concatenate((row["ball"]["position_m"],np.asarray(positions).flatten())),2).tolist())
                contact = [(b["team"],b["id"]) for b in bots if b["ball_contact"]]
                stats["barrier_contact_frames"] += bool(contact)
                stats["kick_request_frames"] += any(b["action_applied_last_10ms"]["kick_speed_mps"]>0 for b in bots)
                if row["last_kick_timestamp_ns"] is not None:
                    kick_times.add(row["last_kick_timestamp_ns"])
                if contact:
                    if last_contact and last_contact!=contact[0] and last_contact[0]==contact[0][0] and kick_times:
                        stats["same_team_contact_transfers"] += 1
                    last_contact = contact[0]
                if j>0 and turns[j-1]:
                    ball_xy = np.array(row["ball"]["position_m"][:2])
                    if min(np.linalg.norm(ball_xy-np.array(b["pose_m_rad"][:2])) for b in bots)<.2:
                        stats["near_robot_turns"] += 1
            stats["native_kick_timestamps"] += len(kick_times)
            signatures[episode] = hashlib.sha256(json.dumps(states).encode()).hexdigest()
            first_signatures[episode] = hashlib.sha256(json.dumps(states[0]).encode()).hexdigest()
    assert len(set(signatures.values()))==len(signatures), "Repeated quantized trajectory"
    assert len(set(first_signatures.values()))==len(first_signatures), "Repeated quantized starting state"
    write_json(output/"dataset_diversity.json",{"physics_ticks_finite":tick_count,"episode_count":len(signatures),
               "duplicate_1cm_quantized_trajectories":len(signatures)-len(set(signatures.values())),
               "duplicate_1cm_quantized_first_states":len(first_signatures)-len(set(first_signatures.values())),"coverage":coverage,
               "limitations":"Turn/contact-transfer counters are descriptive proxies, not tactical pass-success labels or exhaustive collision-solver events. Quantization rounds positions and orientations to 0.01 in their native normalized units."})
    declaration = json.loads((experiment/"protocol.json").read_text())
    seed = declaration["seeds"][0]  # Selected by declaration, never by test performance.
    training = experiment/"seeds"/str(seed)/"training"
    cfg = OmegaConf.load(training/"config.yaml")
    builder = ViewBuilder(**cfg.views)
    torch.manual_seed(20261117)
    train_ids = np.flatnonzero(masks["train"])
    xy = torch.tensor(target[train_ids,2:]*256)
    crop = {}
    for name,scale,size,keep in [("global",builder.global_scale,224,20),("local",builder.local_scale,96,4)]:
        views = 1 if name=="global" else 4
        boxes = builder.sample_boxes(views*len(xy),256,256,scale,"cpu").reshape(views,len(xy),4)
        contained = ((xy[None]>=boxes[...,:2])&(xy[None]<boxes[...,2:])).all(-1)
        transformed = (xy[None]-boxes[...,:2])/(boxes[...,2:]-boxes[...,:2])*size
        radius = 1.5/(boxes[...,2:]-boxes[...,:2])*size
        lo = ((transformed-radius)/16).floor().clamp(0,size//16-1)
        hi = ((transformed+radius)/16).floor().clamp(0,size//16-1)
        count = ((hi-lo+1).prod(-1)*contained).numpy().astype(int)
        N,K = 16*(size//16)**2,16*keep
        hit = np.zeros_like(count,dtype=float)
        for i in range(views):
            for j,m in enumerate(count[i]):
                hit[i,j] = 1-np.prod([(N-K-k)/(N-k) for k in range(m)])
        crop[name] = {"last_frame_ball_inside_crop_rate":float(contained.float().mean()),
                      "last_frame_ball_has_retained_patch_probability":float(hit.mean()),
                      "at_least_one_view_retains_a_last_frame_ball_patch_probability":float((1-np.prod(1-hit,axis=0)).mean()),
                      "minimum_ball_diameter_px_median":float((2*radius.mean(-1)).median()),
                      "view_size":size,"kept_tokens_per_frame_equivalent":keep,"actual_dropped_fraction":1-keep/(size//16)**2}
    crop["local_gray_probability"] = builder.gray_p
    crop["interpretation"] = "Last-frame marker bounding-box overlap with sampled patches, not whole-clip visibility. Actual upstream crop boxes; IID masks use exact hypergeometric hit probability. Source ball radius uses the documented 1.5px minimum."
    write_json(output/"crop_visibility.json",crop)
    # Lower-resolution RGB control, predeclared and independent of labels except scoring.
    rgb = np.load(folder/"rgb_pairs.npy",mmap_mode="r")
    heldout = np.flatnonzero(~masks["train"])
    heldout_clips = [clips[i] for i in heldout]
    resolution = {}
    for size in (224,112,56):
        prediction = np.tile(target[masks["train"]].mean(0),(len(heldout),1)); detected = []
        for j,i in enumerate(heldout):
            positions = []
            for image in rgb[i]:
                p = np.asarray(Image.fromarray(image.transpose(1,2,0)).resize((size,size),Image.Resampling.BILINEAR)).astype(float)
                r,g,b = p[...,0],p[...,1],p[...,2]
                ys,xs = np.where((r>170)&(g>45)&(g<180)&(b<100)&(r>1.3*g))
                positions.append(np.array([xs.mean()+.5,ys.mean()+.5])*256/size if len(xs)>=2 else None)
            detected.append(all(p is not None for p in positions))
            if positions[-1] is not None:
                prediction[j,2:] = positions[-1]/256
            if detected[-1]:
                delta = (positions[-1]-positions[0])/(clips[i]["dt_last_sec"]*(232/9))
                prediction[j,:2] = [delta[0],-delta[1]]
        resolution[str(size)] = {"both_frames_detected_rate":float(np.mean(detected)),"metrics":grouped_metrics(target[heldout],prediction,heldout_clips)}
        np.savez(output/(f"resolution_{size}_predictions.npz"),prediction=prediction)
    write_json(output/"resolution.json",resolution)
    # Preserve the true last image and remove motion by repeating it. Saved probes are applied without fitting.
    subset = output/"temporal_cohort"; subset.mkdir()
    (subset/"clips.jsonl").write_text("".join(json.dumps(c)+"\n" for c in heldout_clips))
    np.save(subset/"targets.npy",target[heldout])
    original_protocol = json.loads((folder/"protocol.json").read_text())
    original_protocol.update(cohort_sha256=checksum(subset/"clips.jsonl"),targets_sha256=checksum(subset/"targets.npy"))
    write_json(subset/"protocol.json",original_protocol)
    temporal = {}
    for weights in ("random","trained"):
        base = experiment/"seeds"/str(seed)/"evaluation"/weights
        feature_folder = output/("repeat_last_"+weights)/"features"
        extraction = extract(dataset,training,feature_folder,subset,weights=weights,batch_size=32,temporal_mode="repeat_last")
        result = _predict_saved(feature_folder,base/"probes",base/"features",subset,feature_folder.parent/"probes")
        temporal[weights] = {"seed":seed,"extraction":extraction,"metrics":result,
                             "interpretation":"Inference intervention, not a retrained baseline; all 16 RGB frames repeat the true last frame, so no observed motion remains. Distribution shift limits causal attribution."}
        (feature_folder/"patches.npy").unlink()
    write_json(output/"temporal.json",temporal)
    variance = {}
    for seed in declaration["seeds"]:
        variance[str(seed)] = {}
        for name in ("random","trained","supervised"):
            features = experiment/"seeds"/str(seed)/"evaluation"/name/"features"
            x = np.load(features/"spatial_linear_features.npz")["features"].reshape(-1,2,49,8)
            value = x[masks["train"]]
            variance[str(seed)][name] = {"spatial_variance":float(value[:,1].var(1).mean()),
                                       "last_two_temporal_mean_square_change":float(((value[:,1]-value[:,0])**2).mean()),
                                       "scope":"fixed 8-channel projected 7x7 patch grids; compare paired models, not raw feature scale across architectures"}
    write_json(output/"patch_variance.json",variance)
    return {"crop_visibility":str(output/"crop_visibility.json"),"temporal":str(output/"temporal.json"),"resolution":str(output/"resolution.json")}
