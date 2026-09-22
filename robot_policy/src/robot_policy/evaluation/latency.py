from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_policy.config import TRAINABLE_ARCHITECTURES, is_continuous_architecture, require_active_model_size
import torch.nn.functional as F
import av

from robot_policy.config import load_config
from robot_policy.data.dataset import create_policy_dataset
from robot_policy.encoders.vision import FrozenDinoSigLIP
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


def _stats(values):
    a=np.asarray(values); return {"mean_ms":float(a.mean()),"p50_ms":float(np.quantile(a,.5)),"p95_ms":float(np.quantile(a,.95)),"p99_ms":float(np.quantile(a,.99))}


def _time(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); values=[]
    for _ in range(iterations):
        start=time.perf_counter(); fn(); torch.cuda.synchronize(); values.append((time.perf_counter()-start)*1000)
    return _stats(values)


def _video_frame(path: Path, index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(index)


def _vision_from_pixels(frontend, dino_pixels, siglip_pixels):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dino=frontend._patches(frontend.dino,dino_pixels)
        siglip=(frontend._patches(frontend.siglip,siglip_pixels)
                if frontend.siglip is not None and siglip_pixels is not None else None)
    fused=torch.cat([dino,siglip],-1) if siglip is not None else dino
    b,n,d=fused.shape; side=int(n**.5)
    return F.adaptive_avg_pool2d(fused.transpose(1,2).reshape(b,d,side,side),frontend.cfg.vision.pooled_grid).flatten(2).transpose(1,2)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--set",action="append",default=[]); p.add_argument("--architecture",required=True,choices=TRAINABLE_ARCHITECTURES); p.add_argument("--checkpoint",required=True); p.add_argument("--warmup",type=int,default=10); p.add_argument("--iterations",type=int,default=100); p.add_argument("--split",choices=("train","val","test"),default="test"); p.add_argument("--output",required=True)
    a=p.parse_args(argv); cfg=load_config(a.config,[*a.set,f"policy.architecture={a.architecture}"])
    require_active_model_size(cfg.policy.model_size, "latency evaluation")
    device=torch.device("cuda")
    model,payload=load_policy_checkpoint(a.checkpoint,cfg,device); codec=create_action_codec(cfg,device); dataset=create_policy_dataset(cfg,a.split); data=dataset[0]
    observation_key="images" if "images" in data else "vision_features"
    batch={observation_key:data[observation_key][None].to(device),"state":data["state"][None].to(device)}
    eid=int(data["episode_id"]); frame=int(data["frame_index"]); raw=[]
    for camera in cfg.data.camera_keys:
        raw.append(_video_frame(Path(cfg.data.dataset_path)/"videos"/"chunk-000"/camera/f"episode_{eid:06d}.mp4",frame))
    rgb=torch.from_numpy(np.stack(raw)).to(device)
    torch.cuda.reset_peak_memory_stats(); result={"checkpoint":str(Path(a.checkpoint).resolve()),"architecture":a.architecture,"training_type":payload["training_type"],"split":a.split,"warmup":a.warmup,"iterations":a.iterations}
    if cfg.data.observation_source == "features":
        frontend=FrozenDinoSigLIP(cfg).to(device).eval()
        dino_pixels,siglip_pixels=frontend.preprocess(rgb)
        result["image_preprocessing"]=_time(lambda:frontend.preprocess(rgb),a.warmup,a.iterations)
        result["vision_encoding_fusion_pool"]=_time(lambda:_vision_from_pixels(frontend,dino_pixels,siglip_pixels),a.warmup,a.iterations)
        result["online_vision_total"]=_time(lambda:frontend(rgb).fused_patches,a.warmup,a.iterations)
    else:
        result["image_preprocessing"]={"included_in_policy_observation_encoder":True}
        result["vision_encoding_fusion_pool"]={"architecture":"scratch ResNet18 + SpatialSoftmax"}
        result["online_vision_total"]=_time(lambda:model.observations(batch),a.warmup,a.iterations)
    result["projector_state"]=_time(lambda:model.observations(batch),a.warmup,a.iterations)
    settings=[]
    if is_continuous_architecture(a.architecture): settings=[{"steps":s} for s in (5,8,12)]
    elif a.architecture=="discrete_joint":
        settings=[]
        for rounds in (4,8,12):
            settings.extend((
                {"rounds":rounds,"use_cache":False},
                {"rounds":rounds,"use_cache":True,"fuse_cache_transition":False},
                {"rounds":rounds,"use_cache":True,"fuse_cache_transition":True},
            ))
    else: settings=[{"rounds":r} for r in (4,8,12)]
    network_calls={}
    for setting in settings:
        name="sampling_"+"_".join(f"{k}-{v}" for k,v in setting.items())
        result[name]=_time(lambda s=setting:model.sample(batch,**s),a.warmup,a.iterations)
        if is_continuous_architecture(a.architecture): calls=setting["steps"]
        elif a.architecture in {"discrete_layerwise", "bsp_unet_discrete"}: calls=setting["rounds"]
        else:
            blocks=(model.action_positions+cfg.policy.block_size-1)//cfg.policy.block_size
            # Legacy caching adds one commitment call after each non-final
            # block.  Fused D2F-style transitions fold it into the next
            # block's first denoising call.
            if not setting["use_cache"]: calls=blocks*setting["rounds"]
            elif setting.get("fuse_cache_transition",True): calls=blocks*setting["rounds"]
            else: calls=blocks*setting["rounds"]+blocks-1
        network_calls[name]=calls
    controls=model.sample(batch); controls=controls.float() if is_continuous_architecture(a.architecture) else codec.decode_tokens(controls)
    result["action_decode"]=_time(lambda:codec.decode_controls(controls),a.warmup,a.iterations)
    delay=torch.ones(1,device=device,dtype=torch.long)
    if cfg.data.action_representation=="raw":
        result["rtc_shift_refit_and_mask"]=_time(lambda:(codec.shift_and_refit(controls,delay),raw_action_prefix_mask(delay,cfg.data.action_horizon)),a.warmup,a.iterations)
    else:
        result["rtc_shift_refit_and_mask"]=_time(
            lambda:(
                codec.shift_and_refit(controls,delay*cfg.spline.span_length_steps),
                control_support_mask(
                    delay,
                    cfg.spline.num_basis,
                    controls.shape[-1],
                    cfg.spline.degree,
                ),
            ),
            a.warmup,
            a.iterations,
        )
    if a.architecture=="discrete_joint":
        torch.manual_seed(20260915); uncached=model.sample(batch,rounds=8,use_cache=False)
        torch.manual_seed(20260915); legacy=model.sample(batch,rounds=8,use_cache=True,fuse_cache_transition=False)
        torch.manual_seed(20260915); fused=model.sample(batch,rounds=8,use_cache=True,fuse_cache_transition=True)
        result["cache_equivalence"]={
            "uncached_vs_legacy_tokens_equal":bool(torch.equal(uncached,legacy)),
            "uncached_vs_fused_tokens_equal":bool(torch.equal(uncached,fused)),
            "legacy_vs_fused_tokens_equal":bool(torch.equal(legacy,fused)),
            "uncached_vs_legacy_max_token_difference":int((uncached-legacy).abs().max()),
            "uncached_vs_fused_max_token_difference":int((uncached-fused).abs().max()),
            "legacy_vs_fused_max_token_difference":int((legacy-fused).abs().max()),
        }
        result["cache_strategy"]="dd-openvla D2F-aligned fused completed-block K/V commit plus next-block first denoising pass"
    result["peak_memory_bytes"]=torch.cuda.max_memory_allocated(); result["gpu"]=torch.cuda.get_device_name(); result["precision"]=("frozen vision BF16 autocast; policy FP32" if cfg.data.observation_source=="features" else "joint scratch vision + policy FP32"); result["batch_size"]=1
    action_steps=cfg.data.action_horizon if cfg.data.action_representation=="raw" else cfg.spline.num_basis
    vision_tokens_per_camera=(cfg.vision.resampler_tokens_per_camera or cfg.vision.pooled_grid**2)
    result["token_lengths"]={"observation":(cfg.data.observation_horizon*(len(cfg.data.camera_keys)*vision_tokens_per_camera+1) if cfg.data.observation_source=="features" else None),"observation_global_vector":(None if cfg.data.observation_source=="features" else cfg.data.observation_horizon*(len(cfg.data.camera_keys)*64+7)),"action_controls":action_steps,"action_scalar_tokens":action_steps*7}; result["network_calls"]=network_calls
    result["compile"]="disabled"; result["attention_backend"]="PyTorch scaled_dot_product_attention / MultiheadAttention automatic CUDA backend"
    result["feature_cache_scope"]=("policy timings consume frozen pre-projector cache; online vision stages are separately measured" if cfg.data.observation_source=="features" else "policy sampling includes the jointly trained scratch image encoder")
    result["action_command_hz"]=30; result["action_representation"]=cfg.data.action_representation; result["replanning_interval"]="raw action delay d, independent from 30 Hz command execution" if cfg.data.action_representation=="raw" else "D=S spans (2 raw actions/span), independent from 30 Hz command execution"
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
