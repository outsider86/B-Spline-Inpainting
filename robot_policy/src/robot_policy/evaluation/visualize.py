from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_policy.config import load_config
from robot_policy.data.dataset import PreparedPolicyDataset
from robot_policy.encoders.vision import FrozenDinoSigLIP
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


def _video_frame(path: Path, index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index: return frame.to_ndarray(format="rgb24")
    raise IndexError(index)


def _save_data_artifacts(cfg, out: Path, episode: int, frame: int):
    root=Path(cfg.data.dataset_path); prepared=Path(cfg.data.prepared_path); item=np.load(prepared/"actions"/f"episode_{episode:06d}.npz")
    frames=[_video_frame(root/"videos"/"chunk-000"/cam/f"episode_{episode:06d}.mp4",frame) for cam in cfg.data.camera_keys]
    vision_root=Path(cfg.data.vision_cache_path).resolve() if cfg.data.vision_cache_path else prepared/"vision"
    features=np.load(vision_root/f"episode_{episode:06d}.npy",mmap_mode="r")[frame].astype(np.float32)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for c in range(2):
        axes[c,0].imshow(frames[c]); axes[c,0].set_title(f"{cfg.data.camera_keys[c]} original 640x480")
        axes[c,1].imshow(np.asarray(torch.nn.functional.interpolate(torch.from_numpy(frames[c]).permute(2,0,1)[None].float(),(224,224),mode="bicubic",align_corners=False,antialias=True)[0].permute(1,2,0).clamp(0,255),dtype=np.uint8)); axes[c,1].set_title("shared geometry 224x224")
        x=features[c]; centered=x-x.mean(0); _,_,vh=np.linalg.svd(centered,full_matrices=False); pca=(centered@vh[0]).reshape(4,4)
        axes[c,2].imshow(pca,cmap="coolwarm"); axes[c,2].set_title("cached fused-feature PCA (4x4)")
        for ax in axes[c]: ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"m1_vision_alignment.png",dpi=160); plt.close(fig)

    all_state=[]
    for path in sorted((prepared/"actions").glob("episode_*.npz")):
        with np.load(path) as z: all_state.append(z["state"])
    state=np.concatenate(all_state); stats=json.loads((prepared/"normalization.json").read_text()); mean=np.asarray(stats["state_mean"]); std=np.asarray(stats["state_std"]); normalized=(state-mean)/std
    fig,axes=plt.subplots(2,4,figsize=(14,6));
    for d,ax in enumerate(axes.flat[:7]): ax.hist(state[:,d],bins=50,alpha=.55,label="raw"); ax2=ax.twinx(); ax2.hist(normalized[:,d],bins=50,alpha=.35,color="tab:orange",label="normalized"); ax.set_title(f"state {d}")
    axes.flat[7].axis("off"); fig.tight_layout(); fig.savefig(out/"m1_state_distributions.png",dpi=160); plt.close(fig)

    gt=item["normalized_action"][np.minimum(np.arange(frame,frame+30),len(item["normalized_action"])-1)]
    fit=item["reconstruction"][frame]; quant=item["quantized_reconstruction"][frame]
    fig,axes=plt.subplots(4,2,figsize=(13,12)); t=np.arange(30)/30
    for d,ax in enumerate(axes.flat[:7]):
        ax.plot(t,gt[:,d],label="normalized source",lw=2); ax.plot(t,fit[:,d],label="representation reconstruction"); ax.plot(t,quant[:,d],label="quantized",ls="--")
        if cfg.data.action_representation == "bspline":
            for boundary in np.arange(0,31,2)/30: ax.axvline(boundary,color="gray",alpha=.12)
        ax.set_title(f"action dimension {d}")
    axes.flat[0].legend(ncol=3,fontsize=8); axes.flat[7].axis("off"); fig.tight_layout(); fig.savefig(out/("m2_spline_reconstruction.png" if cfg.data.action_representation=="bspline" else "m2_raw_action_quantization.png"),dpi=160); plt.close(fig)

    # Explicit cubic support diagram for D=1..5.
    if cfg.data.action_representation == "bspline":
        support=np.zeros((5,18))
        for d in range(1,6): support[d-1,:d+3]=1
        labels=[f"D=S={d}" for d in range(1,6)]; title="Committed spline support (adjacent spans overlap by 3 controls)"; xlabel="control-point index"; filename="m5_delay_support.png"
    else:
        support=np.zeros((10,30))
        for d in range(1,11): support[d-1,:d]=1
        labels=[f"d={d}" for d in range(1,11)]; title="Committed raw-action prefixes"; xlabel="raw action index"; filename="m5_raw_delay_support.png"
    fig,ax=plt.subplots(figsize=(10,3.5)); ax.imshow(support,aspect="auto",cmap="Blues",vmin=0,vmax=1); ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(support.shape[1]),xlabel=xlabel,title=title); fig.tight_layout(); fig.savefig(out/filename,dpi=160); plt.close(fig)


@torch.inference_mode()
def _cache_check(cfg, episode, frame):
    device=torch.device("cuda"); model=FrozenDinoSigLIP(cfg).to(device).eval(); root=Path(cfg.data.dataset_path); images=[]
    def fingerprint():
        return [[float(flat[0]),float(flat[len(flat)//2]),float(flat[-1])] for p in model.parameters() for flat in [p.detach().flatten()] if len(flat)]
    fingerprint_before=fingerprint()
    for cam in cfg.data.camera_keys: images.append(_video_frame(root/"videos"/"chunk-000"/cam/f"episode_{episode:06d}.mp4",frame))
    online=[]
    for image in images: online.append(model(torch.from_numpy(image[None]).to(device)).fused_patches[0].cpu().float().numpy())
    vision_root=Path(cfg.data.vision_cache_path).resolve() if cfg.data.vision_cache_path else Path(cfg.data.prepared_path)/"vision"
    cached=np.load(vision_root/f"episode_{episode:06d}.npy",mmap_mode="r")[frame].astype(np.float32)
    online=np.stack(online); delta=online-cached; error=np.abs(delta); rmse=float(np.sqrt(np.mean(delta**2))); scale=float(np.sqrt(np.mean(online**2)))
    cosine=float(np.sum(online*cached)/(np.linalg.norm(online)*np.linalg.norm(cached)))
    relative_rmse=rmse/max(scale,1e-12)
    return {"shape":list(cached.shape),"dtype_on_disk":"float16","max_abs":float(error.max()),"mean_abs":float(error.mean()),
            "rmse":rmse,"online_rms":scale,"relative_rmse":relative_rmse,"cosine_similarity":cosine,
            "strict_elementwise_allclose_atol_0.01":bool(np.allclose(online,cached,atol=.01,rtol=.01)),
            "storage_precision_acceptance":{"relative_rmse_lt_0.01":relative_rmse<.01,"cosine_gt_0.999":cosine>.999},
            "backbones_require_grad":any(p.requires_grad for p in model.parameters()),
            "sampled_parameter_fingerprint_unchanged":fingerprint_before==fingerprint(),
            "interpretation":"cross-process BF16 kernels plus float16 cache are not elementwise deterministic; acceptance uses relative RMS and cosine while retaining strict-allclose evidence"}


@torch.inference_mode()
def _policy_artifacts(cfg_path, checkpoint_args, out, episode, frame):
    dataset_cache={}; predictions={}; histories={}; parameters={}; scales={}
    for value in checkpoint_args:
        # Labels should be architecture:stage when passed; accept explicit architecture,stage,path with commas.
        if "," in value:
            architecture,label,path=value.split(",",2)
        else:
            label,path=value.split("=",1); architecture=label.split("_")[0] if not label.startswith("discrete_") else ("discrete_joint" if label.startswith("discrete_joint") else "discrete_layerwise")
        cfg=load_config(cfg_path,[f"policy.architecture={architecture}"]); model,payload=load_policy_checkpoint(path,cfg,"cuda"); codec=create_action_codec(cfg,torch.device("cuda"))
        data=dataset_cache.setdefault(architecture,PreparedPolicyDataset(cfg.data.prepared_path,"test")); idx=data.index.index((episode,frame)); item=data[idx]; batch={"vision_features":item["vision_features"][None].cuda(),"state":item["state"][None].cuda()}
        torch.manual_seed(20260915); pred=model.sample(batch); controls=pred.float() if architecture=="fm" else codec.decode_tokens(pred); predictions[label]=codec.decode_controls(controls)[0].cpu().numpy(); histories[label]=payload["history"]; parameters[label]=payload["parameter_counts"]
        obs=model.observations(batch); scales[label]={"observation_token_mean":float(obs.tokens.mean()),"observation_token_std":float(obs.tokens.std())}
    gt=np.load(Path(cfg.data.prepared_path)/"actions"/f"episode_{episode:06d}.npz")["normalized_action"]; gt=gt[np.minimum(np.arange(frame,frame+30),len(gt)-1)]
    fig,axes=plt.subplots(4,2,figsize=(14,12),layout="constrained"); t=np.arange(30)/30
    for d,ax in enumerate(axes.flat[:7]):
        ax.plot(t,gt[:,d],color="black",lw=2,label="ground truth")
        for label,pred in predictions.items(): ax.plot(t,pred[:,d],alpha=.8,label=label)
        ax.set_title(f"normalized action {d}")
    axes.flat[0].legend(fontsize=7,ncol=2); axes.flat[7].axis("off"); fig.savefig(out/"m4_m5_checkpoint_overlays.png",dpi=160); plt.close(fig)
    for label,pred in predictions.items():
        fig,axes=plt.subplots(4,2,figsize=(13,12),layout="constrained")
        for d,ax in enumerate(axes.flat[:7]):
            ax.plot(t,gt[:,d],color="black",lw=2,label="ground truth")
            ax.plot(t,pred[:,d],color="tab:blue",lw=1.6,label=label)
            ax.set_title(f"normalized action {d}"); ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8); axes.flat[7].axis("off")
        fig.savefig(out/f"trajectory_{label}.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout="constrained")
    for label,history in histories.items():
        if history:
            updates=[x["update"] for x in history]
            axes[0].plot(updates,[x["validation"].get("action_mse",x["validation"]["loss"]) for x in history],label=label)
            axes[1].plot(updates,[x["validation"]["loss"] for x in history],label=label)
    axes[0].set(xlabel="optimizer update",ylabel="normalized action MSE",yscale="log",title="Validation action MSE")
    axes[1].set(xlabel="optimizer update",ylabel="training objective",yscale="log",title="Validation objective")
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out/"training_curves.png",dpi=160); plt.close(fig)
    (out/"parameter_counts.json").write_text(json.dumps(parameters,indent=2)+"\n"); (out/"tokenizer_scales.json").write_text(json.dumps(scales,indent=2)+"\n")


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--output",default="outputs/visualizations"); p.add_argument("--episode",type=int,default=8); p.add_argument("--frame",type=int,default=200); p.add_argument("--checkpoint",action="append",default=[],help="architecture,label,path")
    a=p.parse_args(argv); cfg=load_config(a.config); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); _save_data_artifacts(cfg,out,a.episode,a.frame)
    check=_cache_check(cfg,a.episode,a.frame); (out/"m1_cache_check.json").write_text(json.dumps(check,indent=2)+"\n")
    if a.checkpoint: _policy_artifacts(a.config,a.checkpoint,out,a.episode,a.frame)
    print(json.dumps({"output":str(out.resolve()),"cache_check":check},indent=2))
