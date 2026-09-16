from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from robot_policy.config import load_config
from robot_policy.policies import load_policy_checkpoint


def finalize(config_path: str, directory: Path) -> dict:
    base_cfg=load_config(config_path); prepared=Path(base_cfg.data.prepared_path); splits=json.loads((prepared/"splits.json").read_text()); encoder=json.loads((prepared/"encoder.json").read_text())
    vision_root=Path(base_cfg.data.vision_cache_path) if base_cfg.data.vision_cache_path else prepared/"vision"
    vision=json.loads(next(vision_root.glob("worker_*_manifest.json")).read_text()); entries=[]; hashes={}
    rtc_stage="ttrtc" if base_cfg.data.action_representation=="raw" else "rtc"
    checkpoints=(("fm","base"),("fm",rtc_stage),("discrete_layerwise","base"),("discrete_layerwise",rtc_stage),("discrete_joint","base"),("discrete_joint",rtc_stage))
    for architecture,stage in checkpoints:
        path=directory/f"{architecture}_{stage}.pt"
        if not path.exists(): raise FileNotFoundError(path)
        payload=torch.load(path,map_location="cpu",weights_only=False); cfg=load_config(config_path,[f"policy.architecture={architecture}"])
        model,_=load_policy_checkpoint(path,cfg,"cpu")
        actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if payload["architecture"]!=architecture or payload["training_type"]!=stage: raise RuntimeError(f"identity mismatch in {path}")
        if actual!=payload["parameter_counts"]["total_trainable"]: raise RuntimeError(f"parameter-count mismatch in {path}")
        digest=sha256(path.read_bytes()).hexdigest(); hashes[(architecture,stage)]=digest; history=payload["history"]
        entries.append({"architecture":architecture,"training_type":stage,"parent_checkpoint":payload["parent_checkpoint"],
            "parent_checkpoint_sha256":None,"file_path":str(path.resolve()),"sha256":digest,"configuration":payload["config"],
            "code_commit":"root repository has no commit","code_snapshot_sha256":payload["code_snapshot_sha256"],"data_split":splits,
            "encoder_version":{"type":encoder["type"],"config":encoder["config"],"tokenizer_id":encoder["tokenizer_id"],"predicted_fields":encoder["predicted_fields"],"fixed_fields":encoder["fixed_fields"]},
            "vision_weight_versions":{"dino":vision["dino"],"siglip":vision["siglip"],"extraction":vision["extraction"]},
            "training_update_count":payload["update"],"samples_seen":payload["samples_seen"],"world_size":payload["world_size"],
            "wall_seconds":payload["wall_seconds"],"gpu_hours":payload["gpu_hours"],"peak_memory_bytes_per_rank":payload["peak_memory_bytes_per_rank"],
            "checkpoint_selection_metric":{"name":"final validation objective","value":history[-1]["validation"]["loss"],"best_observed_value":payload["best_validation"],"split":"val"},
            "validation_start":history[0]["validation"],"validation_final":history[-1]["validation"],"parameter_counts":payload["parameter_counts"],
            "resume_command":payload["resume_command"],"independent_cpu_reload":"passed",
            "wandb":payload.get("wandb"),"determinism":payload.get("determinism"),"loss_trace_points":len(payload.get("loss_trace",[]))})
    for entry in entries:
        if entry["training_type"] in {"rtc","ttrtc"}: entry["parent_checkpoint_sha256"]=hashes[(entry["architecture"],"base")]
    result={"manifest_version":2,"checkpoint_count":len(entries),"selection_policy":"final-update weights; validation/test never used to select or alter saved weights","checkpoints":entries}
    (directory/"checkpoint_manifest.json").write_text(json.dumps(result,indent=2)+"\n"); return result


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--directory",default="outputs/checkpoints"); a=p.parse_args(argv); result=finalize(a.config,Path(a.directory)); print(json.dumps({"checkpoint_count":result["checkpoint_count"],"reload":"passed"},indent=2))


if __name__ == "__main__": main()
