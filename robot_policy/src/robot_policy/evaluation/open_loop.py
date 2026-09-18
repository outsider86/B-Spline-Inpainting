from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_policy.config import ACTIVE_ARCHITECTURES, load_config, require_active_architecture
from torch.utils.data import DataLoader, Subset

from robot_policy.data.dataset import PreparedPolicyDataset, collate_policy_batch
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


def _summary(errors: np.ndarray) -> dict[str, Any]:
    flat=errors.reshape(-1)
    return {"mae":float(np.mean(np.abs(flat))),"mse":float(np.mean(flat**2)),"rmse":float(np.sqrt(np.mean(flat**2))),"max_abs":float(np.max(np.abs(flat))),
            "abs_quantiles":{str(q):float(np.quantile(np.abs(flat),q)) for q in (.5,.9,.95,.99)}}


@torch.inference_mode()
def evaluate(cfg, checkpoint: str, max_samples: int | None, batch_size: int) -> dict[str, Any]:
    require_active_architecture(cfg.policy.architecture, "open-loop evaluation")
    torch.manual_seed(20260915)
    device=torch.device("cuda"); model,payload=load_policy_checkpoint(checkpoint,cfg,device); codec=create_action_codec(cfg,device)
    dataset=PreparedPolicyDataset(cfg.data.prepared_path,"test"); count=min(len(dataset),max_samples or len(dataset)); subset=Subset(dataset,range(count))
    loader=DataLoader(subset,batch_size=batch_size,num_workers=2,collate_fn=collate_policy_batch)
    stats=json.loads((Path(cfg.data.prepared_path)/"normalization.json").read_text()); low=torch.tensor(stats["action_q01"],device=device); high=torch.tensor(stats["action_q99"],device=device)
    enc_err=[]; traj_err=[]; normalized_traj_err=[]; dim_errors=[]; tokens=[]; fit_err=[]; quant_err=[]; boundary=[]; velocities=[]; accelerations=[]
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}; predicted=model.sample(batch)
        if cfg.policy.architecture=="fm": controls=predicted.float()
        else:
            controls=codec.decode_tokens(predicted); valid=batch["control_valid_mask"].bool(); tokens.append((predicted==batch["discrete_target"]).masked_select(valid).cpu().numpy())
        valid_controls=batch["control_valid_mask"].bool(); enc_err.append((controls-batch["continuous_target"]).masked_select(valid_controls).cpu().numpy())
        norm_traj=codec.decode_controls(controls); physical=(norm_traj+1)*.5*(high-low)+low
        err=(physical-batch["target_trajectory"].float()); raw_valid=batch["action_valid_mask"].bool()[...,None].expand_as(err)
        traj_err.append(err.masked_select(raw_valid).cpu().numpy()); boundary.append(err[:,0].cpu().numpy())
        err_np=err.cpu().numpy(); valid_np=batch["action_valid_mask"].cpu().numpy()
        dim_errors.append(np.concatenate([err_np[i,valid_np[i]] for i in range(len(err_np))]))
        velocity=torch.diff(physical,dim=1)*cfg.data.frequency_hz; acceleration=torch.diff(velocity,dim=1)*cfg.data.frequency_hz
        velocities.append(velocity.cpu().numpy()); accelerations.append(acceleration.cpu().numpy())
        # Baselines are in normalized action coordinates and reported separately.
        normalized_gt=((batch["target_trajectory"]-low)*2/(high-low)-1).clamp(-1,1)
        normalized_traj_err.append((norm_traj-normalized_gt).masked_select(raw_valid).cpu().numpy())
        fit_err.append((batch["encoder_reconstruction"]-normalized_gt).masked_select(raw_valid).cpu().numpy())
        quant_err.append((batch["quantized_reconstruction"]-batch["encoder_reconstruction"]).masked_select(raw_valid).cpu().numpy())
    e=np.concatenate(traj_err); v=np.concatenate(velocities); acc=np.concatenate(accelerations)
    # Preserve physical-unit separation: six joints versus gripper.
    all_dim=[]
    de=np.concatenate(dim_errors)
    for d in range(7): all_dim.append(_summary(de[:,d]))
    report={
        "architecture":cfg.policy.architecture,"action_representation":cfg.data.action_representation,"checkpoint":str(Path(checkpoint).resolve()),"checkpoint_training_type":payload["training_type"],
        "split":"test","samples":count,"scope":"open-loop recorded observations; not closed-loop robot success",
        "sampling_protocol":{"seed_reset_per_checkpoint":20260915,"batch_size":batch_size,"fm_steps":cfg.policy.fm_steps,
                             "discrete_rounds":cfg.policy.discrete_rounds,"joint_kv_cache":True},
        "decoder_contract":"18x7 spline controls -> fixed 30x18 authoritative basis -> 30x7 normalized actions -> training-split q01/q99 physical scaling" if cfg.data.action_representation=="bspline" else "30x7 normalized actions -> identity temporal decode -> training-split q01/q99 physical scaling",
        "encoded_control_error":_summary(np.concatenate(enc_err)),"decoded_all_normalized_error":_summary(np.concatenate(normalized_traj_err)),"decoded_all_physical_error":_summary(e),
        "decoded_per_dimension":{f"joint_{i}":all_dim[i] for i in range(6)}|{"gripper":all_dim[6]},
        "boundary_first_sample_error":_summary(np.concatenate(boundary)),
        "encoder_fit_error_normalized":_summary(np.concatenate(fit_err)),"additional_quantization_error_normalized":_summary(np.concatenate(quant_err)),
        "token_accuracy":None if not tokens else float(np.concatenate(tokens).mean()),
        "velocity_abs_quantiles":{str(q):float(np.quantile(np.abs(v),q)) for q in (.5,.95,.99)},
        "acceleration_abs_quantiles":{str(q):float(np.quantile(np.abs(acc),q)) for q in (.5,.95,.99)},
        "position_rotation_metrics":"not available: dataset exposes absolute joints and gripper, not calibrated Cartesian pose",
    }
    return report


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--set",action="append",default=[]); p.add_argument("--architecture",required=True,choices=ACTIVE_ARCHITECTURES); p.add_argument("--checkpoint",required=True); p.add_argument("--max-samples",type=int); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--output",required=True)
    a=p.parse_args(argv); cfg=load_config(a.config,[*a.set,f"policy.architecture={a.architecture}"]); report=evaluate(cfg,a.checkpoint,a.max_samples,a.batch_size)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
