from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_policy.config import (
    ACTIVE_ARCHITECTURES,
    load_config,
    require_active_architecture,
    require_active_model_size,
)
from robot_policy.data.dataset import PreparedPolicyDataset, collate_policy_batch
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


def _summary(values: list[np.ndarray]) -> dict[str, float] | None:
    if not values:
        return None
    x=np.concatenate([v.reshape(-1) for v in values])
    return {"mae":float(np.abs(x).mean()),"mse":float(np.mean(x*x)),"rmse":float(np.sqrt(np.mean(x*x))),"max_abs":float(np.abs(x).max()),
            "p95_abs":float(np.quantile(np.abs(x),.95)),"p99_abs":float(np.quantile(np.abs(x),.99))}


def _stack(dataset, indices, device):
    batch=collate_policy_batch([dataset[i] for i in indices])
    return {k:v.to(device) for k,v in batch.items()}


@torch.inference_mode()
def evaluate(cfg, checkpoint: str, max_samples: int = 128, batch_size: int = 32) -> dict[str, Any]:
    require_active_architecture(cfg.policy.architecture, "RTC evaluation")
    require_active_model_size(cfg.policy.model_size, "RTC evaluation")
    device=torch.device("cuda"); model,payload=load_policy_checkpoint(checkpoint,cfg,device); model.eval()
    codec=create_action_codec(cfg,device); data=PreparedPolicyDataset(cfg.data.prepared_path,"test")
    lookup={pair:i for i,pair in enumerate(data.index)}
    current_indices=[i for i,(eid,frame) in enumerate(data.index) if frame>=cfg.rtc.raw_delay_max][:max_samples]
    stats=json.loads((Path(cfg.data.prepared_path)/"normalization.json").read_text())
    low=torch.tensor(stats["action_q01"],device=device); high=torch.tensor(stats["action_q99"],device=device)
    curves=[]
    for raw_delay in range(0,cfg.rtc.raw_delay_max+1):
        mapping=map_delay(raw_delay,frequency_hz=cfg.data.frequency_hz,span_length_steps=cfg.spline.span_length_steps,degree=cfg.spline.degree) if cfg.data.action_representation=="bspline" else None
        trajectory_errors=[]; preservation=[]; requantization=[]; fixed_errors=[]; switch_velocity=[]; switch_acceleration=[]
        torch.manual_seed(20260915+raw_delay)
        for offset in range(0,len(current_indices),batch_size):
            chosen=current_indices[offset:offset+batch_size]; current=_stack(data,chosen,device)
            fixed=prefix=shifted=prior_actions=None
            if raw_delay:
                previous_indices=[lookup[(data.index[i][0],data.index[i][1]-raw_delay)] for i in chosen]
                previous=_stack(data,previous_indices,device)
                prior=model.sample(previous)
                prior_controls=prior.float() if cfg.policy.architecture=="fm" else codec.decode_tokens(prior)
                raw=torch.full((len(chosen),),raw_delay,device=device,dtype=torch.long)
                shifted=codec.shift_and_refit(prior_controls,raw)
                if cfg.data.action_representation=="bspline":
                    affected=torch.full_like(raw,mapping.affected_spans)
                    fixed=control_support_mask(
                        affected,
                        cfg.spline.num_basis,
                        shifted.shape[-1],
                        cfg.spline.degree,
                    )
                else:
                    fixed=raw_action_prefix_mask(raw,cfg.data.action_horizon)
                prefix=shifted if cfg.policy.architecture=="fm" else codec.encode_tokens(shifted)
                prior_actions=codec.decode_controls(prior_controls)
            predicted=model.sample(current,prefix_values=prefix,fixed_mask=fixed,use_cache=True)
            controls=predicted.float() if cfg.policy.architecture=="fm" else codec.decode_tokens(predicted)
            decoded=codec.decode_controls(controls)
            physical=(decoded+1)*.5*(high-low)+low
            err=physical-current["target_trajectory"].float(); valid=current["action_valid_mask"].bool()
            trajectory_errors.extend([err[i,valid[i]].cpu().numpy() for i in range(len(err))])
            if raw_delay:
                reference_controls=shifted if cfg.policy.architecture=="fm" else codec.decode_tokens(prefix)
                reference_actions=codec.decode_controls(reference_controls); shifted_actions=codec.decode_controls(shifted)
                preservation.append((decoded[:,:raw_delay]-reference_actions[:,:raw_delay]).cpu().numpy())
                requantization.append((reference_actions[:,:raw_delay]-shifted_actions[:,:raw_delay]).cpu().numpy())
                fixed_errors.append((controls-reference_controls).masked_select(fixed).cpu().numpy())
                scale=.5*(high-low)
                jump=(decoded[:,0]-prior_actions[:,raw_delay-1])*scale*cfg.data.frequency_hz
                switch_velocity.append(jump.cpu().numpy())
                if raw_delay>=2:
                    prev_velocity=(prior_actions[:,raw_delay-1]-prior_actions[:,raw_delay-2])*scale*cfg.data.frequency_hz
                    switch_acceleration.append(((jump-prev_velocity)*cfg.data.frequency_hz).cpu().numpy())
        curves.append({"raw_delay_actions":raw_delay,"delay_ms":1000*raw_delay/cfg.data.frequency_hz,"whole_spans":mapping.whole_spans if mapping else None,
                       "phase_steps":mapping.phase_steps if mapping else None,"affected_spans":mapping.affected_spans if mapping else None,
                       "committed_control_count":mapping.committed_control_count if mapping else raw_delay,
                       "training_delay_status":"zero-delay check" if raw_delay==0 else (("seen exact-span" if raw_delay%cfg.spline.span_length_steps==0 else "unseen within-span") if mapping else "seen raw-action delay"),
                       "decoded_physical_error":_summary(trajectory_errors),"committed_interval_preservation_normalized":_summary(preservation),
                       "discrete_requantization_preservation_error_normalized":_summary(requantization),
                       "fixed_control_error":_summary(fixed_errors),"switch_velocity_physical_per_s":_summary(switch_velocity),
                       "switch_acceleration_physical_per_s2":_summary(switch_acceleration)})
    return {"architecture":cfg.policy.architecture,"action_representation":cfg.data.action_representation,"training_type":payload["training_type"],"checkpoint":str(Path(checkpoint).resolve()),
            "split":"test","samples_per_delay":len(current_indices),"delay_training_support":"raw d=1..10 actions" if cfg.data.action_representation=="raw" else "D=S=1..5 exact spans; raw d=s=2D",
            "unseen_delay_definition":None if cfg.data.action_representation=="raw" else "odd raw-action delays lie within a span and were excluded from fine-tuning draws",
            "scope":"open-loop dataset replay; no environmental feedback or robot success measurement","curves":curves}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--set",action="append",default=[]); p.add_argument("--architecture",required=True,choices=ACTIVE_ARCHITECTURES); p.add_argument("--checkpoint",required=True); p.add_argument("--max-samples",type=int,default=128); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--output",required=True)
    a=p.parse_args(argv); cfg=load_config(a.config,[*a.set,f"policy.architecture={a.architecture}"]); report=evaluate(cfg,a.checkpoint,a.max_samples,a.batch_size)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
