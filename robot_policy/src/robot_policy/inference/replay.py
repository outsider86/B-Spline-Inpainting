from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from robot_policy.config import ACTIVE_ARCHITECTURES, load_config
from robot_policy.data.dataset import PreparedPolicyDataset
from .executor import ActionExecutor, TimedPlan
from .runner import PolicyRunner


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--architecture",required=True,choices=ACTIVE_ARCHITECTURES); p.add_argument("--checkpoint",required=True); p.add_argument("--delay-spans",type=int,default=1); p.add_argument("--delay-raw-actions",type=int); p.add_argument("--max-frames",type=int,default=120); p.add_argument("--output",required=True)
    a=p.parse_args(argv); cfg=load_config(a.config,[f"policy.architecture={a.architecture}"])
    data=PreparedPolicyDataset(cfg.data.prepared_path,"test"); runner=PolicyRunner(cfg,a.checkpoint); executor=ActionExecutor()
    episode=data.episode_ids[0]; indices=[i for i,(eid,_) in enumerate(data.index) if eid==episode][:a.max_frames]
    executed=[]; interval=a.delay_raw_actions if a.delay_raw_actions is not None else 2*a.delay_spans; logical_time=0.0
    if interval < 1 or interval > cfg.rtc.raw_delay_max: raise ValueError(f"replay interval must be in 1..{cfg.rtc.raw_delay_max} raw actions")
    for j,index in enumerate(indices):
        if j%interval==0:
            item=data[index]; batch={"vision_features":item["vision_features"][None],"state":item["state"][None]}
            start=logical_time; actions,meta=runner.plan(batch,episode_id=episode,delay_spans=a.delay_spans if j else 0,delay_raw_actions=interval if j else 0)
            finish=start+meta["sampling_ms"]/1000; plan=TimedPlan(actions[0],logical_time,start,finish,episode)
            executor.submit(plan,finish,committed_steps=interval if j else 0)
        action=executor.command(logical_time)
        executed.append(None if action is None else action.tolist()); logical_time+=1/cfg.data.frequency_hz
    report={"architecture":a.architecture,"action_representation":cfg.data.action_representation,"checkpoint":str(Path(a.checkpoint).resolve()),"episode":episode,"D_equals_S_spans":interval//cfg.spline.span_length_steps if cfg.data.action_representation=="bspline" else None,"phase_raw_actions":interval%cfg.spline.span_length_steps if cfg.data.action_representation=="bspline" else None,"d_equals_s_raw_actions":interval,"executed":executed,"timeline":executor.timeline,"scope":"dataset replay; not closed-loop robot success"}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({k:v for k,v in report.items() if k not in {"executed","timeline"}},indent=2))
