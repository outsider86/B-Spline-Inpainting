from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from robot_policy.config import ACTIVE_ARCHITECTURES, TRAINABLE_ARCHITECTURES, load_config


def _parser(command: str) -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog=command); parser.add_argument("--config",default="configs/default.yaml"); parser.add_argument("--set",action="append",default=[])
    return parser


def _cfg(args, architecture=None):
    overrides=list(args.set)
    if architecture: overrides.append(f"policy.architecture={architecture}")
    return load_config(args.config,overrides)


def audit_data(argv=None):
    from robot_policy.data.lerobot_adapter import audit_dataset
    p=_parser("audit_data"); a=p.parse_args(argv); print(json.dumps(audit_dataset(_cfg(a)),indent=2))


def prepare_actions(argv=None):
    from robot_policy.data.lerobot_adapter import prepare_action_targets
    p=_parser("prepare_actions"); a=p.parse_args(argv); print(json.dumps(prepare_action_targets(_cfg(a))["metrics"],indent=2))


def prepare_observations(argv=None):
    from robot_policy.encoders.vision import prepare_vision_features
    p=_parser("prepare_observations"); p.add_argument("--rank",type=int,default=0); p.add_argument("--world-size",type=int,default=1); p.add_argument("--batch-size",type=int,default=16)
    a=p.parse_args(argv); print(json.dumps(prepare_vision_features(_cfg(a),a.rank,a.world_size,a.batch_size),indent=2))


def prepare_rgb_cache(argv=None):
    from robot_policy.data.rgb_cache import prepare_rgb_cache as build
    p=_parser("prepare_rgb_cache"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--rank",type=int,default=0); p.add_argument("--world-size",type=int,default=1)
    a=p.parse_args(argv); print(json.dumps(build(_cfg(a),a.overwrite,a.rank,a.world_size),indent=2))


def smoke_test(argv=None):
    from robot_policy.policies import create_policy
    p=_parser("smoke_test"); a=p.parse_args(argv)
    for architecture in ACTIVE_ARCHITECTURES:
        cfg=_cfg(a,architecture); model=create_policy(cfg); steps=cfg.spline.num_basis if cfg.data.action_representation=="bspline" else cfg.data.action_horizon
        batch={"vision_features":torch.randn(2,cfg.data.observation_horizon,len(cfg.data.camera_keys),cfg.vision.pooled_grid**2,cfg.vision.feature_dim),"state":torch.randn(2,cfg.data.observation_horizon,7),"continuous_target":torch.randn(2,steps,7),"discrete_target":torch.randint(0,256,(2,steps,7)),"control_valid_mask":torch.ones(2,steps,7,dtype=torch.bool)}
        result=model.loss(batch); result["loss"].backward(); sample=model.sample(batch,steps=2,rounds=2)
        print(architecture,{k:float(v) for k,v in result.items()},tuple(sample.shape))


def _train_cli(rtc: bool, argv=None):
    from robot_policy.training import train
    p=_parser("finetune_rtc" if rtc else "train_base"); p.add_argument("--architecture",required=True,choices=TRAINABLE_ARCHITECTURES); p.add_argument("--output",required=True); p.add_argument("--resume"); p.add_argument("--wandb-resume"); p.add_argument("--fresh-wandb",action="store_true"); p.add_argument("--updates",type=int)
    if rtc: p.add_argument("--parent",required=True)
    a=p.parse_args(argv)
    if a.fresh_wandb and a.wandb_resume:
        p.error("--fresh-wandb and --wandb-resume are mutually exclusive")
    wandb_resume_info=json.loads(Path(a.wandb_resume).read_text()) if a.wandb_resume else None
    entry=train(_cfg(a,a.architecture),a.output,parent_checkpoint=getattr(a,"parent",None),resume=a.resume,updates=a.updates,
                wandb_resume_info=wandb_resume_info,fresh_wandb=a.fresh_wandb)
    if entry: print(json.dumps(entry,indent=2))


def train_base(argv=None): _train_cli(False,argv)
def finetune_rtc(argv=None): _train_cli(True,argv)


def evaluate_open_loop(argv=None):
    from robot_policy.evaluation.open_loop import main
    main(argv)
def benchmark_inference(argv=None):
    from robot_policy.evaluation.latency import main
    main(argv)
def evaluate_rtc(argv=None):
    from robot_policy.evaluation.rtc import main
    main(argv)
def evaluate_inference_rtc(argv=None):
    from robot_policy.evaluation.inference_rtc import main
    main(argv)
def visualize_trajectories(argv=None):
    from robot_policy.evaluation.visualize import main
    main(argv)
def replay_rtc(argv=None):
    from robot_policy.inference.replay import main
    main(argv)
def build_report(argv=None):
    from robot_policy.evaluation.reporting import main
    main(argv)
def finalize_checkpoints(argv=None):
    from robot_policy.evaluation.finalize import main
    main(argv)
def verify_reproducibility(argv=None):
    from robot_policy.evaluation.reproducibility import main
    main(argv)
def cache_parent_predictions(argv=None):
    from robot_policy.data.parent_predictions import main
    main(argv)
def compare_representations(argv=None):
    from robot_policy.evaluation.comparison import main
    main(argv)


if __name__ == "__main__":
    commands={name:globals()[name] for name in ("audit_data","prepare_actions","prepare_observations","prepare_rgb_cache","smoke_test","train_base","finetune_rtc","evaluate_open_loop","evaluate_rtc","evaluate_inference_rtc","benchmark_inference","visualize_trajectories","replay_rtc","build_report","finalize_checkpoints","verify_reproducibility","cache_parent_predictions","compare_representations")}
    if len(sys.argv)<2 or sys.argv[1] not in commands: raise SystemExit("usage: python -m robot_policy.cli {"+",".join(commands)+"} ...")
    commands[sys.argv[1]](sys.argv[2:])
