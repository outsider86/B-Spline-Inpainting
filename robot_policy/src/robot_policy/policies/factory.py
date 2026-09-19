from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .discrete_joint import JointDiscretePolicy
from .discrete_layerwise import LayerwiseDiscretePolicy
from .fm import FlowMatchingPolicy
from .bsp_unet import BSPUNetDiscretePolicy, BSPUNetFlowMatchingPolicy


def create_policy(cfg: Any):
    classes = {
        "fm": FlowMatchingPolicy,
        "discrete_layerwise": LayerwiseDiscretePolicy,
        "discrete_joint": JointDiscretePolicy,
        "bsp_unet_fm": BSPUNetFlowMatchingPolicy,
        "bsp_unet_discrete": BSPUNetDiscretePolicy,
    }
    return classes[cfg.policy.architecture](cfg)


def load_policy_checkpoint(path: str | Path, cfg: Any, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["architecture"] != cfg.policy.architecture:
        raise ValueError(f"checkpoint architecture {payload['architecture']} != config {cfg.policy.architecture}")
    model = create_policy(cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload
