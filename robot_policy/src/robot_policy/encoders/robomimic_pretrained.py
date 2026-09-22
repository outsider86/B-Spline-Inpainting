from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


ROBOMIMIC_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
_OBS_PREFIX = "policy.nets.encoder.nets.obs.obs_nets."


def _camera_state(model_state: dict[str, torch.Tensor], camera: str) -> dict[str, torch.Tensor]:
    """Map a RoboMimic v0.1 VisualCore to BSPCameraEncoder parameter names."""
    prefix = f"{_OBS_PREFIX}{camera}."
    output: dict[str, torch.Tensor] = {}
    backbone_prefix = f"{prefix}vis_core.nets."
    for name, value in model_state.items():
        if name.startswith(backbone_prefix):
            output[f"backbone.{name.removeprefix(backbone_prefix)}"] = value.detach().cpu()

    learned = {
        f"{prefix}pool_net.nets.weight": "pool.maps.weight",
        f"{prefix}pool_net.nets.bias": "pool.maps.bias",
        f"{prefix}nets.3.weight": "projection.weight",
        f"{prefix}nets.3.bias": "projection.bias",
    }
    missing = [source for source in learned if source not in model_state]
    if missing:
        raise KeyError(f"RoboMimic checkpoint is missing VisualCore tensors: {missing}")
    output.update(
        {target: model_state[source].detach().cpu() for source, target in learned.items()}
    )
    if not any(name.startswith("backbone.") for name in output):
        raise KeyError(f"RoboMimic checkpoint has no backbone for camera {camera!r}")
    return output


def extract_visual_core_artifact(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Extract only the two RGB VisualCores from an official RoboMimic checkpoint.

    RoboMimic v0.1 checkpoints predate PyTorch's restricted weights-only
    loader. Call this only for a checkpoint obtained from a trusted source.
    The emitted artifact contains tensors and primitive metadata and is safe to
    reload with ``weights_only=True``.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("RoboMimic checkpoint does not contain a model state dictionary")
    camera_states = {
        camera: _camera_state(model_state, camera) for camera in ROBOMIMIC_CAMERA_KEYS
    }
    return {
        "format": "robot_policy.robomimic_visual_core.v1",
        "source_path": str(checkpoint_path),
        "source_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source_algo_name": checkpoint.get("algo_name"),
        "source_camera_keys": list(ROBOMIMIC_CAMERA_KEYS),
        "target_camera_order": ["global", "hand"],
        "input_rgb_range": "zero_one",
        "normalization": "batch",
        "spatial_keypoints": 32,
        "feature_dimension_per_camera": 64,
        "camera_states": camera_states,
    }


def load_visual_core_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path).resolve(), map_location="cpu", weights_only=True)
    if artifact.get("format") != "robot_policy.robomimic_visual_core.v1":
        raise ValueError(f"unsupported RoboMimic VisualCore artifact: {artifact.get('format')!r}")
    states = artifact.get("camera_states")
    if not isinstance(states, dict) or tuple(states) != ROBOMIMIC_CAMERA_KEYS:
        raise ValueError(
            "RoboMimic VisualCore artifact must contain agentview and eye-in-hand cameras"
        )
    return artifact
