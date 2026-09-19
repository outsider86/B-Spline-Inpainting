from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


@dataclass
class BSPObservation:
    sequence: torch.Tensor
    global_condition: torch.Tensor
    metadata: dict[str, Any]


def _group_norm(channels: int) -> nn.GroupNorm:
    if channels % 16:
        raise ValueError(f"BSP GroupNorm requires channels divisible by 16, got {channels}")
    return nn.GroupNorm(channels // 16, channels)


class SpatialSoftmax(nn.Module):
    """Robomimic-compatible learned keypoint bottleneck."""

    def __init__(self, input_channels: int = 512, keypoints: int = 32):
        super().__init__()
        self.keypoints = int(keypoints)
        self.maps = nn.Conv2d(input_channels, self.keypoints, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.maps(feature)
        batch, keypoints, height, width = feature.shape
        weights = feature.reshape(batch, keypoints, height * width).softmax(-1)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        x = x.reshape(1, 1, -1)
        y = y.reshape(1, 1, -1)
        return torch.stack([(weights * x).sum(-1), (weights * y).sum(-1)], dim=-1)


class BSPCameraEncoder(nn.Module):
    """The reference VisualCore: scratch ResNet-18 + 32-keypoint SpatialSoftmax."""

    def __init__(self, keypoints: int = 32):
        super().__init__()
        backbone = resnet18(weights=None, norm_layer=_group_norm)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = SpatialSoftmax(512, keypoints)
        self.projection = nn.Linear(keypoints * 2, 64)
        self.activation = nn.ReLU()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        keypoints = self.pool(self.backbone(image)).flatten(1)
        return self.activation(self.projection(keypoints))


class BSPObservationEncoder(nn.Module):
    """Encode raw multi-camera history exactly at the BSP policy bottleneck.

    Each camera owns an independent, non-pretrained ResNet-18. Images are
    normalized from uint8/[0,255] to [-1,1], resized to the configured square,
    randomly cropped while training and center-cropped while evaluating.
    Low-dimensional state is concatenated without an additional MLP.
    """

    def __init__(
        self,
        cameras: int,
        observation_horizon: int,
        image_size: int = 84,
        crop_size: int = 76,
        keypoints: int = 32,
        state_dim: int = 7,
    ):
        super().__init__()
        self.cameras = int(cameras)
        self.observation_horizon = int(observation_horizon)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.state_dim = int(state_dim)
        self.camera_encoders = nn.ModuleList(
            [BSPCameraEncoder(keypoints) for _ in range(self.cameras)]
        )
        self.feature_dim = self.cameras * 64 + self.state_dim
        self.global_condition_dim = self.observation_horizon * self.feature_dim

    def _normalize_and_crop(self, images: torch.Tensor) -> torch.Tensor:
        is_uint8 = images.dtype == torch.uint8
        images = images.float()
        images = images / 127.5 - 1.0 if is_uint8 else images * 2.0 - 1.0
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.crop_size == self.image_size:
            return images
        maximum = self.image_size - self.crop_size
        if self.training:
            offsets = torch.randint(0, maximum + 1, (len(images), 2)).tolist()
        else:
            offsets = [[maximum // 2, maximum // 2] for _ in range(len(images))]
        return torch.stack(
            [
                image[:, top : top + self.crop_size, left : left + self.crop_size]
                for image, (top, left) in zip(images, offsets)
            ]
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> BSPObservation:
        if images.ndim == 5:
            images = images[:, None]
        if state.ndim == 2:
            state = state[:, None]
        if images.ndim != 6:
            raise ValueError("images must be [B,T,cameras,3,H,W]")
        batch, horizon, cameras, channels, _, _ = images.shape
        if (horizon, cameras, channels) != (
            self.observation_horizon,
            self.cameras,
            3,
        ):
            raise ValueError(
                f"image history {(horizon, cameras, channels)} != configured "
                f"{(self.observation_horizon, self.cameras, 3)}"
            )
        if state.shape != (batch, horizon, self.state_dim):
            raise ValueError(
                f"state history {tuple(state.shape)} != "
                f"{(batch, horizon, self.state_dim)}"
            )
        flat = images.reshape(batch * horizon, cameras, channels, *images.shape[-2:])
        camera_features = []
        for camera, encoder in enumerate(self.camera_encoders):
            camera_features.append(encoder(self._normalize_and_crop(flat[:, camera])))
        state_flat = state.float().reshape(batch * horizon, self.state_dim)
        sequence = torch.cat([*camera_features, state_flat], dim=-1).reshape(
            batch, horizon, self.feature_dim
        )
        return BSPObservation(
            sequence=sequence,
            global_condition=sequence.flatten(1),
            metadata={
                "camera_count": cameras,
                "camera_feature_dim": 64,
                "state_dim": self.state_dim,
                "observation_horizon": horizon,
                "per_step_dim": self.feature_dim,
            },
        )
