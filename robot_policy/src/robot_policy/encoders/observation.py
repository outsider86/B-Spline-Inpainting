from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ObservationTokens:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    metadata: dict[str, Any]


class ObservationTokenizer(nn.Module):
    def __init__(self, vision_dim: int, state_dim: int, hidden_dim: int, cameras: int, tokens_per_camera: int):
        super().__init__()
        self.cameras = cameras
        self.tokens_per_camera = tokens_per_camera
        self.vision_projector = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.state_tokenizer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.camera_embedding = nn.Parameter(torch.randn(cameras, hidden_dim) * 0.02)
        self.spatial_embedding = nn.Parameter(torch.randn(tokens_per_camera, hidden_dim) * 0.02)
        self.state_modality = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self, fused_features: torch.Tensor, normalized_state: torch.Tensor) -> ObservationTokens:
        if fused_features.ndim != 4:
            raise ValueError("fused_features must be [B,cameras,patches,vision_dim]")
        b, c, n, _ = fused_features.shape
        if (c, n) != (self.cameras, self.tokens_per_camera):
            raise ValueError(f"feature layout {(c,n)} != configured {(self.cameras,self.tokens_per_camera)}")
        vision = self.vision_projector(fused_features.float())
        vision = vision + self.camera_embedding[None, :, None] + self.spatial_embedding[None, None]
        state = self.state_tokenizer(normalized_state.float())[:, None] + self.state_modality[None]
        tokens = torch.cat([vision.flatten(1, 2), state], dim=1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return ObservationTokens(tokens, valid, {"camera_count": c, "tokens_per_camera": n, "state_tokens": 1})

