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


class CameraTokenResampler(nn.Module):
    """Compress one camera's frozen spatial tokens with learned ACT-style queries."""

    def __init__(self, hidden_dim: int, heads: int, output_tokens: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_tokens, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.queries[None].expand(len(memory), -1, -1)
        update = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        result = queries + update
        return result + self.feed_forward(self.output_norm(result))


class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        state_dim: int,
        hidden_dim: int,
        cameras: int,
        tokens_per_camera: int,
        observation_horizon: int = 1,
        resampler_tokens_per_camera: int | None = None,
        resampler_heads: int = 4,
    ):
        super().__init__()
        self.cameras = cameras
        self.input_tokens_per_camera = tokens_per_camera
        self.tokens_per_camera = resampler_tokens_per_camera or tokens_per_camera
        self.observation_horizon = observation_horizon
        self.vision_projector = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.state_tokenizer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.camera_embedding = nn.Parameter(torch.randn(cameras, hidden_dim) * 0.02)
        self.spatial_embedding = nn.Parameter(torch.randn(tokens_per_camera, hidden_dim) * 0.02)
        self.state_modality = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.resamplers = (
            nn.ModuleList(
                [
                    CameraTokenResampler(
                        hidden_dim, resampler_heads, self.tokens_per_camera
                    )
                    for _ in range(cameras)
                ]
            )
            if resampler_tokens_per_camera is not None
            else None
        )
        if self.resamplers is not None:
            self.spatial_embedding = nn.Parameter(
                torch.randn(self.tokens_per_camera, hidden_dim) * 0.02
            )

    def forward(self, fused_features: torch.Tensor, normalized_state: torch.Tensor) -> ObservationTokens:
        # Keep the historical h1 checkpoint contract while giving h2 an
        # explicit time dimension.  Time embeddings are deterministic so h1
        # state dicts remain byte-for-byte load compatible.
        if fused_features.ndim == 4:
            fused_features = fused_features[:, None]
        if normalized_state.ndim == 2:
            normalized_state = normalized_state[:, None]
        if fused_features.ndim != 5:
            raise ValueError(
                "fused_features must be [B,T,cameras,patches,vision_dim]"
            )
        b, t, c, n, _ = fused_features.shape
        expected = (
            self.observation_horizon,
            self.cameras,
            self.input_tokens_per_camera,
        )
        if (t, c, n) != expected:
            raise ValueError(f"feature layout {(t,c,n)} != configured {expected}")
        if normalized_state.shape != (b, t, self.state_tokenizer[0].in_features):
            raise ValueError(
                f"state history {tuple(normalized_state.shape)} != configured "
                f"{(b, t, self.state_tokenizer[0].in_features)}"
            )
        vision = self.vision_projector(fused_features.float())
        if self.resamplers is not None:
            camera_tokens = []
            for camera, resampler in enumerate(self.resamplers):
                memory = vision[:, :, camera].flatten(0, 1)
                camera_tokens.append(
                    resampler(memory).reshape(b, t, self.tokens_per_camera, -1)
                )
            vision = torch.stack(camera_tokens, dim=2)
        temporal = self._temporal_embedding(t, vision.shape[-1], vision.device, vision.dtype)
        vision = (
            vision
            + self.camera_embedding[None, None, :, None]
            + self.spatial_embedding[None, None, None]
            + temporal[None, :, None, None]
        )
        state = (
            self.state_tokenizer(normalized_state.float())
            + self.state_modality[None]
            + temporal[None]
        )
        # Time-major order: all camera patches and then the state token for
        # t-1, followed by the same layout for t.
        per_timestep = torch.cat([vision.flatten(2, 3), state[:, :, None]], dim=2)
        tokens = per_timestep.flatten(1, 2)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return ObservationTokens(
            tokens,
            valid,
            {
                "camera_count": c,
                "tokens_per_camera": n,
                "output_tokens_per_camera": self.tokens_per_camera,
                "observation_horizon": t,
                "state_tokens": t,
            },
        )

    @staticmethod
    def _temporal_embedding(
        length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if length == 1:
            return torch.zeros(1, dim, device=device, dtype=dtype)
        position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        half = max(dim // 2, 1)
        scale = torch.exp(
            -torch.log(torch.tensor(10_000.0, device=device))
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        embedding = torch.cat(
            [torch.sin(position * scale), torch.cos(position * scale)], dim=-1
        )[:, :dim]
        if embedding.shape[-1] < dim:
            embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
        return embedding.to(dtype=dtype)
