from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / max(half, 1))
    args = t.float()[:, None] * freqs[None]
    out = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(out, (0, dim - out.shape[-1]))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(t, self.dim))


class AdaNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(time).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class LayerwiseBlock(nn.Module):
    """Compact StarVLA-style alternating cross/self-attention DiT block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, self_attention: bool):
        super().__init__()
        self.self_attention = self_attention
        self.norm1 = AdaNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor, obs: torch.Tensor, time: torch.Tensor, obs_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.norm1(x, time)
        if self.self_attention:
            update = self.attn(q, q, q, need_weights=False)[0]
        else:
            update = self.attn(q, obs, obs, key_padding_mask=obs_padding_mask, need_weights=False)[0]
        x = x + update
        return x + self.ff(self.norm2(x))


class LayerwiseDiT(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.time = TimeEmbedding(dim)
        self.blocks = nn.ModuleList([
            LayerwiseBlock(dim, heads, mlp_ratio, dropout, self_attention=(i % 2 == 1)) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, obs: torch.Tensor, t: torch.Tensor, obs_valid: torch.Tensor | None = None) -> torch.Tensor:
        temb = self.time(t)
        padding = None if obs_valid is None else ~obs_valid
        for block in self.blocks:
            x = block(x, obs, temb, padding)
        return self.norm(x)


def expected_token_distance(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Required CE + E[|K-y|], both averaged over the same valid positions."""
    if logits.shape[-1] != 256:
        raise ValueError("expected-token-distance is defined on exactly 256 action bins")
    target = target.long()
    valid = valid.bool()
    if valid.shape != target.shape:
        valid = torch.broadcast_to(valid, target.shape)
    if not valid.any():
        raise ValueError("loss has no valid supervised action positions")
    ce_all = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="none").reshape(target.shape)
    probs = logits.float().softmax(dim=-1)
    positions = torch.arange(256, device=logits.device, dtype=probs.dtype)
    distance = (positions.view(*([1] * target.ndim), 256) - target[..., None]).abs()
    l1_all = (probs * distance).sum(dim=-1)
    return ce_all[valid].mean(), l1_all[valid].mean()


def cosine_corruption(tokens: torch.Tensor, valid: torch.Tensor, mask_id: int = 256, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    b, length = tokens.shape
    ratio = 1 - torch.cos(torch.pi * 0.5 * torch.rand(b, device=tokens.device, generator=generator))
    count = valid.sum(1)
    num_mask = (count * ratio).round().long().clamp(min=1)
    scores = torch.rand(tokens.shape, device=tokens.device, generator=generator).masked_fill(~valid, 2.0)
    ranks = scores.argsort(1).argsort(1)
    masked = (ranks < num_mask[:, None]) & valid
    return torch.where(masked, torch.full_like(tokens, mask_id), tokens), masked


def monotonic_block_corruption(tokens: torch.Tensor, valid: torch.Tensor, block_size: int, mask_id: int = 256, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """D2F monotonic block corruption, adapted to hard action-bin targets."""
    b, length = tokens.shape
    blocks = math.ceil(length / block_size)
    p0 = 0.2 + 0.5 * torch.rand(b, 1, device=tokens.device, generator=generator)
    if blocks > 1:
        increments = torch.rand(b, blocks - 1, device=tokens.device, generator=generator)
        increments = increments / increments.sum(1, keepdim=True).clamp_min(1e-8) * (0.7 - p0)
        probs = torch.cat([p0, p0 + increments.cumsum(1)], dim=1)
    else:
        probs = p0
    block_ids = torch.arange(length, device=tokens.device) // block_size
    masked = torch.rand(tokens.shape, device=tokens.device, generator=generator) < probs[:, block_ids]
    masked &= valid
    # Guarantee at least one supervised token per item.
    for i in range(b):
        if not masked[i].any():
            masked[i, torch.nonzero(valid[i], as_tuple=False)[0, 0]] = True
    return torch.where(masked, torch.full_like(tokens, mask_id), tokens), masked


def maskgit_update(logits: torch.Tensor, current: torch.Tensor, mutable: torch.Tensor, step: int, rounds: int) -> torch.Tensor:
    probs = logits.float().softmax(-1)
    sampled = probs.argmax(-1)
    unknown = (current == 256) & mutable
    proposed = torch.where(unknown, sampled, current)
    if step == rounds - 1:
        return proposed
    confidence = probs.gather(-1, sampled[..., None]).squeeze(-1)
    confidence = confidence.masked_fill(~unknown, float("inf"))
    initial = mutable.sum(1)
    keep = (initial.float() * (0.5 * (1 + torch.cos(torch.tensor(math.pi * (step + 1) / rounds, device=logits.device))))).long()
    out = proposed.clone()
    for i in range(len(out)):
        candidates = torch.nonzero(unknown[i], as_tuple=False).flatten()
        n = min(int(keep[i]), len(candidates))
        if n:
            remask = candidates[confidence[i, candidates].argsort()[:n]]
            out[i, remask] = 256
    return out


def parameter_groups(model: nn.Module) -> dict[str, int]:
    groups = {"observation_projector_state": 0, "action_backbone": 0, "embedding_output": 0, "total_trainable": 0, "total": 0}
    for name, p in model.named_parameters():
        groups["total"] += p.numel()
        if p.requires_grad:
            groups["total_trainable"] += p.numel()
            if name.startswith("observation"):
                groups["observation_projector_state"] += p.numel()
            elif any(key in name for key in ("input", "embedding", "position", "output")):
                groups["embedding_output"] += p.numel()
            else:
                groups["action_backbone"] += p.numel()
    return groups

