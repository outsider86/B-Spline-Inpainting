from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import PolicyBase
from .common import LayerwiseDiT


class FlowMatchingPolicy(PolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        d = cfg.policy.hidden_dim
        self.input_projection = nn.Linear(self.action_dim, d)
        self.future_tokens = nn.Parameter(torch.randn(32, d) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(self.num_basis, d) * 0.02)
        self.backbone = LayerwiseDiT(d, cfg.policy.depth, cfg.policy.heads, cfg.policy.mlp_ratio, cfg.policy.dropout)
        self.output_projection = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, self.action_dim))

    def velocity(self, x: torch.Tensor, obs, t: torch.Tensor) -> torch.Tensor:
        actions = self.input_projection(x) + self.position_embedding[None]
        future = self.future_tokens[None].expand(len(x), -1, -1)
        hidden = torch.cat([future, actions], dim=1)
        hidden = self.backbone(hidden, obs.tokens, t * 1000, obs.valid_mask)
        return self.output_projection(hidden[:, -self.num_basis:])

    def loss(self, batch, rtc=None):
        target = batch["continuous_target"].float()
        obs = self.observations(batch)
        beta = torch.distributions.Beta(target.new_tensor(1.5), target.new_tensor(1.0))
        sampled = beta.sample((len(target),))
        # StarVLA's formula produces rare negative values when Beta samples
        # exceed noise_s. Clamp that audited source bug at the valid boundary.
        t = ((0.999 - sampled) / 0.999).clamp(0, 1)
        noise = torch.randn_like(target)
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        velocity_target = target - noise
        mutable = batch["control_valid_mask"].bool()
        if rtc is not None:
            # Hard conditioning: only the representation-aware fixed support
            # is visible; all later control rows retain their corrupted values.
            fixed = rtc["fixed_mask"].bool()
            x = torch.where(fixed, rtc["prefix_values"].float(), x)
            mutable &= ~fixed
        pred = self.velocity(x, obs, t)
        mse = ((pred - velocity_target) ** 2)[mutable].mean()
        endpoint = x + (1 - t[:, None, None]) * pred
        action_mse = self.decoded_action_mse(endpoint, batch, mutable)
        return {"loss": mse, "fm_mse": mse, "action_mse": action_mse.detach(), "gradient_scale_fm": mse.detach()}

    @torch.no_grad()
    def sample(self, batch, steps=None, prefix_values=None, fixed_mask=None, **kwargs):
        obs = self.observations(batch)
        steps = int(steps or self.cfg.policy.fm_steps)
        x = torch.randn(len(obs.tokens), self.num_basis, self.action_dim, device=obs.tokens.device)
        dt = 1.0 / steps
        for i in range(steps):
            if prefix_values is not None:
                # Re-apply the hard mask before and after every Euler step.
                x = torch.where(fixed_mask, prefix_values, x)
            t = x.new_full((len(x),), i / steps)
            x = x + dt * self.velocity(x, obs, t)
            if prefix_values is not None:
                x = torch.where(fixed_mask, prefix_values, x)
        return x
