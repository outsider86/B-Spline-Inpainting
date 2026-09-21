from __future__ import annotations

from typing import Any

import torch
from torch import nn

from robot_policy.rtc.pigdm import hard_mask_pigdm_sample

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
        if t.ndim == 2:
            future_time = t[:, -1:].expand(-1, future.shape[1])
            t = torch.cat([future_time, t], dim=1)
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
        time_map = t[:, None].expand(-1, target.shape[1])
        fixed = None
        if rtc is not None:
            fixed = rtc["fixed_mask"].bool()
            time_map = torch.where(
                fixed.any(dim=-1), torch.ones_like(time_map), time_map
            )
        x = (1 - time_map[..., None]) * noise + time_map[..., None] * target
        velocity_target = target - noise
        mutable = batch["control_valid_mask"].bool()
        if rtc is not None:
            # Hard conditioning: only the representation-aware fixed support
            # is visible; all later control rows retain their corrupted values.
            x = torch.where(fixed, rtc["prefix_values"].float(), x)
            mutable &= ~fixed
        pred = self.velocity(x, obs, time_map)
        mse = ((pred - velocity_target) ** 2)[mutable].mean()
        endpoint = x + (1 - time_map[..., None]) * pred
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

    def sample_realtime_pigdm(
        self,
        batch,
        *,
        prefix_values: torch.Tensor,
        fixed_mask: torch.Tensor,
        steps: int | None = None,
        max_guidance_weight: float = 5.0,
    ) -> torch.Tensor:
        """Base-FM RTC using reference PiGDM with a binary hard mask."""
        steps = int(steps or self.cfg.policy.fm_steps)
        with torch.no_grad():
            obs = self.observations(batch)
            noise = torch.randn(
                len(obs.tokens), self.num_basis, self.action_dim,
                device=obs.tokens.device,
            )
        return hard_mask_pigdm_sample(
            noise,
            lambda state, time: self.velocity(state, obs, time),
            prefix_values.float(),
            fixed_mask.bool(),
            steps=steps,
            max_guidance_weight=max_guidance_weight,
        )
