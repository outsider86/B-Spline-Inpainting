from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from robot_policy.encoders.bsp_observation import BSPObservationEncoder
from robot_policy.rtc.pigdm import hard_mask_pigdm_sample

from .base import PolicyBase
from .common import (
    expected_token_distance,
    maskgit_update,
    mixed_block_corruption,
    timestep_embedding,
)


class Conv1DBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, groups: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, output_dim),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.blocks = nn.ModuleList(
            [
                Conv1DBlock(input_dim, output_dim, kernel_size, groups),
                Conv1DBlock(output_dim, output_dim, kernel_size, groups),
            ]
        )
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, output_dim * 2)
        )
        self.residual = (
            nn.Conv1d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        modulation = self.condition(condition)
        if modulation.ndim == 2:
            modulation = modulation.reshape(len(x), 2, self.output_dim, 1)
        elif modulation.ndim == 3:
            modulation = modulation.permute(0, 2, 1)
            if modulation.shape[-1] != x.shape[-1]:
                modulation = F.interpolate(
                    modulation, size=x.shape[-1], mode="nearest"
                )
            modulation = modulation.reshape(
                len(x), 2, self.output_dim, x.shape[-1]
            )
        else:
            raise ValueError("U-Net condition must be [B,C] or [B,T,C]")
        scale, bias = modulation.unbind(1)
        out = scale * out + bias
        return self.blocks[1](out) + self.residual(x)


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BSPConditionalUNet1D(nn.Module):
    """Reference-sized conditional temporal U-Net with a configurable head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        global_condition_dim: int,
        time_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        groups: int = 8,
    ):
        super().__init__()
        dimensions = [input_dim, *down_dims]
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        self.time_dim = time_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        condition_dim = time_dim + global_condition_dim
        self.down = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(pairs):
            final = index == len(pairs) - 1
            self.down.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, condition_dim, kernel_size, groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, condition_dim, kernel_size, groups
                        ),
                        nn.Identity() if final else Downsample1D(dim_out),
                    ]
                )
            )
        deepest = down_dims[-1]
        self.middle = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    deepest, deepest, condition_dim, kernel_size, groups
                ),
                ConditionalResidualBlock1D(
                    deepest, deepest, condition_dim, kernel_size, groups
                ),
            ]
        )
        self.up = nn.ModuleList()
        for dim_in, dim_out in reversed(pairs[1:]):
            self.up.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            condition_dim,
                            kernel_size,
                            groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, condition_dim, kernel_size, groups
                        ),
                        Upsample1D(dim_in),
                    ]
                )
            )
        first = down_dims[0]
        self.final = nn.Sequential(
            Conv1DBlock(first, first, kernel_size, groups),
            nn.Conv1d(first, output_dim, 1),
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, global_condition: torch.Tensor
    ) -> torch.Tensor:
        if sample.ndim != 3:
            raise ValueError("U-Net sample must be [B,T,C]")
        time = self.time_encoder(timestep_embedding(timestep, self.time_dim))
        if time.ndim == 3:
            global_condition = global_condition[:, None].expand(
                -1, time.shape[1], -1
            )
        condition = torch.cat([time, global_condition], dim=-1)
        x = sample.transpose(1, 2)
        skips = []
        for first, second, downsample in self.down:
            x = first(x, condition)
            x = second(x, condition)
            skips.append(x)
            x = downsample(x)
        for block in self.middle:
            x = block(x, condition)
        for first, second, upsample in self.up:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                raise ValueError(
                    f"temporal length {sample.shape[1]} is not compatible with U-Net"
                )
            x = torch.cat([x, skip], dim=1)
            x = first(x, condition)
            x = second(x, condition)
            x = upsample(x)
        return self.final(x).transpose(1, 2)


class BSPUNetPolicyBase(PolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg, token_observation=False)
        self.observation = BSPObservationEncoder(
            cameras=len(cfg.data.camera_keys),
            observation_horizon=cfg.data.observation_horizon,
            state_dim=cfg.data.state_dim,
            image_size=cfg.vision.image_size,
            crop_size=cfg.vision.crop_size,
            keypoints=cfg.vision.spatial_keypoints,
            encoder_weights=cfg.vision.bsp_encoder_weights,
            encoder_norm=cfg.vision.bsp_encoder_norm,
            rgb_normalization=cfg.vision.bsp_rgb_normalization,
        )
        levels = len(cfg.policy.unet_down_dims) - 1
        multiple = 2**levels
        self.padded_steps = int(math.ceil(self.num_basis / multiple) * multiple)

    def observations(self, batch: dict[str, torch.Tensor]):
        return self.observation(batch["images"], batch["state"])

    def _pad_rows(self, value: torch.Tensor, fill: float | int = 0) -> torch.Tensor:
        count = self.padded_steps - value.shape[1]
        if count <= 0:
            return value
        shape = (value.shape[0], count, *value.shape[2:])
        return torch.cat([value, value.new_full(shape, fill)], dim=1)


class BSPUNetFlowMatchingPolicy(BSPUNetPolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.unet = BSPConditionalUNet1D(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            global_condition_dim=self.observation.global_condition_dim,
            time_dim=cfg.policy.unet_time_dim,
            down_dims=tuple(cfg.policy.unet_down_dims),
            kernel_size=cfg.policy.unet_kernel_size,
            groups=cfg.policy.unet_groups,
        )

    def velocity(self, x: torch.Tensor, batch, time: torch.Tensor) -> torch.Tensor:
        condition = self.observations(batch).global_condition
        padded = self._pad_rows(x)
        if time.ndim == 2 and time.shape[1] != padded.shape[1]:
            padding = padded.shape[1] - time.shape[1]
            if padding < 0:
                raise ValueError("time map is longer than the padded action sequence")
            time = torch.cat([time, time[:, -1:].expand(-1, padding)], dim=1)
        return self.unet(padded, time * 1000.0, condition)[:, : self.num_basis]

    def loss(self, batch, rtc=None):
        target = batch["continuous_target"].float()
        beta = torch.distributions.Beta(target.new_tensor(1.5), target.new_tensor(1.0))
        sampled = beta.sample((len(target),))
        time = ((0.999 - sampled) / 0.999).clamp(0, 1)
        noise = torch.randn_like(target)
        time_map = time[:, None].expand(-1, target.shape[1])
        fixed = None
        if rtc is not None:
            fixed = rtc["fixed_mask"].bool()
            fixed_rows = fixed.any(dim=-1)
            time_map = torch.where(fixed_rows, torch.ones_like(time_map), time_map)
        x = (1 - time_map[..., None]) * noise + time_map[..., None] * target
        velocity_target = target - noise
        mutable = batch["control_valid_mask"].bool()
        if rtc is not None:
            x = torch.where(fixed, rtc["prefix_values"].float(), x)
            mutable &= ~fixed
        prediction = self.velocity(x, batch, time_map)
        mse = ((prediction - velocity_target) ** 2)[mutable].mean()
        endpoint = x + (1 - time_map[..., None]) * prediction
        action_mse = self.decoded_action_mse(endpoint, batch, mutable)
        return {
            "loss": mse,
            "fm_mse": mse,
            "action_mse": action_mse.detach(),
            "gradient_scale_fm": mse.detach(),
        }

    @torch.no_grad()
    def sample(self, batch, steps=None, prefix_values=None, fixed_mask=None, **kwargs):
        steps = int(steps or self.cfg.policy.fm_steps)
        device = batch["images"].device
        x = torch.randn(len(batch["images"]), self.num_basis, self.action_dim, device=device)
        delta = 1.0 / steps
        for index in range(steps):
            if fixed_mask is not None:
                x = torch.where(fixed_mask, prefix_values, x)
            time = x.new_full((len(x),), index / steps)
            x = x + delta * self.velocity(x, batch, time)
            if fixed_mask is not None:
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
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor | float | int]]]:
        """Base-FM RTC using reference PiGDM with a binary hard mask."""
        steps = int(steps or self.cfg.policy.fm_steps)
        device = batch["images"].device
        with torch.no_grad():
            condition = self.observations(batch).global_condition
            noise = torch.randn(
                len(batch["images"]), self.num_basis, self.action_dim, device=device
            )

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            padded = self._pad_rows(state)
            return self.unet(
                padded,
                time * 1000.0,
                condition,
            )[:, : self.num_basis]

        return hard_mask_pigdm_sample(
            noise,
            velocity,
            prefix_values.float(),
            fixed_mask.bool(),
            steps=steps,
            max_guidance_weight=max_guidance_weight,
            return_trace=return_trace,
        )


class BSPUNetDiscretePolicy(BSPUNetPolicyBase):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.mask_id = 256
        embedding_dim = cfg.policy.discrete_embed_dim
        self.token_embedding = nn.Embedding(257, embedding_dim)
        self.dimension_embedding = nn.Parameter(
            torch.randn(self.action_dim, embedding_dim) * 0.02
        )
        self.unet = BSPConditionalUNet1D(
            input_dim=self.action_dim * embedding_dim,
            output_dim=self.action_dim * 256,
            global_condition_dim=self.observation.global_condition_dim,
            time_dim=cfg.policy.unet_time_dim,
            down_dims=tuple(cfg.policy.unet_down_dims),
            kernel_size=cfg.policy.unet_kernel_size,
            groups=cfg.policy.unet_groups,
        )

    def logits(self, tokens: torch.Tensor, batch, noise_level: torch.Tensor) -> torch.Tensor:
        condition = self.observations(batch).global_condition
        embedded = self.token_embedding(tokens) + self.dimension_embedding[None, None]
        embedded = embedded.flatten(2)
        embedded = self._pad_rows(embedded)
        logits = self.unet(embedded, noise_level * 1000.0, condition)
        return logits[:, : self.num_basis].reshape(
            len(tokens), self.num_basis, self.action_dim, 256
        )

    def loss(self, batch, rtc=None):
        target = batch["discrete_target"].long()
        valid = batch["control_valid_mask"].bool()
        flat_target = target.flatten(1)
        flat_valid = valid.flatten(1)
        corrupted, supervised = mixed_block_corruption(
            flat_target,
            flat_valid,
            self.cfg.policy.block_size,
            self.cfg.policy.discrete_full_mask_probability,
        )
        mutable = flat_valid.clone()
        if rtc is not None:
            fixed = rtc["fixed_mask"].bool().flatten(1)
            fixed_tokens = rtc["prefix_values"].long().flatten(1)
            corrupted = torch.where(fixed, fixed_tokens, corrupted)
            supervised &= ~fixed
            mutable &= ~fixed
        initial_mask = (corrupted == self.mask_id) & mutable
        noise_level = initial_mask.sum(1).float() / mutable.sum(1).clamp_min(1)
        logits = self.logits(
            corrupted.reshape_as(target), batch, noise_level
        )
        flat_logits = logits.flatten(1, 2)
        ce, l1 = expected_token_distance(flat_logits, flat_target, supervised)
        total = ce + self.cfg.train.lambda_l1 * l1
        return {
            "loss": total,
            "loss_ce": ce,
            "loss_l1": l1,
            "action_mse": self.logits_action_mse(
                flat_logits, batch, supervised
            ).detach(),
            "gradient_scale_ce": ce.detach(),
            "gradient_scale_l1": l1.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        batch,
        rounds=None,
        prefix_values=None,
        fixed_mask=None,
        return_trace=False,
        **kwargs,
    ):
        rounds = int(rounds or self.cfg.policy.discrete_rounds)
        batch_size = len(batch["images"])
        device = batch["images"].device
        tokens = torch.full(
            (batch_size, self.num_basis, self.action_dim),
            self.mask_id,
            dtype=torch.long,
            device=device,
        )
        immutable = torch.zeros_like(tokens, dtype=torch.bool)
        if fixed_mask is not None:
            immutable = fixed_mask.bool()
            prefix_values = prefix_values.long()
            tokens = torch.where(immutable, prefix_values, tokens)
        mutable = (~immutable).flatten(1)
        trace = []
        for step in range(rounds):
            unknown = ((tokens == self.mask_id) & ~immutable).flatten(1)
            noise_level = unknown.sum(1).float() / mutable.sum(1).clamp_min(1)
            logits = self.logits(tokens, batch, noise_level)
            updated = maskgit_update(
                logits.flatten(1, 2), tokens.flatten(1), mutable, step, rounds
            ).reshape_as(tokens)
            tokens = (
                torch.where(immutable, prefix_values, updated)
                if prefix_values is not None
                else updated
            )
            if return_trace:
                trace.append({"step": step, "tokens": tokens.clone(), "logits": logits.clone()})
        return (tokens, trace) if return_trace else tokens
