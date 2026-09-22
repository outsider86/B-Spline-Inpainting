from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from robot_policy.encoders.observation import ObservationTokenizer, ObservationTokens


class PolicyBase(nn.Module):
    def __init__(self, cfg: Any, *, token_observation: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_basis = cfg.spline.num_basis if cfg.data.action_representation == "bspline" else cfg.data.action_horizon
        self.action_steps = self.num_basis
        self.action_dim = 7
        self.action_positions = self.num_basis * self.action_dim
        self.observation = (
            ObservationTokenizer(
                vision_dim=cfg.vision.feature_dim,
                state_dim=cfg.data.state_dim,
                hidden_dim=cfg.policy.hidden_dim,
                cameras=len(cfg.data.camera_keys),
                tokens_per_camera=cfg.vision.pooled_grid ** 2,
                observation_horizon=cfg.data.observation_horizon,
                resampler_tokens_per_camera=cfg.vision.resampler_tokens_per_camera,
                resampler_heads=cfg.policy.heads,
            )
            if token_observation
            else None
        )
        if cfg.data.action_representation == "bspline":
            from robot_policy.encoders.bspline_adapter import BSplineAdapter

            record_path = Path(cfg.data.prepared_path) / "encoder.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            calibration = record.get("calibration")
            adapter = BSplineAdapter(cfg, calibration)
            basis = torch.as_tensor(adapter.basis, dtype=torch.float32)
            if calibration is None:
                low = torch.full((self.num_basis, self.action_dim), -1.0)
                high = torch.full((self.num_basis, self.action_dim), 1.0)
            else:
                low = torch.as_tensor(calibration["low"], dtype=torch.float32)
                high = torch.as_tensor(calibration["high"], dtype=torch.float32)
        else:
            basis = torch.eye(cfg.data.action_horizon, dtype=torch.float32)
            low = torch.full((self.num_basis, self.action_dim), -1.0)
            high = torch.full((self.num_basis, self.action_dim), 1.0)
        # These are fixed representation metadata, not learned checkpoint state.
        self.register_buffer("action_decode_basis", basis, persistent=False)
        self.register_buffer("action_token_low", low, persistent=False)
        self.register_buffer("action_token_high", high, persistent=False)

    def observations(self, batch: dict[str, torch.Tensor]) -> ObservationTokens:
        if self.observation is None:
            raise RuntimeError("this policy does not use the token observation encoder")
        return self.observation(batch["vision_features"], batch["state"])

    @torch.no_grad()
    def decoded_action_mse(self, controls: torch.Tensor, batch: dict[str, torch.Tensor],
                           control_mask: torch.Tensor | None = None) -> torch.Tensor:
        target_controls = batch["continuous_target"].float()
        if control_mask is not None:
            control_mask = control_mask.bool().reshape_as(target_controls)
            # Match the raw-action diagnostic: only corruption-supervised values
            # contribute. Ground-truth-fill all other controls before decoding.
            controls = torch.where(control_mask, controls.float(), target_controls)
        actions = torch.einsum("tn,bnd->btd", self.action_decode_basis, controls.float())
        target = batch.get("normalized_target_trajectory")
        if target is None:
            target = torch.einsum("tn,bnd->btd", self.action_decode_basis, target_controls)
        valid = batch.get("action_valid_mask")
        if valid is None:
            valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        action_mask = valid.bool().unsqueeze(-1).expand_as(actions)
        if control_mask is not None:
            support = self.action_decode_basis.abs() > 1e-12
            affected = torch.einsum("tn,bnd->btd", support.float(), control_mask.float()) > 0
            action_mask = action_mask & affected
        return ((actions - target.float()) ** 2)[action_mask].mean()

    @torch.no_grad()
    def logits_action_mse(self, logits: torch.Tensor, batch: dict[str, torch.Tensor],
                          supervised: torch.Tensor) -> torch.Tensor:
        bins = torch.arange(256, device=logits.device, dtype=torch.float32)
        expected_token = (logits.float().softmax(-1) * bins).sum(-1).reshape(-1, self.num_basis, self.action_dim)
        controls = self.action_token_low + expected_token / 255 * (self.action_token_high - self.action_token_low)
        return self.decoded_action_mse(controls, batch, supervised)

    def loss(self, batch: dict[str, torch.Tensor], rtc: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor], rtc: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        return self.loss(batch, rtc)

    @torch.no_grad()
    def sample(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        raise NotImplementedError
