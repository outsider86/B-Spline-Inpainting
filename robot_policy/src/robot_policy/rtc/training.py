from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_policy.config import is_continuous_architecture

from .delay_mapping import control_support_mask, raw_action_prefix_mask


class TorchSplineCodec:
    """Torch form of the pinned encoder's fixed basis/solver and calibration."""
    def __init__(self, prepared_path: str | Path, device: torch.device):
        self.representation = "bspline"
        root = Path(prepared_path)
        record = json.loads((root / "encoder.json").read_text())
        # Reconstruct matrices through the authoritative adapter rather than
        # maintaining an independent spline formula.
        from robot_policy.config import Config
        from robot_policy.encoders.bspline_adapter import BSplineAdapter
        cfg = Config(); calibration = record["calibration"]
        adapter = BSplineAdapter(cfg, calibration)
        self.basis = torch.as_tensor(adapter.basis, dtype=torch.float32, device=device)
        self.solver = torch.as_tensor(adapter.encoder._solver, dtype=torch.float32, device=device)
        self.num_basis = int(record["config"]["num_basis"])
        self.degree = int(record["config"]["degree"])
        self.span_length_steps = int(record["config"]["span_length_steps"])
        self.low = torch.as_tensor(calibration["low"], dtype=torch.float32, device=device)
        self.high = torch.as_tensor(calibration["high"], dtype=torch.float32, device=device)

    def decode_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tn,bnd->btd", self.basis, controls.float())

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.low + tokens.float() / 255 * (self.high - self.low)

    def encode_tokens(self, controls: torch.Tensor) -> torch.Tensor:
        span = (self.high - self.low).clamp_min(1e-8)
        return (((controls - self.low) / span).clamp(0, 1) * 255).round().long()

    def shift_and_refit(self, controls: torch.Tensor, raw_delays: torch.Tensor) -> torch.Tensor:
        actions = self.decode_controls(controls)
        shifted = []
        for item, delay in zip(actions, raw_delays.tolist()):
            tail = item[delay:]
            if len(tail) == 0:
                raise ValueError("delay exceeds prediction coverage")
            shifted.append(torch.cat([tail, tail[-1:].expand(delay, -1)], dim=0))
        shifted = torch.stack(shifted)
        padded = torch.cat([shifted, shifted[:, -1:]], dim=1)
        return torch.einsum("nt,btd->bnd", self.solver, padded)


class RawActionCodec:
    """Identity sequence codec with uniform scalar quantization in normalized action space."""

    def __init__(self, prepared_path: str | Path, device: torch.device):
        self.representation = "raw"
        record = json.loads((Path(prepared_path) / "encoder.json").read_text())
        self.action_horizon = int(record["config"]["action_horizon"])
        self.low = torch.as_tensor(record["calibration"]["low"], dtype=torch.float32, device=device)
        self.high = torch.as_tensor(record["calibration"]["high"], dtype=torch.float32, device=device)

    def decode_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return controls.float()

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.low + tokens.float() / 255 * (self.high - self.low)

    def encode_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        span = (self.high - self.low).clamp_min(1e-8)
        return (((actions - self.low) / span).clamp(0, 1) * 255).round().long()

    def shift_and_refit(self, actions: torch.Tensor, raw_delays: torch.Tensor) -> torch.Tensor:
        shifted = []
        for item, delay in zip(actions.float(), raw_delays.tolist()):
            tail = item[delay:]
            if len(tail) == 0:
                raise ValueError("delay exceeds prediction coverage")
            shifted.append(torch.cat([tail, tail[-1:].expand(delay, -1)], dim=0))
        return torch.stack(shifted)


def create_action_codec(cfg: Any, device: torch.device) -> TorchSplineCodec | RawActionCodec:
    if cfg.data.action_representation == "raw":
        return RawActionCodec(cfg.data.prepared_path, device)
    return TorchSplineCodec(cfg.data.prepared_path, device)


def make_reference_ttrtc_condition(
    batch: dict[str, torch.Tensor],
    codec: TorchSplineCodec | RawActionCodec,
    raw_delays: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the reference ttRTC teacher hard mask from the current target.

    Kinetix trains simulated delay with ground-truth action rows at flow time
    one.  Raw actions use the first ``d`` rows.  B-splines use the exact union
    of cubic control rows supporting the ``ceil(d / span)`` affected spans.
    """
    if raw_delays.dtype == torch.bool or raw_delays.is_floating_point():
        raise TypeError("ttRTC delays must be integer raw-action counts")
    if codec.representation == "bspline":
        affected_spans = torch.div(
            raw_delays + codec.span_length_steps - 1,
            codec.span_length_steps,
            rounding_mode="floor",
        )
        fixed = control_support_mask(
            affected_spans,
            num_basis=codec.num_basis,
            action_dim=batch["continuous_target"].shape[-1],
            degree=codec.degree,
        )
    else:
        affected_spans = torch.zeros_like(raw_delays)
        fixed = raw_action_prefix_mask(
            raw_delays,
            codec.action_horizon,
            batch["continuous_target"].shape[-1],
        )
    return {
        "fixed_mask": fixed,
        "prefix_values": batch["continuous_target"].float(),
        "delay_spans": affected_spans,
        "delay_raw_actions": raw_delays,
    }


@torch.no_grad()
def make_rtc_condition(parent, previous_batch: dict[str, torch.Tensor], architecture: str,
                       codec: TorchSplineCodec | RawActionCodec, delays: torch.Tensor, has_previous: torch.Tensor,
                       predicted: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    predicted = parent.sample(previous_batch) if predicted is None else predicted
    controls = predicted.float() if is_continuous_architecture(architecture) else codec.decode_tokens(predicted)
    raw_delays = delays * codec.span_length_steps if codec.representation == "bspline" else delays
    shifted = codec.shift_and_refit(controls, raw_delays)
    fixed = (
        control_support_mask(
            delays,
            num_basis=codec.num_basis,
            action_dim=shifted.shape[-1],
            degree=codec.degree,
        )
        if codec.representation == "bspline"
        else raw_action_prefix_mask(delays, codec.action_horizon)
    ).clone()
    fixed &= has_previous[:, None, None]
    values = shifted if is_continuous_architecture(architecture) else codec.encode_tokens(shifted)
    return {"fixed_mask": fixed, "prefix_values": values, "delay_spans": delays if codec.representation == "bspline" else torch.zeros_like(delays), "delay_raw_actions": raw_delays}
