from __future__ import annotations

from collections.abc import Callable

import torch


def hard_mask_pigdm_sample(
    noise: torch.Tensor,
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    steps: int,
    max_guidance_weight: float = 5.0,
) -> torch.Tensor:
    """Reference PiGDM Euler sampler with a binary conditioning operator.

    This is the `prefix_attention_schedule="zeros"` case from the Kinetix
    reference: conditioned coordinates have weight one and every other
    coordinate has weight zero.  The time-dependent scalar is the PiGDM
    correction coefficient, not a soft prefix mask.
    """
    if steps < 1:
        raise ValueError("PiGDM steps must be positive")
    if max_guidance_weight <= 0:
        raise ValueError("PiGDM max_guidance_weight must be positive")
    if noise.shape != condition_values.shape or noise.shape != condition_mask.shape:
        raise ValueError("PiGDM noise, condition values, and mask must have identical shapes")
    if condition_mask.dtype != torch.bool:
        raise TypeError("PiGDM condition_mask must be boolean")

    x = noise
    delta = 1.0 / steps
    for index in range(steps):
        time_value = index / steps
        time = x.new_full((len(x),), time_value)
        with torch.enable_grad():
            state = x.detach().requires_grad_(True)
            predicted_velocity = velocity(state, time)
            endpoint = state + (1.0 - time_value) * predicted_velocity
            endpoint_error = torch.where(
                condition_mask,
                condition_values - endpoint,
                torch.zeros_like(endpoint),
            )
            correction = torch.autograd.grad(
                endpoint,
                state,
                grad_outputs=endpoint_error,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        if time_value == 0.0:
            guidance_weight = max_guidance_weight
        else:
            remaining = 1.0 - time_value
            inverse_r2 = (time_value**2 + remaining**2) / (remaining**2)
            coefficient = remaining / time_value
            guidance_weight = min(
                coefficient * inverse_r2,
                max_guidance_weight,
            )
        x = (
            state
            + delta
            * (
                predicted_velocity.detach()
                + guidance_weight * correction.detach()
            )
        ).detach()
    return x
