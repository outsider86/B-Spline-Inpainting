from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DelayMapping:
    raw_actions: int
    milliseconds: float
    whole_spans: int
    phase_steps: int
    affected_spans: int
    support_control_indices: tuple[int, ...]
    committed_control_count: int
    coverage_exceeded: bool


def map_delay(raw_actions: int, *, frequency_hz: float = 30.0, span_length_steps: int = 2,
              degree: int = 3, executable_spans: int = 15) -> DelayMapping:
    if raw_actions < 0:
        raise ValueError("delay cannot be negative")
    whole, phase = divmod(raw_actions, span_length_steps)
    affected = math.ceil(raw_actions / span_length_steps) if raw_actions else 0
    exceeded = affected > executable_spans
    if exceeded:
        support = ()
    elif affected:
        # Union of p+1 local controls for all affected open spans: 0..D+p-1.
        support = tuple(range(affected + degree))
    else:
        support = ()
    return DelayMapping(
        raw_actions=raw_actions, milliseconds=1000 * raw_actions / frequency_hz,
        whole_spans=whole, phase_steps=phase, affected_spans=affected,
        support_control_indices=support, committed_control_count=len(support),
        coverage_exceeded=exceeded,
    )


def sample_spline_delays(batch_size: int, device: torch.device, minimum: int = 1, maximum: int = 5,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Kinetix reference shape adapted from q=0..4 to required D=1..5."""
    values = torch.arange(minimum, maximum + 1, device=device)
    probabilities = torch.exp((maximum - values).float())
    probabilities /= probabilities.sum()
    indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    return values[indices]


def sample_raw_delays(batch_size: int, device: torch.device, minimum: int = 1, maximum: int = 10,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """The same truncated decreasing-exponential law, in raw-action units."""
    return sample_spline_delays(batch_size, device, minimum, maximum, generator)


def sample_ttrtc_raw_delays(
    batch_size: int,
    device: torch.device,
    maximum: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reference ttRTC delay law, including the unconditional delay-zero case."""
    if maximum < 1:
        raise ValueError("ttRTC maximum delay must be positive")
    values = torch.arange(0, maximum + 1, device=device)
    probabilities = torch.exp((maximum - values).float())
    probabilities /= probabilities.sum()
    indices = torch.multinomial(
        probabilities, batch_size, replacement=True, generator=generator
    )
    return values[indices]


def control_support_mask(
    delays: torch.Tensor,
    num_basis: int = 18,
    action_dim: int = 7,
    degree: int = 3,
) -> torch.Tensor:
    """Hard-mask exactly the control rows supporting the committed spans.

    For ``D`` consecutive left-boundary spans of a degree-``p`` B-spline, the
    union of their local supports is control rows ``[0, D + p)``.  No control
    row outside that union is conditioned.  The returned mask is shared by
    continuous flow-matching RTC and discrete joint-diffusion RTC.
    """
    if delays.dtype == torch.bool or delays.is_floating_point():
        raise TypeError("B-spline span delays must be integer tensors")
    # No committed span means no conditioned control row.  For D>0 spans the
    # exact union of cubic local support is rows [0, D+3).
    counts = torch.where(delays > 0, delays + degree, torch.zeros_like(delays))
    indices = torch.arange(num_basis, device=delays.device)[None, :, None]
    return (indices < counts[:, None, None]).expand(-1, -1, action_dim)


def raw_action_prefix_mask(delays: torch.Tensor, action_horizon: int = 30, action_dim: int = 7) -> torch.Tensor:
    indices = torch.arange(action_horizon, device=delays.device)[None, :, None]
    return (indices < delays[:, None, None]).expand(-1, -1, action_dim)
