from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


def _import_encoder() -> Any:
    try:
        import spline_encoder
        return spline_encoder
    except ImportError:
        # Development checkout fallback; core models remain free of absolute paths.
        checkout = Path(__file__).resolve().parents[4] / "BSplineEncoder" / "src"
        if checkout.exists():
            sys.path.insert(0, str(checkout))
            import spline_encoder
            return spline_encoder
        raise ImportError("Install the pinned BSplineEncoder with `pip install -e ../BSplineEncoder`")


@dataclass(frozen=True)
class EncodedAction:
    continuous: np.ndarray
    discrete: np.ndarray | None
    reconstruction: np.ndarray
    quantized_reconstruction: np.ndarray | None
    raw_valid_mask: np.ndarray
    control_valid_mask: np.ndarray
    boundary_metadata: dict[str, Any]


class BSplineAdapter:
    """Thin adapter around the authoritative BSplineEncoder implementation."""

    def __init__(self, cfg: Any, calibration: dict[str, Any] | None = None):
        se = _import_encoder()
        self.config = se.UniformLeftBSplineConfig(
            action_dim=7,
            chunk_size=cfg.data.action_horizon,
            frequency_hz=cfg.data.frequency_hz,
            degree=cfg.spline.degree,
            num_basis=cfg.spline.num_basis,
            span_length_steps=cfg.spline.span_length_steps,
            vocab_size=cfg.spline.vocab_size,
            regularization=cfg.spline.regularization,
            end_padding=True,
            implementation_version=cfg.spline.implementation_version,
        )
        self.encoder = se.create_encoder(self.config, calibration=calibration)

    @property
    def basis(self) -> np.ndarray:
        return np.asarray(self.encoder.basis)

    @property
    def geometry(self) -> Any:
        return self.encoder.geometry()

    @property
    def tokenizer_id(self) -> str:
        return self.encoder.tokenizer_id

    def fit(self, window: np.ndarray) -> np.ndarray:
        return self.encoder.fit_control_points(window)

    def calibrate_controls(self, controls: Iterable[np.ndarray]) -> None:
        low = high = None
        count = 0
        for item in controls:
            item = np.asarray(item, dtype=np.float64)
            low = item.copy() if low is None else np.minimum(low, item)
            high = item.copy() if high is None else np.maximum(high, item)
            count += 1
        if count == 0:
            raise ValueError("cannot calibrate with no training controls")
        self.encoder.quantizer.set_bounds(low, high)

    def calibration_state(self) -> dict[str, list]:
        return self.encoder.calibration_state()

    def encode_window(self, window: np.ndarray, valid_steps: int) -> EncodedAction:
        if not self.encoder.calibrated:
            raise RuntimeError("calibrate the adapter before discrete encoding")
        result = self.encoder.encode_chunk(window, quantize=True)
        raw_valid = np.arange(self.config.chunk_size) < valid_steps
        # Every control participates in the padded, executable spline target.  This
        # mask records which controls have Greville locations inside recorded time;
        # callers choose padded-target or recorded-only supervision explicitly.
        control_valid = self.geometry.greville_steps < valid_steps
        control_valid = np.broadcast_to(control_valid[:, None], result.control_points.shape).copy()
        return EncodedAction(
            continuous=result.control_points.astype(np.float32),
            discrete=result.tokens.astype(np.int64),
            reconstruction=result.decode().astype(np.float32),
            quantized_reconstruction=result.decode(dequantized=True).astype(np.float32),
            raw_valid_mask=raw_valid,
            control_valid_mask=control_valid,
            boundary_metadata={
                "phase_steps": 0,
                "degree": result.degree,
                "sample_period": result.sample_period,
                "executable_steps": result.executable_steps,
                "span_length_steps": self.config.span_length_steps,
                "fixed_knot_steps": result.knot_steps.tolist(),
                "duration_steps": result.duration_steps.tolist(),
                "boundary_condition": result.metadata["boundary_condition"],
                "implementation_version": result.metadata["implementation_version"],
            },
        )

    def decode_controls(self, controls: np.ndarray) -> np.ndarray:
        return (self.basis @ np.asarray(controls, dtype=np.float64)).astype(np.float32)

    def decode_tokens(self, tokens: np.ndarray) -> np.ndarray:
        return self.encoder.decode(tokens).astype(np.float32)

