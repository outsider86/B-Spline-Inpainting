from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

from robot_policy.config import is_continuous_architecture
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


class PolicyRunner:
    """Checkpoint-owned sampling and spline decode with explicit RTC state."""
    def __init__(self, cfg: Any, checkpoint: str | Path, device: str = "cuda"):
        self.cfg = cfg; self.device = torch.device(device)
        self.model, self.payload = load_policy_checkpoint(checkpoint, cfg, self.device)
        self.codec = create_action_codec(cfg, self.device)
        stats = json.loads((Path(cfg.data.prepared_path) / "normalization.json").read_text())
        self.action_low = torch.tensor(stats["action_q01"], device=self.device)
        self.action_high = torch.tensor(stats["action_q99"], device=self.device)
        self.previous_controls: torch.Tensor | None = None
        self.episode_id: int | None = None

    def reset(self, episode_id: int | None = None) -> None:
        self.previous_controls = None; self.episode_id = episode_id

    def plan(self, batch: dict[str, torch.Tensor], *, episode_id: int, delay_spans: int = 0,
             delay_raw_actions: int | None = None,
             fm_steps: int | None = None, discrete_rounds: int | None = None,
             use_cache: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.episode_id != episode_id: self.reset(episode_id)
        batch = {k: v.to(self.device) for k,v in batch.items()}
        fixed = prefix = None
        raw_delay = delay_spans * self.cfg.spline.span_length_steps if delay_raw_actions is None else delay_raw_actions
        mapping = map_delay(raw_delay, frequency_hz=self.cfg.data.frequency_hz,
                            span_length_steps=self.cfg.spline.span_length_steps, degree=self.cfg.spline.degree)
        if raw_delay < 0 or raw_delay > self.cfg.rtc.raw_delay_max:
            raise ValueError(f"raw inference delay must be in 0..{self.cfg.rtc.raw_delay_max}; it is never silently clamped")
        if raw_delay and self.previous_controls is not None:
            raw = torch.full((len(batch["state"]),), raw_delay, device=self.device, dtype=torch.long)
            shifted = self.codec.shift_and_refit(self.previous_controls, raw)
            if self.cfg.data.action_representation == "bspline":
                affected = torch.full_like(raw, mapping.affected_spans)
                fixed = control_support_mask(
                    affected,
                    self.cfg.spline.num_basis,
                    shifted.shape[-1],
                    self.cfg.spline.degree,
                )
            else:
                fixed = raw_action_prefix_mask(raw, self.cfg.data.action_horizon)
            prefix = shifted if is_continuous_architecture(self.cfg.policy.architecture) else self.codec.encode_tokens(shifted)
        started = time.perf_counter()
        if (
            prefix is not None
            and is_continuous_architecture(self.cfg.policy.architecture)
            and str(self.payload.get("training_type", "base")).lower() == "base"
        ):
            predicted = self.model.sample_realtime_pigdm(
                batch,
                steps=fm_steps,
                prefix_values=prefix,
                fixed_mask=fixed,
            )
            rtc_method = "pigdm_hard_mask"
        else:
            with torch.no_grad():
                predicted = self.model.sample(
                    batch,
                    steps=fm_steps,
                    rounds=discrete_rounds,
                    prefix_values=prefix,
                    fixed_mask=fixed,
                    use_cache=use_cache,
                )
            rtc_method = (
                "training_time_hard_mask" if prefix is not None else "from_scratch"
            )
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        sampling_ms = (time.perf_counter()-started)*1000
        controls = predicted.float() if is_continuous_architecture(self.cfg.policy.architecture) else self.codec.decode_tokens(predicted)
        self.previous_controls = controls.detach()
        normalized = self.codec.decode_controls(controls)
        physical = (normalized + 1) * 0.5 * (self.action_high-self.action_low) + self.action_low
        metadata = {"sampling_ms":sampling_ms,"delay_spans":mapping.whole_spans,"delay_phase_steps":mapping.phase_steps,
                    "affected_spans":mapping.affected_spans,"delay_raw_actions":raw_delay,"delay_ms":mapping.milliseconds,
                    "fixed_controls":0 if fixed is None else int(fixed[0,:,0].sum()),"cache_enabled":bool(use_cache and self.cfg.policy.architecture=="discrete_joint"),
                    "action_representation":self.cfg.data.action_representation}
        metadata["rtc_inference_method"] = rtc_method
        return physical, metadata
