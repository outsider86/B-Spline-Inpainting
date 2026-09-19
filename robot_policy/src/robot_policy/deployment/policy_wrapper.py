from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from robot_policy.config import is_continuous_architecture, uses_bsp_image_encoder
from robot_policy.deployment.checkpoint import (
    CheckpointMetadata,
    inspect_checkpoint,
    load_deployment_policy,
)
from robot_policy.rtc.delay_mapping import (
    control_support_mask,
    map_delay,
    raw_action_prefix_mask,
)
from robot_policy.rtc.training import create_action_codec


def _as_numpy_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"each image must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("images must contain finite numeric RGB values")
        if image.min() < 0 or image.max() > 255:
            raise ValueError("non-uint8 image values must lie in [0,255]")
        image = np.rint(image).astype(np.uint8)
    return np.ascontiguousarray(image)


class PolicyServerWrapper:
    """Deploy any robot-policy checkpoint behind the established Piper API.

    The wire-level call remains ``predict_action(examples=[...])`` and returns
    physical robot-coordinate ``actions``. Images are supplied in configured
    camera order. State defaults to the reference client's min/max-normalized
    ``[-1,1]`` coordinates and is converted to this model's z-score domain.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        prepared_path: str | Path | None = None,
        precision: str = "bf16",
        task_instruction: str = "Stack the cups.",
        binary_gripper: bool = True,
        gripper_threshold: float = 0.3,
        vision_encoder: Any | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")
        if precision == "bf16" and self.device.type != "cuda":
            precision = "fp32"
        if not 0.0 < gripper_threshold < 1.0:
            raise ValueError("gripper_threshold must lie strictly between 0 and 1")

        self.checkpoint: CheckpointMetadata = inspect_checkpoint(
            checkpoint, prepared_path=prepared_path
        )
        self.cfg = self.checkpoint.config
        self.precision = precision
        self.task_instruction = task_instruction
        self.binary_gripper = bool(binary_gripper)
        self.gripper_threshold = float(gripper_threshold)
        self.model = (
            load_deployment_policy(self.checkpoint, self.device)
            if model is None
            else model.to(self.device).eval()
        )
        self.codec = create_action_codec(self.cfg, self.device)

        stats = json.loads(
            (self.checkpoint.prepared_path / "normalization.json").read_text()
        )
        self.state_mean = torch.tensor(stats["state_mean"], device=self.device)
        self.state_std = torch.tensor(stats["state_std"], device=self.device)
        self.action_low = torch.tensor(stats["action_q01"], device=self.device)
        self.action_high = torch.tensor(stats["action_q99"], device=self.device)
        self.state_min, self.state_max = self._state_bounds(stats)

        if vision_encoder is None and not uses_bsp_image_encoder(
            self.cfg.policy.architecture
        ):
            from robot_policy.encoders.vision import FrozenDinoSigLIP

            vision_encoder = FrozenDinoSigLIP(self.cfg).to(self.device).eval()
        self.vision_encoder = vision_encoder

    @property
    def metadata(self) -> dict[str, Any]:
        representation = self.cfg.data.action_representation
        is_rtc = self.checkpoint.is_rtc
        return {
            "env": "robot_policy_server",
            "protocol_version": 1,
            "checkpoint": str(self.checkpoint.path),
            "checkpoint_update": self.checkpoint.update,
            "checkpoint_training_type": self.checkpoint.training_type,
            "architecture": self.checkpoint.architecture,
            "model_size": self.cfg.policy.model_size,
            "action_chunk_size": self.cfg.data.action_horizon,
            "action_dimension": 7,
            "action_coordinates": "physical_absolute_robot_coordinates",
            "action_normalization": "training_split_q01_q99",
            "action_representation": representation,
            "action_parameter_shape": [
                self.cfg.spline.num_basis
                if representation == "bspline"
                else self.cfg.data.action_horizon,
                7,
            ],
            "camera_keys": list(self.cfg.data.camera_keys),
            "camera_order": ["global", "hand"],
            "camera_count": len(self.cfg.data.camera_keys),
            "training_obs_image_size": [self.cfg.vision.image_size] * 2,
            "image_preprocessing": (
                "RGB resized to 84x84; policy applies [-1,1] normalization, "
                "train-time random 76x76 crop, and scratch ResNet18+SpatialSoftmax"
                if uses_bsp_image_encoder(self.cfg.policy.architecture)
                else "RGB; server bicubic-resizes to training size and applies "
                "checkpoint-specific DINOv2/SigLIP normalization"
            ),
            "observation_source": self.cfg.data.observation_source,
            "observation_horizon": self.cfg.data.observation_horizon,
            "observation_history_fields": (
                ["image_history", "state_history"]
                if self.cfg.data.observation_horizon == 2
                else []
            ),
            "observation_history_bootstrap": (
                "repeat current image/state when history fields are omitted"
                if self.cfg.data.observation_horizon == 2
                else None
            ),
            "state_shape": [7],
            "state_coordinates_default": "legacy_minmax_minus1_plus1",
            "state_coordinates_supported": ["normalized", "zscore", "physical"],
            "state_model_normalization": "(physical_state - training_mean) / training_std",
            "state_client_normalization": "2 * (physical_state - training_min) / (training_max - training_min) - 1",
            "task_instruction": self.task_instruction,
            "language_conditioning": "fixed_task_metadata_only; this compact policy has no language tokens",
            "frequency_hz": self.cfg.data.frequency_hz,
            "supports_inference_time_rtc": is_rtc and representation == "raw",
            "supports_parameter_row_rtc": is_rtc and representation == "bspline",
            "rtc_conditioning_space": (
                "decoded_actions" if representation == "raw" else "spline_parameter_rows"
            ),
            "rtc_delay_units": "raw_action_steps",
            "rtc_max_delay_steps": self.cfg.rtc.raw_delay_max,
            "rtc_mode": "training_time_hard_prefix" if is_rtc else None,
            "rtc_mask_type": "hard" if is_rtc else None,
            "rtc_bspline_mask_scope": (
                "exact_union_of_control_rows_supporting_affected_spans"
                if is_rtc and representation == "bspline"
                else None
            ),
            "rtc_delay_sampling": self.cfg.rtc.delay_distribution if is_rtc else None,
            "rtc_num_inference_timesteps": (
                self.cfg.policy.fm_steps
                if is_continuous_architecture(self.cfg.policy.architecture)
                else self.cfg.policy.discrete_rounds
            ) if is_rtc else None,
            "rtc_spline_span_length_steps": (
                self.cfg.spline.span_length_steps if representation == "bspline" else None
            ),
            "rtc_requires_previous_field": (
                "prev_action_chunk" if representation == "raw" else "prev_control_rows"
            ) if is_rtc else None,
            "available_unnorm_keys": ["new_embodiment"],
            "default_unnorm_key": "new_embodiment",
            "gripper_constraint": (
                {
                    "mode": "binary",
                    "dimension": -1,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "threshold": self.gripper_threshold,
                }
                if self.binary_gripper
                else None
            ),
        }

    def reset(self, **_: Any) -> dict[str, Any]:
        # Requests are stateless: the client explicitly supplies the previous
        # chunk/control rows to RTC calls. This avoids cross-client cache leaks.
        return {"reset": True, "stateful_server_cache": False}

    def _validate_unnorm_key(self, unnorm_key: str | None) -> None:
        if unnorm_key not in {None, "new_embodiment"}:
            raise ValueError("unnorm_key must be 'new_embodiment' for this checkpoint")

    def _state_bounds(self, stats: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "state_min" in stats and "state_max" in stats:
            low = np.asarray(stats["state_min"], dtype=np.float32)
            high = np.asarray(stats["state_max"], dtype=np.float32)
        else:
            splits_path = self.checkpoint.prepared_path / "splits.json"
            actions_path = self.checkpoint.prepared_path / "actions"
            if not splits_path.is_file() or not actions_path.is_dir():
                raise ValueError(
                    "normalization.json has no state_min/state_max and prepared "
                    "training action sidecars are unavailable; cannot reproduce the "
                    "legacy Piper min/max-normalized state interface"
                )
            train_ids = json.loads(splits_path.read_text())["train"]
            lows: list[np.ndarray] = []
            highs: list[np.ndarray] = []
            for episode_id in train_ids:
                with np.load(actions_path / f"episode_{int(episode_id):06d}.npz") as episode:
                    state = np.asarray(episode["state"], dtype=np.float32)
                lows.append(state.min(axis=0))
                highs.append(state.max(axis=0))
            low = np.stack(lows).min(axis=0)
            high = np.stack(highs).max(axis=0)
        if low.shape != (7,) or high.shape != (7,) or np.any(high <= low):
            raise ValueError("invalid state_min/state_max deployment bounds")
        return (
            torch.tensor(low, device=self.device),
            torch.tensor(high, device=self.device),
        )

    def _prepare_examples(
        self, examples: Sequence[Mapping[str, Any]], state_coordinates: str
    ) -> dict[str, torch.Tensor]:
        if not isinstance(examples, (list, tuple)) or not examples:
            raise ValueError("examples must be a non-empty list")
        if state_coordinates not in {"normalized", "zscore", "physical"}:
            raise ValueError("state_coordinates must be 'normalized', 'zscore', or 'physical'")

        states: list[np.ndarray] = []
        precomputed: list[np.ndarray] = []
        rgb_histories: list[list[list[np.ndarray]]] = []
        uses_precomputed: bool | None = None
        for index, example in enumerate(examples):
            if not isinstance(example, Mapping):
                raise ValueError(f"example {index} must be a mapping")
            language = example.get("lang", self.task_instruction)
            if isinstance(language, (list, tuple)) and len(language) == 1:
                language = language[0]
            if language != self.task_instruction:
                raise ValueError(
                    f"example {index} task {language!r} does not match fixed task "
                    f"{self.task_instruction!r}"
                )
            state = np.asarray(example.get("state"), dtype=np.float32).reshape(-1)
            if state.shape != (7,) or not np.isfinite(state).all():
                raise ValueError(f"example {index} state must be a finite 7-vector")

            has_features = "vision_features" in example
            if uses_precomputed is None:
                uses_precomputed = has_features
            elif uses_precomputed != has_features:
                raise ValueError("a batch cannot mix images and precomputed vision_features")
            if has_features:
                states.append(state)
                features = np.asarray(example["vision_features"], dtype=np.float32)
                expected = (
                    len(self.cfg.data.camera_keys),
                    self.cfg.vision.pooled_grid**2,
                    2176,
                )
                if features.shape != expected or not np.isfinite(features).all():
                    raise ValueError(
                        f"example {index} vision_features must have shape {expected}"
                    )
                precomputed.append(features)
            else:
                if not uses_bsp_image_encoder(self.cfg.policy.architecture):
                    images = example.get("image")
                    if not isinstance(images, (list, tuple)):
                        images = [images]
                    if len(images) != len(self.cfg.data.camera_keys):
                        raise ValueError(
                            f"example {index} must provide {len(self.cfg.data.camera_keys)} "
                            "images in [global, hand] order"
                        )
                    rgb_histories.append(
                        [[_as_numpy_image(image) for image in images]]
                    )
                    states.append(state)
                    continue
                horizon = self.cfg.data.observation_horizon
                image_history = example.get("image_history")
                state_history = example.get("state_history")
                if image_history is None:
                    images = example.get("image")
                    if not isinstance(images, (list, tuple)):
                        images = [images]
                    image_history = [images for _ in range(horizon)]
                if not isinstance(image_history, (list, tuple)) or len(image_history) != horizon:
                    raise ValueError(
                        f"example {index} image_history must contain {horizon} timesteps"
                    )
                parsed_history: list[list[np.ndarray]] = []
                for timestep, images in enumerate(image_history):
                    if not isinstance(images, (list, tuple)):
                        images = [images]
                    if len(images) != len(self.cfg.data.camera_keys):
                        raise ValueError(
                            f"example {index} image_history[{timestep}] must provide "
                            f"{len(self.cfg.data.camera_keys)} images in [global, hand] order"
                        )
                    parsed_history.append([_as_numpy_image(image) for image in images])
                rgb_histories.append(parsed_history)

                if state_history is None:
                    parsed_states = np.repeat(state[None], horizon, axis=0)
                else:
                    parsed_states = np.asarray(state_history, dtype=np.float32)
                    if parsed_states.shape != (horizon, 7) or not np.isfinite(parsed_states).all():
                        raise ValueError(
                            f"example {index} state_history must be finite with shape {(horizon, 7)}"
                        )
                states.append(parsed_states)

        state_tensor = torch.from_numpy(np.stack(states)).to(self.device)
        if state_coordinates == "normalized":
            state_tensor = (state_tensor + 1.0) * 0.5 * (self.state_max - self.state_min) + self.state_min
            state_tensor = (state_tensor - self.state_mean) / self.state_std
        elif state_coordinates == "physical":
            state_tensor = (state_tensor - self.state_mean) / self.state_std
        if uses_precomputed:
            vision = torch.from_numpy(np.stack(precomputed)).to(self.device)
            return {"vision_features": vision, "state": state_tensor}

        if uses_bsp_image_encoder(self.cfg.policy.architecture):
            resized = []
            for history in rgb_histories:
                resized_steps = []
                for images in history:
                    resized_cameras = []
                    for image in images:
                        tensor = torch.from_numpy(image).to(self.device).permute(2, 0, 1)
                        tensor = F.interpolate(
                            tensor[None].float() / 255.0,
                            size=(self.cfg.vision.image_size, self.cfg.vision.image_size),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )[0]
                        resized_cameras.append(tensor)
                    resized_steps.append(torch.stack(resized_cameras))
                resized.append(torch.stack(resized_steps))
            return {"images": torch.stack(resized), "state": state_tensor}

        # Batch the normal Piper case (equal native camera resolutions) in one
        # frozen-frontend pass. Retain a correct fallback for mixed resolutions.
        all_images = [
            image
            for history in rgb_histories
            for images in history
            for image in images
        ]
        if len({image.shape for image in all_images}) == 1:
            rgb = torch.from_numpy(np.stack(all_images)).to(self.device)
            chunks = list(self.vision_encoder(rgb).fused_patches)
        else:
            chunks = []
            for image in all_images:
                rgb = torch.from_numpy(image[None]).to(self.device)
                encoded = self.vision_encoder(rgb)
                chunks.append(encoded.fused_patches[0])
        b = len(examples)
        c = len(self.cfg.data.camera_keys)
        vision = torch.stack(chunks).reshape(
            b, c, self.cfg.vision.pooled_grid**2, -1
        )
        return {"vision_features": vision, "state": state_tensor}

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        seed: int,
        fm_steps: int | None,
        discrete_rounds: int | None,
        use_cache: bool,
        prefix_values: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        started = time.perf_counter()
        with torch.inference_mode(), torch.random.fork_rng(devices=devices), self._autocast():
            torch.manual_seed(int(seed))
            predicted = self.model.sample(
                batch,
                steps=fm_steps,
                rounds=discrete_rounds,
                prefix_values=prefix_values,
                fixed_mask=fixed_mask,
                use_cache=use_cache,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sampling_ms = (time.perf_counter() - started) * 1000.0
        controls = (
            predicted.float()
            if is_continuous_architecture(self.cfg.policy.architecture)
            else self.codec.decode_tokens(predicted).float()
        )
        normalized_actions = self.codec.decode_controls(controls)
        return controls, normalized_actions, sampling_ms

    def _physical_actions(self, normalized: torch.Tensor) -> torch.Tensor:
        actions = (normalized + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
        if self.binary_gripper:
            actions = actions.clone()
            actions[..., -1] = (actions[..., -1] > self.gripper_threshold).to(actions.dtype)
        return actions

    def _result(
        self,
        controls: torch.Tensor,
        normalized_actions: torch.Tensor,
        sampling_ms: float,
        **inference_metadata: Any,
    ) -> dict[str, Any]:
        actions = self._physical_actions(normalized_actions)
        result: dict[str, Any] = {
            "actions": actions.float().cpu().numpy(),
            "sampling_ms": np.float64(sampling_ms),
            "inference_metadata": {
                "architecture": self.cfg.policy.architecture,
                "action_representation": self.cfg.data.action_representation,
                "fm_steps": self.cfg.policy.fm_steps,
                "discrete_rounds": self.cfg.policy.discrete_rounds,
                **inference_metadata,
            },
        }
        if self.cfg.data.action_representation == "bspline":
            result["normalized_control_rows"] = controls.float().cpu().numpy()
        return result

    def predict_action(
        self,
        examples: Sequence[Mapping[str, Any]],
        unnorm_key: str | None = None,
        *,
        state_coordinates: str = "normalized",
        seed: int = 20260915,
        fm_steps: int | None = None,
        discrete_rounds: int | None = None,
        use_cache: bool = True,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_unnorm_key(unnorm_key)
        allowed_legacy = {"do_sample", "use_ddim", "num_ddim_steps"}
        unknown = set(legacy_kwargs) - allowed_legacy
        if unknown:
            raise TypeError(f"unsupported inference arguments: {sorted(unknown)}")
        batch = self._prepare_examples(examples, state_coordinates)
        controls, normalized, elapsed = self._sample(
            batch,
            seed=seed,
            fm_steps=fm_steps,
            discrete_rounds=discrete_rounds,
            use_cache=use_cache,
        )
        return self._result(
            controls, normalized, elapsed, rtc=False, seed=int(seed), cache_enabled=bool(use_cache)
        )

    def predict_action_realtime(
        self,
        examples: Sequence[Mapping[str, Any]],
        inference_delay: int,
        prev_action_chunk: np.ndarray | None = None,
        prev_control_rows: np.ndarray | None = None,
        unnorm_key: str | None = None,
        *,
        state_coordinates: str = "normalized",
        seed: int = 20260915,
        fm_steps: int | None = None,
        discrete_rounds: int | None = None,
        use_cache: bool = True,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_unnorm_key(unnorm_key)
        if not self.checkpoint.is_rtc:
            raise RuntimeError("realtime inference requires an RTC/ttRTC checkpoint")
        if legacy_kwargs:
            unsupported = sorted(set(legacy_kwargs) - {"mode"})
            if unsupported:
                raise TypeError(f"unsupported realtime arguments: {unsupported}")
        raw_delay = int(inference_delay)
        if raw_delay < 1 or raw_delay > self.cfg.rtc.raw_delay_max:
            raise ValueError(
                f"inference_delay must be in 1..{self.cfg.rtc.raw_delay_max} raw action steps"
            )
        batch = self._prepare_examples(examples, state_coordinates)
        batch_size = len(examples)

        if self.cfg.data.action_representation == "bspline":
            if prev_action_chunk is not None:
                raise ValueError(
                    "B-spline RTC accepts prev_control_rows in normalized spline-parameter "
                    "space; decoded prev_action_chunk is forbidden"
                )
            if prev_control_rows is None:
                raise ValueError("B-spline RTC requires prev_control_rows")
            # msgpack decoders commonly expose read-only NumPy views.  Own the
            # request buffer before crossing into torch so RTC never aliases
            # transport memory or emits a non-writable-array warning.
            previous = torch.as_tensor(
                np.array(prev_control_rows, dtype=np.float32, copy=True),
                device=self.device,
            )
            if previous.ndim == 2:
                previous = previous.unsqueeze(0)
            expected = (batch_size, self.cfg.spline.num_basis, 7)
            if tuple(previous.shape) != expected:
                raise ValueError(f"prev_control_rows must have shape {expected}, got {tuple(previous.shape)}")
            spans = map_delay(
                raw_delay,
                frequency_hz=self.cfg.data.frequency_hz,
                span_length_steps=self.cfg.spline.span_length_steps,
                degree=self.cfg.spline.degree,
            ).affected_spans
            delay_tensor = torch.full((batch_size,), raw_delay, device=self.device, dtype=torch.long)
            shifted = self.codec.shift_and_refit(previous, delay_tensor)
            span_tensor = torch.full((batch_size,), spans, device=self.device, dtype=torch.long)
            fixed = control_support_mask(
                span_tensor,
                num_basis=self.cfg.spline.num_basis,
                degree=self.cfg.spline.degree,
            )
        else:
            if prev_control_rows is not None:
                raise ValueError("raw-action RTC does not accept prev_control_rows")
            if prev_action_chunk is None:
                raise ValueError("raw-action RTC requires prev_action_chunk")
            previous_physical = torch.as_tensor(
                np.array(prev_action_chunk, dtype=np.float32, copy=True),
                device=self.device,
            )
            if previous_physical.ndim == 2:
                previous_physical = previous_physical.unsqueeze(0)
            expected = (batch_size, self.cfg.data.action_horizon, 7)
            if tuple(previous_physical.shape) != expected:
                raise ValueError(f"prev_action_chunk must have shape {expected}, got {tuple(previous_physical.shape)}")
            previous = 2.0 * (previous_physical - self.action_low) / (
                self.action_high - self.action_low
            ) - 1.0
            delay_tensor = torch.full((batch_size,), raw_delay, device=self.device, dtype=torch.long)
            shifted = self.codec.shift_and_refit(previous, delay_tensor)
            fixed = raw_action_prefix_mask(delay_tensor, self.cfg.data.action_horizon)
            spans = 0

        prefix = (
            shifted
            if is_continuous_architecture(self.cfg.policy.architecture)
            else self.codec.encode_tokens(shifted)
        )
        controls, normalized, elapsed = self._sample(
            batch,
            seed=seed,
            fm_steps=fm_steps,
            discrete_rounds=discrete_rounds,
            use_cache=use_cache,
            prefix_values=prefix,
            fixed_mask=fixed,
        )
        return self._result(
            controls,
            normalized,
            elapsed,
            rtc=True,
            seed=int(seed),
            delay_raw_actions=raw_delay,
            affected_spans=spans,
            fixed_control_rows=int(fixed[0, :, 0].sum().item()),
            cache_enabled=bool(use_cache),
        )
