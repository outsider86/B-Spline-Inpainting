from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# New experiments intentionally compare only the continuous FM baseline and the
# joint discrete-diffusion policy.  Keep the layerwise ID loadable so the
# existing checkpoints and historical reports remain reproducible.
ACTIVE_ARCHITECTURES = ("fm", "discrete_joint")
BSP_UNET_ARCHITECTURES = ("bsp_unet_fm", "bsp_unet_discrete")
TRAINABLE_ARCHITECTURES = ACTIVE_ARCHITECTURES + BSP_UNET_ARCHITECTURES
LEGACY_ARCHITECTURES = ("discrete_layerwise",)
SUPPORTED_ARCHITECTURES = TRAINABLE_ARCHITECTURES + LEGACY_ARCHITECTURES

# Capacity scope follows the same active-versus-legacy contract.  DiT-L
# checkpoints and configs remain loadable for historical reproducibility, but
# new research runs use DiT-S/B only.
ACTIVE_MODEL_SIZES = ("custom", "DiT-S", "DiT-B")
BSP_UNET_MODEL_SIZES = ("BSP-UNet",)
TRAINABLE_MODEL_SIZES = ACTIVE_MODEL_SIZES + BSP_UNET_MODEL_SIZES
LEGACY_MODEL_SIZES = ("DiT-L",)
SUPPORTED_MODEL_SIZES = TRAINABLE_MODEL_SIZES + LEGACY_MODEL_SIZES


def is_continuous_architecture(architecture: str) -> bool:
    return architecture in {"fm", "bsp_unet_fm"}


def is_discrete_architecture(architecture: str) -> bool:
    return architecture in {"discrete_layerwise", "discrete_joint", "bsp_unet_discrete"}


def uses_bsp_image_encoder(architecture: str) -> bool:
    return architecture in BSP_UNET_ARCHITECTURES


def require_active_architecture(architecture: str, operation: str) -> None:
    if architecture not in TRAINABLE_ARCHITECTURES:
        raise ValueError(
            f"{architecture!r} is legacy/load-only and cannot be used for {operation}; "
            f"choose one of {TRAINABLE_ARCHITECTURES}"
        )


def require_active_model_size(model_size: str, operation: str) -> None:
    if model_size not in TRAINABLE_MODEL_SIZES:
        raise ValueError(
            f"{model_size!r} is legacy/load-only and cannot be used for {operation}; "
            f"choose one of {TRAINABLE_MODEL_SIZES}"
        )


@dataclass
class DataConfig:
    dataset_path: str = "../Data/stacking_cups_action_30hz"
    prepared_path: str = "outputs/prepared"
    action_representation: str = "bspline"
    vision_cache_path: str | None = None
    camera_keys: tuple[str, ...] = ("observation.images.global", "observation.images.hand")
    state_key: str = "observation.state"
    action_key: str = "action"
    observation_horizon: int = 1
    observation_source: str = "features"
    rgb_cache_path: str | None = None
    parent_prediction_cache_path: str | None = None
    action_horizon: int = 30
    frequency_hz: float = 30.0
    split_seed: int = 20260915
    train_episodes: int = 42
    val_episodes: int = 5
    test_episodes: int = 5


@dataclass
class SplineConfig:
    degree: int = 3
    span_length_steps: int = 2
    num_basis: int = 18
    vocab_size: int = 256
    regularization: float = 1e-4
    implementation_version: str = "uniform_left_direct_fit_v1"


@dataclass
class VisionConfig:
    image_size: int = 224
    pooled_grid: int = 4
    tokenizer: str = "dinov2_siglip"
    feature_dim: int = 2176
    resampler_tokens_per_camera: int | None = None
    dino_model: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model: str = "vit_so400m_patch14_siglip_224"
    extraction_layer_offset: int = -2
    pretrained: bool = True
    cache_dtype: str = "float16"
    crop_size: int = 76
    spatial_keypoints: int = 32
    # Optional two-camera VisualCore artifact extracted from a RoboMimic image
    # policy. The default remains the paper-reference scratch GroupNorm encoder.
    bsp_encoder_weights: str | None = None
    bsp_encoder_norm: str = "group"
    bsp_rgb_normalization: str = "minus_one_one"


@dataclass
class PolicyConfig:
    architecture: str = "fm"
    model_size: str = "custom"
    hidden_dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    discrete_rounds: int = 8
    block_size: int = 21
    # A fraction of training examples start from a completely masked action
    # sequence.  This closes the exposure gap between partial-corruption
    # teacher training and generation from scratch.
    discrete_full_mask_probability: float = 0.0
    fm_steps: int = 12
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)
    unet_time_dim: int = 128
    unet_kernel_size: int = 5
    unet_groups: int = 8
    discrete_embed_dim: int = 32


@dataclass
class TrainConfig:
    seed: int = 7
    updates: int = 2000
    rtc_updates: int = 800
    batch_size: int = 64
    effective_batch_size: int = 128
    learning_rate: float = 3e-4
    rtc_learning_rate: float | None = None
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    warmup_updates: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    # Reject rare finite-but-invalid BF16 FM backward spikes before Adam can
    # retain their moments. Healthy audited FM pre-clip norms are single-digit.
    fm_grad_skip_threshold: float = 3.0
    fm_rtc_grad_skip_threshold: float = 100.0
    fm_grad_skip_after_updates: int = 1000
    fm_math_sdp_training: bool = True
    eval_every: int = 100
    validation_max_batches: int = 16
    # Decouple validation coverage from the optimization batch size. When
    # unset, legacy experiments retain train.batch_size behavior.
    validation_batch_size: int | None = None
    save_every: int = 200
    # Keep immutable step-numbered snapshots in addition to the rolling resume
    # checkpoint. Disabled by default to avoid multiplying storage use.
    keep_periodic_checkpoints: bool = False
    num_workers: int = 4
    precision: str = "bf16"
    lambda_l1: float = 1.0
    deterministic: bool = True
    use_ema: bool = False
    ema_update_after_step: int = 0
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_value: float = 0.0
    ema_max_value: float = 0.9999


@dataclass
class WandbConfig:
    enabled: bool = False
    mode: str = "online"
    entity: str | None = None
    base_project: str = "raw-actions-basic"
    rtc_project: str = "raw-actions-ttRTC"
    run_suffix: str = ""
    output_dir: str = "outputs/wandb"


@dataclass
class RTCConfig:
    spline_delay_min: int = 1
    spline_delay_max: int = 5
    raw_delay_min: int = 1
    raw_delay_max: int = 10
    delay_distribution: str = "truncated_decreasing_exponential_reference"
    zero_delay_eval: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rtc: RTCConfig = field(default_factory=RTCConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.policy.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"unknown architecture {self.policy.architecture!r}")
        if self.data.observation_horizon not in {1, 2}:
            raise ValueError("observation_horizon must be 1 or 2")
        if self.data.observation_source not in {"features", "rgb"}:
            raise ValueError("observation_source must be 'features' or 'rgb'")
        if uses_bsp_image_encoder(self.policy.architecture):
            if self.data.observation_source != "rgb":
                raise ValueError("BSP U-Net policies require data.observation_source='rgb'")
            if self.policy.model_size != "BSP-UNet":
                raise ValueError("BSP U-Net policies require policy.model_size='BSP-UNet'")
            if self.vision.pretrained != bool(self.vision.bsp_encoder_weights):
                raise ValueError(
                    "BSP U-Net vision.pretrained must be false with no "
                    "bsp_encoder_weights, or true with bsp_encoder_weights"
                )
        elif self.data.observation_source != "features":
            raise ValueError("token-DiT policies require data.observation_source='features'")
        if self.data.action_representation not in {"bspline", "raw"}:
            raise ValueError("action_representation must be 'bspline' or 'raw'")
        size_presets = {
            "DiT-S": (384, 6, 4),
            "DiT-B": (768, 12, 12),
            "DiT-L": (1024, 24, 16),
        }
        if self.policy.model_size == "BSP-UNet":
            if tuple(self.policy.unet_down_dims) != (256, 512, 1024):
                raise ValueError("BSP-UNet requires unet_down_dims=(256,512,1024)")
            if (self.policy.unet_time_dim, self.policy.unet_kernel_size, self.policy.unet_groups) != (128, 5, 8):
                raise ValueError("BSP-UNet requires time_dim/kernel_size/groups=(128,5,8)")
        elif self.policy.model_size != "custom":
            if self.policy.model_size not in size_presets:
                raise ValueError("policy.model_size must be custom, DiT-S, DiT-B, or DiT-L")
            actual = (self.policy.hidden_dim, self.policy.depth, self.policy.heads)
            expected = size_presets[self.policy.model_size]
            if actual != expected:
                raise ValueError(
                    f"policy.model_size {self.policy.model_size} requires "
                    f"hidden_dim/depth/heads={expected}, got {actual}"
                )
        if self.policy.hidden_dim % self.policy.heads:
            raise ValueError("policy.hidden_dim must be divisible by policy.heads")
        if self.vision.crop_size > self.vision.image_size:
            raise ValueError("vision.crop_size cannot exceed vision.image_size")
        if self.vision.bsp_encoder_norm not in {"group", "batch"}:
            raise ValueError("vision.bsp_encoder_norm must be 'group' or 'batch'")
        if self.vision.bsp_rgb_normalization not in {"minus_one_one", "zero_one"}:
            raise ValueError(
                "vision.bsp_rgb_normalization must be 'minus_one_one' or 'zero_one'"
            )
        if self.vision.bsp_encoder_weights and not uses_bsp_image_encoder(
            self.policy.architecture
        ):
            raise ValueError(
                "vision.bsp_encoder_weights is only valid for BSP U-Net policies"
            )
        if self.vision.tokenizer not in {"dinov2", "dinov2_siglip"}:
            raise ValueError("vision.tokenizer must be 'dinov2' or 'dinov2_siglip'")
        expected_vision_dim = 1024 if self.vision.tokenizer == "dinov2" else 2176
        if self.vision.feature_dim != expected_vision_dim:
            raise ValueError(
                f"vision.feature_dim must be {expected_vision_dim} for "
                f"vision.tokenizer={self.vision.tokenizer!r}"
            )
        if (
            self.vision.resampler_tokens_per_camera is not None
            and self.vision.resampler_tokens_per_camera < 1
        ):
            raise ValueError("vision.resampler_tokens_per_camera must be positive")
        if self.data.action_representation == "bspline":
            expected = self.spline.num_basis - self.spline.degree
            if expected * self.spline.span_length_steps != self.data.action_horizon:
                raise ValueError("(num_basis - degree) * span_length_steps must equal action_horizon")
        if self.spline.vocab_size != 256:
            raise ValueError("the confirmed action vocabulary has 256 bins")
        if not 0.0 <= self.policy.discrete_full_mask_probability <= 1.0:
            raise ValueError("discrete_full_mask_probability must be within [0, 1]")
        if self.rtc.spline_delay_min < 1 or self.rtc.spline_delay_max > 5:
            raise ValueError("spline RTC delays must be within 1..5 spans")
        if self.rtc.raw_delay_min < 1 or self.rtc.raw_delay_max > 10:
            raise ValueError("raw RTC delays must be within 1..10 actions")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.train.validation_max_batches < 1:
            raise ValueError("train.validation_max_batches must be positive")
        if (
            self.train.validation_batch_size is not None
            and self.train.validation_batch_size < 1
        ):
            raise ValueError("train.validation_batch_size must be positive")
        if self.train.save_every < 1:
            raise ValueError("train.save_every must be positive")
        if self.train.rtc_learning_rate is not None and self.train.rtc_learning_rate <= 0:
            raise ValueError("train.rtc_learning_rate must be positive when configured")
        if not 0 <= self.train.optimizer_beta1 < 1 or not 0 <= self.train.optimizer_beta2 < 1:
            raise ValueError("optimizer betas must lie in [0,1)")
        if self.train.optimizer_epsilon <= 0:
            raise ValueError("optimizer epsilon must be positive")
        if not 0 <= self.train.ema_min_value <= self.train.ema_max_value < 1:
            raise ValueError("EMA min/max values must satisfy 0 <= min <= max < 1")
        if self.train.fm_grad_skip_threshold <= self.train.grad_clip:
            raise ValueError("train.fm_grad_skip_threshold must exceed train.grad_clip")
        if self.train.fm_rtc_grad_skip_threshold <= self.train.grad_clip:
            raise ValueError("train.fm_rtc_grad_skip_threshold must exceed train.grad_clip")
        if self.train.fm_grad_skip_after_updates < self.train.warmup_updates:
            raise ValueError("train.fm_grad_skip_after_updates must cover optimizer warm-up")


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"unknown configuration key {type(instance).__name__}.{key}")
        current = getattr(instance, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(instance, key, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    values = yaml.safe_load(path.read_text()) or {}
    parent = values.pop("extends", None)
    if parent is None:
        return values
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_yaml_with_extends(parent_path, seen), values)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    source = Path(path).resolve()
    values = _load_yaml_with_extends(source)
    cfg = config_from_dict(values)
    for override in overrides or []:
        key, raw = override.split("=", 1)
        section, name = key.split(".", 1)
        value = yaml.safe_load(raw)
        _update_dataclass(getattr(cfg, section), {name: value})
    cfg.validate()
    # Runtime provenance is intentionally excluded from config_dict/asdict.
    cfg._source_path = str(source)
    cfg._overrides = list(overrides or [])
    return cfg


def config_from_dict(values: dict[str, Any]) -> Config:
    """Reconstruct a validated config from checkpoint-embedded metadata.

    Deployment must use the configuration saved with the checkpoint rather
    than requiring an operator to select a matching YAML file manually.
    """
    cfg = Config()
    for section, section_values in values.items():
        if not hasattr(cfg, section) or not isinstance(section_values, dict):
            raise ValueError(f"unknown configuration section {section!r}")
        _update_dataclass(getattr(cfg, section), section_values)
    cfg.validate()
    return cfg


def config_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)
