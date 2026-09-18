from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# New experiments intentionally compare only the continuous FM baseline and the
# joint discrete-diffusion policy.  Keep the layerwise ID loadable so the
# existing checkpoints and historical reports remain reproducible.
ACTIVE_ARCHITECTURES = ("fm", "discrete_joint")
LEGACY_ARCHITECTURES = ("discrete_layerwise",)
SUPPORTED_ARCHITECTURES = ACTIVE_ARCHITECTURES + LEGACY_ARCHITECTURES


def require_active_architecture(architecture: str, operation: str) -> None:
    if architecture not in ACTIVE_ARCHITECTURES:
        raise ValueError(
            f"{architecture!r} is legacy/load-only and cannot be used for {operation}; "
            f"choose one of {ACTIVE_ARCHITECTURES}"
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
    dino_model: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model: str = "vit_so400m_patch14_siglip_224"
    extraction_layer_offset: int = -2
    pretrained: bool = True
    cache_dtype: str = "float16"


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
    fm_steps: int = 12


@dataclass
class TrainConfig:
    seed: int = 7
    updates: int = 2000
    rtc_updates: int = 800
    batch_size: int = 64
    effective_batch_size: int = 128
    learning_rate: float = 3e-4
    rtc_learning_rate: float | None = None
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
    save_every: int = 200
    num_workers: int = 4
    precision: str = "bf16"
    lambda_l1: float = 1.0
    deterministic: bool = True


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
        if self.data.observation_horizon != 1:
            raise ValueError("the implemented observation_horizon is fixed to 1")
        if self.data.action_representation not in {"bspline", "raw"}:
            raise ValueError("action_representation must be 'bspline' or 'raw'")
        size_presets = {
            "DiT-S": (384, 6, 4),
            "DiT-B": (768, 12, 12),
            "DiT-L": (1024, 24, 16),
        }
        if self.policy.model_size != "custom":
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
        if self.data.action_representation == "bspline":
            expected = self.spline.num_basis - self.spline.degree
            if expected * self.spline.span_length_steps != self.data.action_horizon:
                raise ValueError("(num_basis - degree) * span_length_steps must equal action_horizon")
        if self.spline.vocab_size != 256:
            raise ValueError("the confirmed action vocabulary has 256 bins")
        if self.rtc.spline_delay_min < 1 or self.rtc.spline_delay_max > 5:
            raise ValueError("spline RTC delays must be within 1..5 spans")
        if self.rtc.raw_delay_min < 1 or self.rtc.raw_delay_max > 10:
            raise ValueError("raw RTC delays must be within 1..10 actions")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.train.validation_max_batches < 1:
            raise ValueError("train.validation_max_batches must be positive")
        if self.train.rtc_learning_rate is not None and self.train.rtc_learning_rate <= 0:
            raise ValueError("train.rtc_learning_rate must be positive when configured")
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
    values = _load_yaml_with_extends(Path(path))
    cfg = config_from_dict(values)
    for override in overrides or []:
        key, raw = override.split("=", 1)
        section, name = key.split(".", 1)
        value = yaml.safe_load(raw)
        _update_dataclass(getattr(cfg, section), {name: value})
    cfg.validate()
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
