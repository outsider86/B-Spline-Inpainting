from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from robot_policy.config import Config, config_from_dict


REQUIRED_SIDECARS = ("encoder.json", "normalization.json")


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    architecture: str
    training_type: str
    config: Config
    prepared_path: Path
    update: int
    parent_checkpoint: str | None
    code_snapshot_sha256: str | None

    @property
    def is_rtc(self) -> bool:
        return self.training_type.lower() in {"rtc", "ttrtc"}


def _has_sidecars(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_SIDECARS)


def _candidate_prepared_paths(
    checkpoint: Path, configured: str, representation: str
) -> list[Path]:
    configured_path = Path(configured).expanduser()
    candidates: list[Path] = []
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.extend(
            [
                Path.cwd() / configured_path,
                Path(__file__).resolve().parents[3] / configured_path,
            ]
        )
    # Conventional run layouts.
    candidates.extend(
        [
            checkpoint.parent.parent / "prepared",
            checkpoint.parent.parent.parent / "prepared",
        ]
    )
    # Organized releases intentionally share one immutable set of
    # representation sidecars rather than duplicating them per checkpoint.
    for parent in checkpoint.parents:
        # Versioned Hugging Face releases use
        # NewModel/vN/sidecars/{raw,bspline}.  Checking each ancestor keeps
        # downloaded releases self-contained without hard-coding v1/v2.
        candidates.append(parent / "sidecars" / representation)
        candidates.append(parent / "summary" / "cache" / representation)
        if parent.name == "SWEEP":
            break
    result: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def resolve_prepared_path(
    checkpoint: str | Path,
    configured: str,
    representation: str,
    override: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    candidates = (
        [Path(override).expanduser().resolve()]
        if override is not None
        else _candidate_prepared_paths(checkpoint, configured, representation)
    )
    for candidate in candidates:
        if _has_sidecars(candidate):
            _validate_sidecars(candidate, representation)
            return candidate
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate encoder.json and normalization.json for the checkpoint. "
        f"Checked:\n  - {rendered}\nPass --prepared-path explicitly."
    )


def _validate_sidecars(path: Path, representation: str) -> None:
    encoder = json.loads((path / "encoder.json").read_text())
    normalization = json.loads((path / "normalization.json").read_text())
    expected_type = "RawActionSequenceConfig" if representation == "raw" else "UniformLeftBSplineConfig"
    if encoder.get("type") != expected_type:
        raise ValueError(
            f"Prepared sidecar representation mismatch at {path}: "
            f"expected {expected_type}, got {encoder.get('type')!r}"
        )
    for key in ("state_mean", "state_std", "action_q01", "action_q99"):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(f"{path / 'normalization.json'} has invalid {key!r}")
    embedded = normalization.get("encoder")
    if embedded is not None and embedded != encoder:
        raise ValueError("encoder.json disagrees with normalization.json['encoder']")
    expected_hash = normalization.get("encoder_sha256")
    if expected_hash:
        canonical = json.dumps(encoder, sort_keys=True).encode()
        if sha256(canonical).hexdigest() != expected_hash:
            raise ValueError(f"encoder hash mismatch in {path}")


def inspect_checkpoint(
    checkpoint: str | Path, prepared_path: str | Path | None = None
) -> CheckpointMetadata:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    try:
        config_values = payload.get("config")
        if not isinstance(config_values, dict):
            raise ValueError(f"checkpoint {path} has no embedded config")
        cfg = config_from_dict(config_values)
        architecture = str(payload.get("architecture", ""))
        if architecture != cfg.policy.architecture:
            raise ValueError(
                f"checkpoint architecture {architecture!r} disagrees with "
                f"embedded config {cfg.policy.architecture!r}"
            )
        resolved = resolve_prepared_path(
            path,
            cfg.data.prepared_path,
            cfg.data.action_representation,
            prepared_path,
        )
        cfg.data.prepared_path = str(resolved)
        return CheckpointMetadata(
            path=path,
            architecture=architecture,
            training_type=str(payload.get("training_type", "base")),
            config=cfg,
            prepared_path=resolved,
            update=int(payload.get("update", 0)),
            parent_checkpoint=payload.get("parent_checkpoint"),
            code_snapshot_sha256=payload.get("code_snapshot_sha256"),
        )
    finally:
        del payload


def load_deployment_policy(metadata: CheckpointMetadata, device: torch.device):
    """Load only the model state needed for serving a training checkpoint."""
    from robot_policy.policies import create_policy

    payload: dict[str, Any] = torch.load(
        metadata.path, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("architecture") != metadata.architecture:
        raise ValueError("checkpoint metadata changed between inspection and loading")
    model = create_policy(metadata.config)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    return model.to(device).eval()
