from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ServerConfig:
    """Small, portable launch contract kept beside a policy checkpoint."""

    path: Path
    prepared_path: str | None
    task_instruction: str
    precision: str
    binary_gripper: bool
    gripper_threshold: float
    expected: dict[str, Any]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "prepared_path",
    "task_instruction",
    "precision",
    "binary_gripper",
    "gripper_threshold",
    "expected",
}
_EXPECTED_KEYS = {
    "architecture",
    "action_representation",
    "observation_horizon",
    "state_dim",
    "camera_keys",
}


def load_server_config(path: str | Path) -> ServerConfig:
    source = Path(path).expanduser().resolve()
    values = json.loads(source.read_text())
    if not isinstance(values, dict):
        raise ValueError("policy-server JSON must contain an object")
    unknown = set(values) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown policy-server JSON keys: {sorted(unknown)}")
    if values.get("schema_version") != 1:
        raise ValueError("policy-server JSON schema_version must be 1")
    expected = values.get("expected", {})
    if not isinstance(expected, dict):
        raise ValueError("policy-server JSON expected must be an object")
    unknown_expected = set(expected) - _EXPECTED_KEYS
    if unknown_expected:
        raise ValueError(
            f"unknown policy-server expected keys: {sorted(unknown_expected)}"
        )
    precision = values.get("precision", "bf16")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("policy-server JSON precision must be fp32 or bf16")
    threshold = float(values.get("gripper_threshold", 0.3))
    if not 0.0 < threshold < 1.0:
        raise ValueError("policy-server JSON gripper_threshold must lie in (0,1)")
    prepared = values.get("prepared_path")
    if prepared is not None:
        prepared_path = Path(prepared).expanduser()
        if not prepared_path.is_absolute():
            prepared_path = source.parent / prepared_path
        prepared = str(prepared_path.resolve())
    return ServerConfig(
        path=source,
        prepared_path=prepared,
        task_instruction=str(values.get("task_instruction", "Stack the cups.")),
        precision=precision,
        binary_gripper=bool(values.get("binary_gripper", True)),
        gripper_threshold=threshold,
        expected=expected,
    )


def validate_server_config(
    config: ServerConfig, metadata: Mapping[str, Any]
) -> None:
    """Reject accidentally paired model/JSON files before serving requests."""

    actual = {
        "architecture": metadata["architecture"],
        "action_representation": metadata["action_representation"],
        "observation_horizon": metadata["observation_horizon"],
        "state_dim": metadata["state_shape"][0],
        "camera_keys": metadata["camera_keys"],
    }
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in config.expected.items()
        if expected != actual[key]
    }
    if mismatches:
        raise ValueError(
            "model does not match policy-server JSON: "
            + json.dumps(mismatches, sort_keys=True)
        )


def write_server_config(
    checkpoint: str | Path,
    output: str | Path,
    *,
    task_instruction: str,
    prepared_path: str | Path | None = None,
) -> Path:
    """Write the launch JSON paired with a deployment-loadable model."""

    from robot_policy.deployment.checkpoint import inspect_checkpoint

    metadata = inspect_checkpoint(checkpoint, prepared_path=prepared_path)
    destination = Path(output).expanduser().resolve()
    values = {
        "schema_version": 1,
        "prepared_path": str(metadata.prepared_path),
        "task_instruction": task_instruction,
        "precision": "bf16",
        "binary_gripper": True,
        "gripper_threshold": 0.3,
        "expected": {
            "architecture": metadata.architecture,
            "action_representation": metadata.config.data.action_representation,
            "observation_horizon": metadata.config.data.observation_horizon,
            "state_dim": metadata.config.data.state_dim,
            "camera_keys": list(metadata.config.data.camera_keys),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(destination)
    return destination
