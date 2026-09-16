from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _numeric_deltas(left: Any, right: Any, path: str = "") -> list[tuple[str, float]]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [(path + ".keys", float("inf"))]
        out = []
        for key in sorted(left):
            out.extend(_numeric_deltas(left[key], right[key], f"{path}.{key}"))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [(path + ".length", float("inf"))]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(_numeric_deltas(a, b, f"{path}[{index}]"))
        return out
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return [(path, abs(float(left) - float(right)))]
    return [] if left == right else [(path, float("inf"))]


def compare(first_path: str | Path, rerun_path: str | Path) -> dict[str, Any]:
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    rerun = torch.load(rerun_path, map_location="cpu", weights_only=False)
    identity_keys = ("architecture", "training_type", "update", "code_snapshot_sha256")
    identity = {key: first.get(key) == rerun.get(key) for key in identity_keys}
    first_cfg = dict(first["config"]); rerun_cfg = dict(rerun["config"])
    first_cfg.pop("wandb", None); rerun_cfg.pop("wandb", None)
    identity["configuration_except_tracking"] = first_cfg == rerun_cfg
    trace_deltas = _numeric_deltas(first.get("loss_trace", []), rerun.get("loss_trace", []), "loss_trace")
    validation_deltas = _numeric_deltas(first.get("history", []), rerun.get("history", []), "history")
    state_equal = all(
        name in rerun["model"] and torch.equal(tensor.cpu(), rerun["model"][name].cpu())
        for name, tensor in first["model"].items()
    ) and set(first["model"]) == set(rerun["model"])
    finite_trace = [delta for _, delta in trace_deltas]
    finite_validation = [delta for _, delta in validation_deltas]
    report = {
        "first_checkpoint": str(Path(first_path).resolve()),
        "rerun_checkpoint": str(Path(rerun_path).resolve()),
        "identity_checks": identity,
        "seed": first["config"]["train"]["seed"],
        "deterministic_mode": first.get("determinism"),
        "loss_trace_points": len(first.get("loss_trace", [])),
        "validation_points": len(first.get("history", [])),
        "max_loss_trace_absolute_delta": max(finite_trace, default=0.0),
        "max_validation_absolute_delta": max(finite_validation, default=0.0),
        "model_state_exactly_equal": state_equal,
        "first_model_sha256": _tensor_digest(first["model"]),
        "rerun_model_sha256": _tensor_digest(rerun["model"]),
    }
    report["reproducible"] = (
        all(identity.values())
        and report["max_loss_trace_absolute_delta"] == 0.0
        and report["max_validation_absolute_delta"] == 0.0
        and state_equal
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare(args.first, args.rerun)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["reproducible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
