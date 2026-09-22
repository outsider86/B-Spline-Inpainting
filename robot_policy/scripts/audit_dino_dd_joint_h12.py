#!/usr/bin/env python3
"""Completion audit for the DINOv2 h1/h2 joint discrete-DD experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs" / "DINO_DD_JOINT_H12"
VARIANTS = ("raw_h1", "raw_h2", "bspline_h1", "bspline_h2")
STAGES = ("base", "rtc")
SPLITS = ("train", "test")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_trace(payload: dict[str, Any]) -> bool:
    keys = ("loss", "loss_ce", "loss_l1", "action_mse", "grad_norm", "lr")
    return all(
        all(math.isfinite(float(row[key])) for key in keys if key in row)
        for row in payload.get("loss_trace", [])
    )


def _remote_wandb(payload: dict[str, Any]) -> dict[str, Any]:
    import wandb

    info = payload.get("wandb") or {}
    path = f"{info['entity']}/{info['project']}/{info['run_id']}"
    run = wandb.Api(timeout=30).run(path)
    return {
        "path": path,
        "state": run.state,
        "finished": run.state == "finished",
        "validation_mode": run.summary.get("validation_mode"),
        "best_validation_action_mse": run.summary.get("best_validation_action_mse"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-wandb", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "summary" / "completion_audit.json",
    )
    args = parser.parse_args()
    errors: list[str] = []
    checks: dict[str, Any] = {}

    cache = EXPERIMENT / "cache" / "dinov2_patch16"
    episode_files = sorted(cache.glob("episode_*.npy"))
    cache_shapes = set()
    frame_count = 0
    for path in episode_files:
        value = np.load(path, mmap_mode="r")
        cache_shapes.add(tuple(value.shape[1:]))
        frame_count += len(value)
    checks["vision_cache"] = {
        "episodes": len(episode_files),
        "frames": frame_count,
        "tail_shapes": [list(shape) for shape in sorted(cache_shapes)],
        "passed": len(episode_files) == 52
        and frame_count == 31_706
        and cache_shapes == {(2, 256, 1024)},
    }
    if not checks["vision_cache"]["passed"]:
        errors.append("shared DINOv2 cache coverage/shape is invalid")

    checkpoint_checks: dict[str, Any] = {}
    for variant in VARIANTS:
        representation = "bspline" if variant.startswith("bspline") else "raw"
        horizon = 2 if variant.endswith("h2") else 1
        base_path = EXPERIMENT / "checkpoints" / variant / "base.pt"
        parent_hash = _hash(base_path) if base_path.is_file() else None
        for stage in STAGES:
            key = f"{variant}/{stage}"
            path = EXPERIMENT / "checkpoints" / variant / f"{stage}.pt"
            if not path.is_file():
                checkpoint_checks[key] = {"exists": False}
                errors.append(f"missing checkpoint {key}")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cfg = payload["config"]
            expected_updates = 50_000 if stage == "base" else 5_000
            expected_type = {"base"} if stage == "base" else {"rtc", "ttrtc"}
            parent_ok = payload.get("parent_checkpoint") is None
            cache_ok = stage == "base"
            parent_manifest = None
            if stage == "rtc":
                parent = Path(payload.get("parent_checkpoint", "")).resolve()
                parent_ok = parent == base_path.resolve()
                parent_cache = (
                    EXPERIMENT / "cache" / "parent_predictions" / "discrete_joint"
                    / str(parent_hash)
                )
                manifest_path = parent_cache / "manifest.json"
                if manifest_path.is_file():
                    parent_manifest = json.loads(manifest_path.read_text())
                    cache_ok = (
                        parent_manifest.get("parent_checkpoint_sha256") == parent_hash
                        and parent_manifest.get("episodes") == 52
                        and parent_manifest.get("frames") == 31_706
                        and parent_manifest.get("exact_integer_tokens") is True
                    )
                else:
                    cache_ok = False
            selected_update = int(payload.get("selected_update", payload["update"]))
            record = {
                "exists": True,
                "update": int(payload["update"]),
                "selected_update": selected_update,
                "best_validation_action_mse": float(payload["best_validation"]),
                "architecture": payload["architecture"],
                "training_type": payload["training_type"],
                "representation": cfg["data"]["action_representation"],
                "observation_horizon": cfg["data"]["observation_horizon"],
                "vision_tokenizer": cfg["vision"]["tokenizer"],
                "vision_resampler_tokens_per_camera": cfg["vision"]["resampler_tokens_per_camera"],
                "full_mask_probability": cfg["policy"]["discrete_full_mask_probability"],
                "updates_exact": int(payload["update"]) == expected_updates,
                "identity_valid": (
                    payload["architecture"] == "discrete_joint"
                    and str(payload["training_type"]).lower() in expected_type
                    and cfg["data"]["action_representation"] == representation
                    and int(cfg["data"]["observation_horizon"]) == horizon
                    and cfg["vision"]["tokenizer"] == "dinov2"
                    and int(cfg["vision"]["resampler_tokens_per_camera"]) == 32
                    and float(cfg["policy"]["discrete_full_mask_probability"]) == 0.5
                ),
                "selection_from_full_validation": stage == "rtc" or selected_update >= 3_000,
                "finite_trace": _finite_trace(payload),
                "optimizer_steps_skipped": int(sum(float(row.get("optimizer_step_skipped", 0.0)) for row in payload.get("loss_trace", []))),
                "parent_exact": parent_ok,
                "parent_cache_exact": cache_ok,
                "parent_cache_manifest": parent_manifest,
                "wandb": payload.get("wandb"),
            }
            if args.remote_wandb:
                try:
                    record["remote_wandb"] = _remote_wandb(payload)
                except Exception as exc:
                    record["remote_wandb"] = {"finished": False, "error": str(exc)}
            required = (
                record["updates_exact"], record["identity_valid"],
                record["selection_from_full_validation"], record["finite_trace"],
                record["optimizer_steps_skipped"] == 0, record["parent_exact"],
                record["parent_cache_exact"], bool(record["wandb"]),
            )
            if args.remote_wandb:
                required += (record["remote_wandb"].get("finished") is True,)
            record["passed"] = all(required)
            if not record["passed"]:
                errors.append(f"checkpoint audit failed for {key}")
            checkpoint_checks[key] = record
    checks["checkpoints"] = checkpoint_checks

    summary_path = EXPERIMENT / "summary" / "evaluation_summary.json"
    deployment_path = EXPERIMENT / "summary" / "deployment" / "websocket_validation.json"
    evaluation = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    deployment = json.loads(deployment_path.read_text()) if deployment_path.is_file() else None
    expected_artifacts = []
    for variant in VARIANTS:
        for stage in STAGES:
            folder = EXPERIMENT / "summary" / variant / stage
            expected_artifacts.extend([
                folder / "open_loop_train.json", folder / "open_loop_test.json",
                folder / "rtc_train.json", folder / "rtc_test.json",
                folder / "rtc_inpainting_train.png", folder / "rtc_inpainting_test.png",
                EXPERIMENT / "summary" / "latency" / variant / f"{stage}.json",
            ])
    evaluation_ok = (
        evaluation is not None
        and evaluation.get("validation_passed") is True
        and len(evaluation.get("checks", {})) == 8
        and all(path.is_file() for path in expected_artifacts)
    )
    deployment_ok = (
        deployment is not None
        and deployment.get("passed") is True
        and deployment.get("server_count") == 8
    )
    checks["evaluation"] = {
        "passed": evaluation_ok,
        "expected_artifacts": len(expected_artifacts),
        "present_artifacts": sum(path.is_file() for path in expected_artifacts),
    }
    checks["deployment"] = {"passed": deployment_ok, "report": deployment}
    if not evaluation_ok:
        errors.append("full evaluation matrix is missing or invalid")
    if not deployment_ok:
        errors.append("WebSocket deployment matrix is missing or invalid")

    report = {
        "scope": "DINOv2 raw/B-spline x h1/h2 x base/ttRTC joint discrete diffusion",
        "passed": not errors,
        "errors": errors,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
