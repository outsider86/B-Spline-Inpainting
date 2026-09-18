#!/usr/bin/env python3
"""Requirement-level audit for all or a capacity subset of the full-vision sweep."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_policy.config import ACTIVE_ARCHITECTURES, load_config


SIZES = {
    "dit_s": ("DiT-S", 384, 6, 4),
    "dit_b": ("DiT-B", 768, 12, 12),
}
REPRESENTATIONS = ("raw", "bspline")
STAGES = ("base", "ttrtc")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/FULL_VISION_512"))
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--check-wandb", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sizes = tuple(item.strip() for item in args.sizes.split(",") if item.strip())
    unknown_sizes = sorted(set(sizes) - set(SIZES))
    if not sizes or unknown_sizes:
        parser.error(f"--sizes must contain one or more of {tuple(SIZES)}; unknown={unknown_sizes}")
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    api = None
    if args.check_wandb:
        import wandb
        api = wandb.Api()

    vision_files = sorted((root / "cache" / "vision").glob("episode_*.npy"))
    vision_frames = 0
    for path in vision_files:
        array = np.load(path, mmap_mode="r")
        if array.shape[1:] != (2, 256, 2176) or array.dtype != np.float16:
            errors.append(f"invalid vision layout {path}: {array.shape} {array.dtype}")
        vision_frames += int(array.shape[0])
    if len(vision_files) != 52 or vision_frames != 31_706:
        errors.append(f"vision inventory mismatch: {len(vision_files)} episodes, {vision_frames} frames")

    expected_paths: set[Path] = set()
    for size in sizes:
        model_size, width, depth, heads = SIZES[size]
        for representation in REPRESENTATIONS:
            config_path = Path("configs/full_vision_512") / f"{size}_{representation}.yaml"
            cfg = load_config(config_path)
            if cfg.vision.pooled_grid != 16:
                errors.append(f"{config_path} does not configure 256 tokens/camera")
            manifest_path = root / size / representation / "checkpoints" / "checkpoint_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"checkpoints": []}
            manifest_index = {
                (entry["architecture"], entry["training_type"]): entry
                for entry in manifest.get("checkpoints", [])
            }
            for architecture in ACTIVE_ARCHITECTURES:
                for stage in STAGES:
                    path = root / size / representation / "checkpoints" / f"{architecture}_{stage}.pt"
                    expected_paths.add(path)
                    record: dict[str, Any] = {
                        "model_size": model_size,
                        "representation": representation,
                        "architecture": architecture,
                        "stage": stage,
                        "path": str(path.resolve()),
                        "checks": {},
                    }
                    records.append(record)
                    if not path.exists():
                        errors.append(f"missing {path}")
                        continue
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    config = payload.get("config", {})
                    policy = config.get("policy", {})
                    vision = config.get("vision", {})
                    train = config.get("train", {})
                    expected_update = cfg.train.updates if stage == "base" else cfg.train.rtc_updates
                    expected_type = "base" if stage == "base" else ("ttrtc" if representation == "raw" else "rtc")
                    history = payload.get("history", [])
                    validation_ok = bool(history) and all(
                        row.get("validation", {}).get("generation_action_mse")
                        == row.get("validation", {}).get("action_mse")
                        == row.get("validation", {}).get("loss")
                        for row in history
                    )
                    gradients = [row.get("grad_norm") for row in payload.get("loss_trace", [])]
                    gradients_ok = bool(gradients) and all(
                        value is not None and math.isfinite(float(value)) for value in gradients
                    )
                    threshold = float(train.get(
                        "fm_grad_skip_threshold" if stage == "base" else "fm_rtc_grad_skip_threshold",
                        3.0 if stage == "base" else 100.0,
                    ))
                    guard_after = (
                        int(train.get("fm_grad_skip_after_updates", 1000))
                        if stage == "base"
                        else 0
                    )
                    optimizer_guard_ok = all(
                        not (
                            architecture == "fm"
                            and int(row.get("update", 0)) > guard_after
                            and float(row.get("grad_norm", float("inf"))) > threshold
                        )
                        or float(row.get("optimizer_step_skipped", 0.0)) == 1.0
                        for row in payload.get("loss_trace", [])
                    )
                    digest = file_sha256(path)
                    entry = manifest_index.get((architecture, expected_type))
                    checks = record["checks"]
                    wandb_info = payload.get("wandb") or {}
                    checks.update({
                        "identity": payload.get("architecture") == architecture
                        and payload.get("training_type") == expected_type,
                        "update": payload.get("update") == expected_update,
                        "model_shape": policy.get("model_size") == model_size
                        and (policy.get("hidden_dim"), policy.get("depth"), policy.get("heads"))
                        == (width, depth, heads),
                        "vision_tokens": vision.get("pooled_grid") == 16,
                        "effective_batch_32": train.get("effective_batch_size") == 32,
                        "rtc_learning_rate": (
                            stage == "base"
                            or float(train.get("rtc_learning_rate", float("nan")))
                            == float(cfg.train.rtc_learning_rate)
                        ),
                        "fm_math_sdp_training": (
                            architecture != "fm"
                            or stage == "base"
                            or train.get("fm_math_sdp_training") is True
                        ),
                        "generation_validation": validation_ok,
                        "finite_gradient_trace": gradients_ok,
                        "optimizer_spike_guard": optimizer_guard_ok,
                        "manifest_hash": entry is not None and entry.get("sha256") == digest,
                        "wandb_online": wandb_info.get("mode") == "online"
                        and bool(wandb_info.get("url")),
                    })
                    if stage == "ttrtc":
                        parent = root / size / representation / "checkpoints" / f"{architecture}_base.pt"
                        checks["exact_parent"] = (
                            Path(payload.get("parent_checkpoint", "")).resolve() == parent.resolve()
                        )
                    checkpoint_path = path.resolve()
                    open_loop = read_json(
                        root / "summary" / "open_loop" / size / representation
                        / f"{architecture}_{stage}.json"
                    )
                    checks["open_loop_evaluation"] = bool(
                        open_loop
                        and Path(open_loop.get("checkpoint", "")).resolve() == checkpoint_path
                        and open_loop.get("architecture") == architecture
                        and open_loop.get("action_representation") == representation
                        and open_loop.get("checkpoint_training_type") == expected_type
                        and open_loop.get("split") == "test"
                        and int(open_loop.get("samples", 0)) > 0
                        and "not closed-loop robot success" in open_loop.get("scope", "")
                    )
                    latency = read_json(
                        root / "summary" / "latency" / size / representation
                        / f"{architecture}_{stage}.json"
                    )
                    checks["latency_evaluation"] = bool(
                        latency
                        and Path(latency.get("checkpoint", "")).resolve() == checkpoint_path
                        and latency.get("architecture") == architecture
                        and latency.get("action_representation") == representation
                        and latency.get("training_type") == expected_type
                        and latency.get("batch_size") == 1
                        and int(latency.get("warmup", 0)) > 0
                        and int(latency.get("iterations", 0)) > 0
                    )
                    rtc = read_json(
                        root / "summary" / "rtc" / size / representation
                        / f"{architecture}_{stage}.json"
                    )
                    checks["delay_rtc_evaluation"] = bool(
                        rtc
                        and Path(rtc.get("checkpoint", "")).resolve() == checkpoint_path
                        and rtc.get("architecture") == architecture
                        and rtc.get("action_representation") == representation
                        and rtc.get("training_type") == expected_type
                        and rtc.get("split") == "test"
                        and int(rtc.get("samples_per_delay", 0)) == 128
                        and bool(rtc.get("curves"))
                    )
                    inference_rtc = read_json(
                        root / "summary" / "inference_rtc" / size
                        / f"{representation}_{architecture}_{stage}" / "report.json"
                    )
                    inference_curves = inference_rtc.get("curves", []) if inference_rtc else []
                    metric_axis = inference_rtc.get("metric_plot_axis_limits", {}) if inference_rtc else {}
                    trajectory_axis = inference_rtc.get("trajectory_axis_limits", {}) if inference_rtc else {}
                    checks["inference_rtc_evaluation"] = bool(
                        inference_rtc
                        and Path(inference_rtc.get("checkpoint", "")).resolve() == checkpoint_path
                        and inference_rtc.get("model_size") == model_size
                        and inference_rtc.get("vision_tokens") == 512
                        and inference_rtc.get("architecture") == architecture
                        and inference_rtc.get("action_representation") == representation
                        and inference_rtc.get("checkpoint_training_type") == expected_type
                        and inference_rtc.get("split") == "test"
                        and int(inference_rtc.get("samples", 0)) == 64
                        and inference_rtc.get("inference_mode")
                        == "representation-native fixed-prefix inpainting from scratch"
                        and inference_rtc.get("prefix_source")
                        == "current ground-truth action chunk (oracle prefix)"
                        and {curve.get("raw_prefix_actions") for curve in inference_curves}
                        == {2, 4, 6, 8, 10}
                        and all(float(curve.get("fixed_control_max_abs", float("inf"))) <= 1e-6
                                for curve in inference_curves)
                    )
                    checks["aligned_rtc_plot_axes"] = bool(
                        metric_axis.get("yscale") == "log"
                        and float(metric_axis.get("suffix_physical_mse_min", 0.0)) > 0.0
                        and float(metric_axis.get("suffix_physical_mse_max", 0.0))
                        > float(metric_axis.get("suffix_physical_mse_min", float("inf")))
                        and trajectory_axis.get("source")
                        == "true per-channel physical min/max across every prepared dataset episode"
                        and len(trajectory_axis.get("min", [])) == 7
                        and len(trajectory_axis.get("max", [])) == 7
                    )
                    if api is not None and wandb_info:
                        info = wandb_info
                        run = api.run(f"{info['entity']}/{info['project']}/{info['run_id']}")
                        checks["wandb_remote_finished"] = run.state == "finished"
                        checks["wandb_generation_metric"] = run.summary.get("final_validation_action_mse") is not None
                    failed = [name for name, passed in checks.items() if not passed]
                    if failed:
                        errors.append(f"{path}: failed {failed}")
                    record.update({
                        "sha256": digest,
                        "parameters": payload.get("parameter_counts", {}).get("total_trainable"),
                        "final_train_action_mse": payload.get("loss_trace", [{}])[-1].get("action_mse"),
                        "final_validation_action_mse": history[-1]["validation"]["action_mse"] if history else None,
                        "best_validation_action_mse": payload.get("best_validation"),
                        "max_gradient_norm": max(map(float, gradients)) if gradients_ok else None,
                        "skipped_optimizer_updates": sum(
                            int(float(row.get("optimizer_step_skipped", 0.0)))
                            for row in payload.get("loss_trace", [])
                        ),
                        "wandb": payload.get("wandb"),
                    })

    actual_paths = {
        path
        for size in sizes
        for path in (root / size).glob("*/checkpoints/*.pt")
    }
    extras = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if extras:
        errors.append(f"unexpected checkpoints: {[str(path) for path in extras]}")
    if missing:
        errors.append(f"missing checkpoint paths: {[str(path) for path in missing]}")
    for size in sizes:
        metric_contracts: set[str] = set()
        trajectory_contracts: set[str] = set()
        for representation in REPRESENTATIONS:
            for architecture in ACTIVE_ARCHITECTURES:
                for stage in STAGES:
                    report = read_json(
                        root / "summary" / "inference_rtc" / size
                        / f"{representation}_{architecture}_{stage}" / "report.json"
                    )
                    if report is None:
                        continue
                    metric_contracts.add(json.dumps(report.get("metric_plot_axis_limits"), sort_keys=True))
                    trajectory = report.get("trajectory_axis_limits", {})
                    trajectory_contracts.add(json.dumps({
                        "source": trajectory.get("source"),
                        "min": trajectory.get("min"),
                        "max": trajectory.get("max"),
                    }, sort_keys=True))
        if len(metric_contracts) > 1:
            errors.append(f"{size} RTCEVAL metric plots do not share one y-axis contract")
        if len(trajectory_contracts) > 1:
            errors.append(f"{size} RTCEVAL trajectory plots do not share one physical-axis contract")
    result = {
        "status": "passed" if not errors else "failed",
        "checkpoint_count": len(actual_paths),
        "expected_checkpoint_count": len(sizes) * len(REPRESENTATIONS) * len(ACTIVE_ARCHITECTURES) * len(STAGES),
        "sizes": list(sizes),
        "active_architectures": list(ACTIVE_ARCHITECTURES),
        "vision": {
            "episodes": len(vision_files),
            "frames": vision_frames,
            "tokens_per_camera": 256,
            "camera_count": 2,
            "vision_tokens": 512,
            "bytes": sum(path.stat().st_size for path in vision_files),
        },
        "validation_contract": "unconditional generation from noise/MASK using vision+state only",
        "records": records,
        "errors": errors,
    }
    suffix = "" if sizes == tuple(SIZES) else "_" + "_".join(sizes)
    output = root / f"completion_audit{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_count", "expected_checkpoint_count", "vision", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
