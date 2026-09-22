#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_policy.config import ACTIVE_ARCHITECTURES
from robot_policy.deployment.checkpoint import inspect_checkpoint
from robot_policy.deployment.policy_wrapper import PolicyServerWrapper
from robot_policy.encoders.vision import FrozenDinoSigLIP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=Path("outputs/SWEEP"))
    parser.add_argument("--model-size", choices=("dit_s", "dit_b"), default="dit_s")
    parser.add_argument(
        "--inventory-sizes",
        default="dit_s,dit_b",
        help="comma-separated checkpoint families included in the inventory gate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=16)
    args = parser.parse_args()
    root = args.sweep_root.resolve()
    device = torch.device(args.device)
    inventory_sizes = tuple(
        item.strip() for item in args.inventory_sizes.split(",") if item.strip()
    )
    allowed_sizes = {"dit_s", "dit_b"}
    unknown_sizes = sorted(set(inventory_sizes) - allowed_sizes)
    if not inventory_sizes or unknown_sizes:
        parser.error(f"invalid --inventory-sizes: {unknown_sizes}")

    checkpoints = sorted(
        checkpoint
        for size in inventory_sizes
        for checkpoint in (root / size).glob("*/checkpoints/*.pt")
    )
    if len(checkpoints) != args.expected_checkpoints:
        raise RuntimeError(f"expected {args.expected_checkpoints} checkpoints, found {len(checkpoints)}")
    inventory = []
    for checkpoint in checkpoints:
        metadata = inspect_checkpoint(checkpoint)
        inventory.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "architecture": metadata.architecture,
                "training_type": metadata.training_type,
                "is_rtc": metadata.is_rtc,
                "model_size": metadata.config.policy.model_size,
                "representation": metadata.config.data.action_representation,
                "prepared_path": str(metadata.prepared_path),
            }
        )

    first_metadata = inspect_checkpoint(
        root / args.model_size / "bspline" / "checkpoints" / "fm_base.pt"
    )
    vision = FrozenDinoSigLIP(first_metadata.config).to(device).eval()
    blank = torch.zeros(2, 224, 224, 3, dtype=torch.uint8, device=device)
    started = time.perf_counter()
    encoded = vision(blank).fused_patches
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    vision_ms = (time.perf_counter() - started) * 1000
    features = encoded.float().cpu().numpy()
    tokens_per_camera = first_metadata.config.vision.pooled_grid**2
    example = {
        "vision_features": features.reshape(
            2, tokens_per_camera, first_metadata.config.vision.feature_dim
        ),
        "state": np.zeros((1, 7), dtype=np.float32),
        "lang": "Stack the cups.",
    }

    runtime = []
    for representation in ("raw", "bspline"):
        for architecture in ACTIVE_ARCHITECTURES:
            for stage in ("base", "ttrtc"):
                checkpoint = (
                    root
                    / args.model_size
                    / representation
                    / "checkpoints"
                    / f"{architecture}_{stage}.pt"
                )
                load_started = time.perf_counter()
                wrapper = PolicyServerWrapper(
                    checkpoint,
                    device=device,
                    precision="bf16",
                    binary_gripper=False,
                    vision_encoder=vision,
                )
                load_seconds = time.perf_counter() - load_started
                initial = wrapper.predict_action([example], seed=20260915)
                record = {
                    "checkpoint": str(checkpoint),
                    "architecture": architecture,
                    "representation": representation,
                    "stage": stage,
                    "load_seconds": load_seconds,
                    "actions_shape": list(initial["actions"].shape),
                    "actions_finite": bool(np.isfinite(initial["actions"]).all()),
                    "sampling_ms": float(initial["sampling_ms"]),
                    "metadata_supports_raw_rtc": wrapper.metadata["supports_inference_time_rtc"],
                    "metadata_supports_parameter_row_rtc": wrapper.metadata["supports_parameter_row_rtc"],
                }
                if representation == "bspline":
                    record["control_rows_shape"] = list(
                        initial["normalized_control_rows"].shape
                    )
                if stage == "ttrtc":
                    previous = (
                        initial["actions"]
                        if representation == "raw"
                        else initial["normalized_control_rows"]
                    )
                    realtime = wrapper.predict_action_realtime(
                        [example],
                        inference_delay=3,
                        prev_action_chunk=previous if representation == "raw" else None,
                        prev_control_rows=previous if representation == "bspline" else None,
                        seed=20260915,
                    )
                    record.update(
                        {
                            "realtime_actions_shape": list(realtime["actions"].shape),
                            "realtime_actions_finite": bool(
                                np.isfinite(realtime["actions"]).all()
                            ),
                            "realtime_sampling_ms": float(realtime["sampling_ms"]),
                            "realtime_fixed_control_rows": realtime["inference_metadata"][
                                "fixed_control_rows"
                            ],
                        }
                    )
                if record["actions_shape"] != [1, 30, 7] or not record["actions_finite"]:
                    raise RuntimeError(f"invalid runtime result: {record}")
                runtime.append(record)
                del wrapper
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    report = {
        "scope": "all checkpoint headers plus real GPU runtime matrix for one active model-size family",
        "sweep_root": str(root),
        "checkpoint_inventory_count": len(inventory),
        "inventory_sizes": list(inventory_sizes),
        "inventory": inventory,
        "runtime_model_size": args.model_size,
        "vision_encoder_output_shape": list(encoded.shape),
        "vision_encoder_finite": bool(torch.isfinite(encoded).all().item()),
        "vision_encoder_blank_pair_ms": vision_ms,
        "runtime_case_count": len(runtime),
        "runtime": runtime,
        "passed": len(inventory) == args.expected_checkpoints and len(runtime) == 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("checkpoint_inventory_count", "runtime_model_size", "runtime_case_count", "passed")}, indent=2))


if __name__ == "__main__":
    main()
