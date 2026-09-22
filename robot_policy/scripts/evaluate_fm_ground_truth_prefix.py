#!/usr/bin/env python3
"""Evaluate base Flow Matching with a ground-truth prefix over a full split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from robot_policy.config import load_config
from robot_policy.data.dataset import collate_policy_batch, create_policy_dataset
from robot_policy.evaluation.inference_rtc import ground_truth_condition
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


class ErrorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_square = 0.0
        self.max_abs = 0.0

    def update(self, error: torch.Tensor, mask: torch.Tensor) -> None:
        selected = error.masked_select(mask).float()
        if not selected.numel():
            return
        self.count += selected.numel()
        self.sum_abs += float(selected.abs().sum().item())
        self.sum_square += float(selected.square().sum().item())
        self.max_abs = max(self.max_abs, float(selected.abs().max().item()))

    def summary(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("no valid values were accumulated")
        mse = self.sum_square / self.count
        return {
            "values": self.count,
            "mae": self.sum_abs / self.count,
            "mse": mse,
            "rmse": mse**0.5,
            "max_abs": self.max_abs,
        }


def evaluate(
    config: str | Path,
    checkpoint: str | Path,
    split: str,
    prefix: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    cfg = load_config(config, ["policy.architecture=bsp_unet_fm"])
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if not 0 < prefix < cfg.data.action_horizon:
        raise ValueError(f"prefix must be in 1..{cfg.data.action_horizon - 1}")

    torch_device = torch.device(device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, torch_device)
    model.eval()
    training_type = str(payload.get("training_type", "base")).lower()
    if training_type != "base":
        raise ValueError(
            "this evaluator is for base-FM PiGDM checkpoints; "
            f"checkpoint training_type={training_type!r}"
        )
    codec = create_action_codec(cfg, torch_device)
    dataset = create_policy_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_policy_batch,
    )
    normalization = json.loads(
        (Path(cfg.data.prepared_path) / "normalization.json").read_text()
    )
    low = torch.as_tensor(
        normalization["action_q01"], dtype=torch.float32, device=torch_device
    )
    high = torch.as_tensor(
        normalization["action_q99"], dtype=torch.float32, device=torch_device
    )

    whole = ErrorAccumulator()
    suffix = ErrorAccumulator()
    committed_prefix = ErrorAccumulator()
    unstitched_pigdm_prefix = ErrorAccumulator()
    fixed_control_max_abs = 0.0
    mapping: dict[str, Any] | None = None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    for cpu_batch in loader:
        batch = {
            key: value.to(torch_device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        prefix_values, fixed_mask, mapping = ground_truth_condition(
            batch, cfg, cfg.policy.architecture, prefix
        )
        prediction = model.sample_realtime_pigdm(
            batch,
            prefix_values=prefix_values,
            fixed_mask=fixed_mask,
        )
        normalized = codec.decode_controls(prediction.float())
        pigdm_physical = (normalized + 1.0) * 0.5 * (high - low) + low
        target_physical = batch["target_trajectory"].float()
        # The prefix is already committed at RTC time.  The deployable output
        # is exact GT/observed prefix + newly generated suffix; the raw PiGDM
        # prefix residual is retained only as a sampler diagnostic.
        physical = pigdm_physical.clone()
        physical[:, :prefix] = target_physical[:, :prefix]
        if not torch.equal(physical[:, :prefix], target_physical[:, :prefix]):
            raise AssertionError("deployed RTC prefix must exactly equal the supplied GT prefix")
        error = physical - target_physical
        unstitched_error = pigdm_physical - target_physical
        valid = batch["action_valid_mask"].bool()
        expanded = valid[..., None].expand_as(error)
        suffix_valid = valid.clone()
        suffix_valid[:, :prefix] = False
        prefix_valid = valid.clone()
        prefix_valid[:, prefix:] = False
        whole.update(error, expanded)
        suffix.update(error, suffix_valid[..., None].expand_as(error))
        committed_prefix.update(error, prefix_valid[..., None].expand_as(error))
        unstitched_pigdm_prefix.update(
            unstitched_error, prefix_valid[..., None].expand_as(error)
        )
        fixed_error = (prediction.float() - prefix_values.float()).masked_select(fixed_mask)
        if fixed_error.numel():
            fixed_control_max_abs = max(
                fixed_control_max_abs, float(fixed_error.abs().max().item())
            )
    torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    assert mapping is not None
    return {
        "schema_version": 1,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_training_type": training_type,
        "architecture": cfg.policy.architecture,
        "action_representation": cfg.data.action_representation,
        "split": split,
        "samples": len(dataset),
        "coverage": "complete split; sequential DataLoader; padded action rows excluded",
        "condition_source": "ground-truth prefix from the current action chunk",
        "rtc_method": "base-FM PiGDM with a binary representation-native hard mask",
        "raw_prefix_actions": prefix,
        "ground_truth_prefix_mapping": mapping,
        "primary_metric": "generated_suffix_physical_error.mse",
        "generated_suffix_physical_error": suffix.summary(),
        "whole_chunk_physical_error": whole.summary(),
        "committed_prefix_physical_error": committed_prefix.summary(),
        "unstitched_pigdm_prefix_diagnostic": unstitched_pigdm_prefix.summary(),
        "deployed_trajectory_contract": (
            "exact supplied GT raw-action prefix concatenated with the decoded "
            "PiGDM-generated suffix"
        ),
        "fixed_control_max_abs": fixed_control_max_abs,
        "batch_size": batch_size,
        "seed": seed,
        "fm_steps": cfg.policy.fm_steps,
        "elapsed_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / len(dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.prefix,
        args.batch_size,
        args.device,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
