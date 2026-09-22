#!/usr/bin/env python3
"""Make one train/val figure comparing raw and B-spline FM generations."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_policy.config import load_config
from robot_policy.data.dataset import collate_policy_batch, create_policy_dataset
from robot_policy.evaluation.inference_rtc import (
    dataset_action_minmax,
    ground_truth_condition,
)
from robot_policy.evaluation.rtc import select_rtc_indices
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


DIMENSIONS = ("joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "gripper")
POLICIES = ("raw", "B-spline")


def _to_device(dataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    batch = collate_policy_batch([dataset[index] for index in indices])
    return {key: value.to(device) for key, value in batch.items()}


def _physical(controls, codec, low, high):
    normalized = codec.decode_controls(controls.float())
    return (normalized + 1.0) * 0.5 * (high - low) + low


def _generate(
    config: str | Path,
    checkpoint: str | Path,
    split: str,
    pairs: list[tuple[int, int]],
    prefix: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cfg = load_config(config, ["policy.architecture=bsp_unet_fm"])
    dataset = create_policy_dataset(cfg, split)
    lookup = {pair: index for index, pair in enumerate(dataset.index)}
    missing = [pair for pair in pairs if pair not in lookup]
    if missing:
        raise RuntimeError(f"selected examples are missing in {cfg.data.action_representation}: {missing}")
    indices = [lookup[pair] for pair in pairs]
    batch = _to_device(dataset, indices, device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, device)
    model.eval()
    if str(payload.get("training_type", "base")).lower() != "base":
        raise ValueError("comparison requires base-FM checkpoints so RTC uses PiGDM")
    codec = create_action_codec(cfg, device)
    normalization = json.loads(
        (Path(cfg.data.prepared_path) / "normalization.json").read_text()
    )
    low = torch.as_tensor(normalization["action_q01"], dtype=torch.float32, device=device)
    high = torch.as_tensor(normalization["action_q99"], dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        scratch_controls = model.sample(batch)
    prefix_values, fixed_mask, mapping = ground_truth_condition(
        batch, cfg, cfg.policy.architecture, prefix
    )
    # Pair the starting random draw within each representation.  Raw and
    # B-spline have different latent shapes, so their random tensors cannot be
    # elementwise identical, but both use the same documented seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rtc_controls = model.sample_realtime_pigdm(
        batch,
        prefix_values=prefix_values,
        fixed_mask=fixed_mask,
    )
    ground_truth = batch["target_trajectory"].float()
    rtc_decoded = _physical(rtc_controls, codec, low, high)
    # RTC does not execute a newly decoded prediction for already committed
    # actions.  The deployed chunk is the supplied GT prefix followed by the
    # newly generated suffix.  This is especially important for B-splines:
    # PiGDM is conditioned in control-point space, while the committed raw
    # action prefix remains the exact observed/executed trajectory.
    rtc_deployed = rtc_decoded.clone()
    rtc_deployed[:, :prefix] = ground_truth[:, :prefix]
    if not torch.equal(rtc_deployed[:, :prefix], ground_truth[:, :prefix]):
        raise AssertionError("displayed RTC prefix must exactly equal the supplied GT prefix")
    arrays = {
        "ground_truth": ground_truth.cpu().numpy(),
        "from_scratch": _physical(scratch_controls, codec, low, high).cpu().numpy(),
        "gt_prefix_rtc": rtc_deployed.cpu().numpy(),
        "pigdm_decoded_unstitched": rtc_decoded.cpu().numpy(),
        "condition_values": prefix_values.float().cpu().numpy(),
        "condition_mask": fixed_mask.cpu().numpy(),
    }
    metadata = {
        "representation": cfg.data.action_representation,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_training_type": str(payload.get("training_type")),
        "ground_truth_prefix_mapping": mapping,
        "condition_values": (
            "first six ground-truth raw actions"
            if cfg.data.action_representation == "raw"
            else "ground-truth B-spline control points on affected-span cubic support"
        ),
        "displayed_rtc_trajectory": (
            "exact supplied ground-truth raw-action prefix concatenated with "
            "the PiGDM-generated decoded suffix"
        ),
        "displayed_prefix_exact_match": True,
    }
    del model, codec, batch, scratch_controls, rtc_controls
    gc.collect()
    torch.cuda.empty_cache()
    return arrays, metadata


def visualize(
    raw_config: str | Path,
    raw_checkpoint: str | Path,
    bspline_config: str | Path,
    bspline_checkpoint: str | Path,
    split: str,
    output: str | Path,
    *,
    prefix: int = 6,
    samples: int = 3,
    seed: int = 20260915,
    device: str = "cuda",
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    raw_cfg = load_config(raw_config, ["policy.architecture=bsp_unet_fm"])
    raw_dataset = create_policy_dataset(raw_cfg, split)
    selected = select_rtc_indices(
        raw_dataset,
        samples,
        0,
        strategy="episode_balanced_motion",
        seed=seed,
    )
    pairs = [raw_dataset.index[index] for index in selected]
    torch_device = torch.device(device)
    raw, raw_metadata = _generate(
        raw_config, raw_checkpoint, split, pairs, prefix, seed, torch_device
    )
    bspline, bspline_metadata = _generate(
        bspline_config, bspline_checkpoint, split, pairs, prefix, seed, torch_device
    )
    action_min, action_max = dataset_action_minmax(raw_cfg.data.prepared_path)

    rows = samples * 2
    fig, axes = plt.subplots(
        rows,
        7,
        figsize=(23, max(2.65 * rows, 8.0)),
        sharex=True,
        squeeze=False,
    )
    steps = np.arange(raw_cfg.data.action_horizon)
    colors = {"ground_truth": "#2ca02c", "from_scratch": "#d62728", "gt_prefix_rtc": "#1f77b4"}
    labels = {"ground_truth": "ground truth", "from_scratch": "from-scratch prediction", "gt_prefix_rtc": "GT-prefix RTC prediction"}
    styles = {"ground_truth": "--", "from_scratch": "-", "gt_prefix_rtc": "-"}
    widths = {"ground_truth": 2.2, "from_scratch": 1.45, "gt_prefix_rtc": 1.75}
    for sample_index, (episode, frame) in enumerate(pairs):
        for policy_index, (policy_name, artifact) in enumerate(zip(POLICIES, (raw, bspline))):
            row = sample_index * 2 + policy_index
            for dimension in range(7):
                axis = axes[row, dimension]
                axis.axvspan(-0.5, prefix - 0.5, color="#4c78a8", alpha=0.08)
                for key in ("ground_truth", "from_scratch", "gt_prefix_rtc"):
                    axis.plot(
                        steps,
                        artifact[key][sample_index, :, dimension],
                        color=colors[key],
                        linestyle=styles[key],
                        linewidth=widths[key],
                        label=labels[key],
                    )
                axis.axvline(prefix - 0.5, color="0.35", linestyle=":", linewidth=0.9)
                axis.set_ylim(float(action_min[dimension]), float(action_max[dimension]))
                axis.grid(alpha=0.22)
                if row == 0:
                    axis.set_title(DIMENSIONS[dimension])
                if dimension == 0:
                    axis.set_ylabel(f"{policy_name}\nep {episode}, frame {frame}")
                if row == rows - 1:
                    axis.set_xlabel("chunk step")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"FM raw vs B-spline [{split}] — from scratch and GT-prefix RTC (prefix={prefix})",
        y=0.997,
        fontsize=15,
    )
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.981),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.003,
        "Blue shading marks the supplied GT prefix. Every column uses the full-dataset physical min/max for that action dimension.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.018, 1, 0.945))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175)
    plt.close(fig)

    np.savez_compressed(
        output.with_suffix(".npz"),
        episode_id=np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        frame_index=np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        action_min=action_min,
        action_max=action_max,
        raw_ground_truth=raw["ground_truth"],
        raw_from_scratch=raw["from_scratch"],
        raw_gt_prefix_rtc=raw["gt_prefix_rtc"],
        raw_pigdm_decoded_unstitched=raw["pigdm_decoded_unstitched"],
        raw_condition_values=raw["condition_values"],
        raw_condition_mask=raw["condition_mask"],
        bspline_ground_truth=bspline["ground_truth"],
        bspline_from_scratch=bspline["from_scratch"],
        bspline_gt_prefix_rtc=bspline["gt_prefix_rtc"],
        bspline_pigdm_decoded_unstitched=bspline["pigdm_decoded_unstitched"],
        bspline_condition_values=bspline["condition_values"],
        bspline_condition_mask=bspline["condition_mask"],
    )
    report = {
        "schema_version": 1,
        "split": split,
        "samples": samples,
        "selection": "deterministic episode-balanced highest-motion complete chunks",
        "seed": seed,
        "episode_frame_pairs": [list(pair) for pair in pairs],
        "raw_prefix_actions": prefix,
        "rtc_display_contract": (
            "GT prefix is committed and copied exactly; only the suffix is generated"
        ),
        "lines_per_subplot": [
            "ground truth",
            "from-scratch prediction",
            "GT-prefix RTC prediction",
        ],
        "axis_limits": {
            "source": "physical min/max across the complete dataset",
            "minimum": action_min.tolist(),
            "maximum": action_max.tolist(),
        },
        "raw": raw_metadata,
        "bspline": bspline_metadata,
        "plot": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-config", required=True)
    parser.add_argument("--raw-checkpoint", required=True)
    parser.add_argument("--bspline-config", required=True)
    parser.add_argument("--bspline-checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = visualize(
        args.raw_config,
        args.raw_checkpoint,
        args.bspline_config,
        args.bspline_checkpoint,
        args.split,
        args.output,
        prefix=args.prefix,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
