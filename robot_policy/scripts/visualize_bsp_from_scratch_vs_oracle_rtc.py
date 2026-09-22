#!/usr/bin/env python3
"""Compare scratch generation and oracle-prefix RTC on identical action chunks."""

from __future__ import annotations

import argparse
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
    select_full_horizon_indices,
)
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


DIMENSION_NAMES = (
    "joint 1",
    "joint 2",
    "joint 3",
    "joint 4",
    "joint 5",
    "joint 6",
    "gripper",
)


def _batch(dataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    batch = collate_policy_batch([dataset[index] for index in indices])
    return {key: value.to(device) for key, value in batch.items()}


def _physical_actions(
    controls: torch.Tensor,
    codec,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    normalized = codec.decode_controls(controls.float())
    return (normalized + 1.0) * 0.5 * (high - low) + low


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    mask = valid[..., None].expand_as(prediction)
    return float((prediction - target).square().masked_select(mask).mean().item())


def _mark_outside(
    axis,
    steps: np.ndarray,
    values: np.ndarray,
    minimum: float,
    maximum: float,
    color: str,
) -> None:
    below = values < minimum
    above = values > maximum
    if below.any():
        axis.scatter(
            steps[below],
            np.full(int(below.sum()), minimum),
            marker="v",
            s=15,
            color=color,
            zorder=6,
        )
    if above.any():
        axis.scatter(
            steps[above],
            np.full(int(above.sum()), maximum),
            marker="^",
            s=15,
            color=color,
            zorder=6,
        )


def visualize(
    config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    prefix: int = 6,
    samples: int = 5,
    seed: int = 20260921,
    device: str = "cuda",
) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    cfg = load_config(config, ["policy.architecture=bsp_unet_fm"])
    if not 0 < prefix < cfg.data.action_horizon:
        raise ValueError(f"prefix must be in 1..{cfg.data.action_horizon - 1}")
    torch_device = torch.device(device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, torch_device)
    model.eval()
    codec = create_action_codec(cfg, torch_device)
    dataset = create_policy_dataset(cfg, split)
    indices = select_full_horizon_indices(dataset, samples, seed)
    batch = _batch(dataset, indices, torch_device)

    stats = json.loads((Path(cfg.data.prepared_path) / "normalization.json").read_text())
    low = torch.as_tensor(stats["action_q01"], dtype=torch.float32, device=torch_device)
    high = torch.as_tensor(stats["action_q99"], dtype=torch.float32, device=torch_device)

    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        scratch_controls = model.sample(batch)

    prefix_values, fixed_mask, mapping = ground_truth_condition(
        batch, cfg, cfg.policy.architecture, prefix
    )
    # Reset to the same seed so the scratch and guided trajectories start from
    # identical Gaussian noise. Their difference then isolates prefix guidance.
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    training_type = str(payload.get("training_type", "base")).lower()
    if training_type == "base":
        rtc_controls = model.sample_realtime_pigdm(
            batch,
            prefix_values=prefix_values,
            fixed_mask=fixed_mask,
        )
        rtc_method = "base PiGDM binary prefix guidance"
    else:
        with torch.no_grad():
            rtc_controls = model.sample(
                batch,
                prefix_values=prefix_values,
                fixed_mask=fixed_mask,
            )
        rtc_method = (
            "ttRTC direct hard-prefix inpainting"
            if training_type in {"rtc", "ttrtc"}
            else "base representation-native hard-prefix inpainting"
        )

    scratch_physical = _physical_actions(scratch_controls, codec, low, high)
    rtc_physical = _physical_actions(rtc_controls, codec, low, high)
    target_physical = batch["target_trajectory"].float()
    valid = batch["action_valid_mask"].bool()
    suffix_valid = valid.clone()
    suffix_valid[:, :prefix] = False
    prefix_valid = valid.clone()
    prefix_valid[:, prefix:] = False

    scratch_mse = _masked_mse(scratch_physical, target_physical, valid)
    rtc_suffix_mse = _masked_mse(rtc_physical, target_physical, suffix_valid)
    rtc_prefix_mse = _masked_mse(rtc_physical, target_physical, prefix_valid)
    fixed_error = (rtc_controls - prefix_values).masked_select(fixed_mask)
    fixed_control_max_abs = float(fixed_error.abs().max().item())

    target = target_physical.cpu().numpy()
    scratch = scratch_physical.cpu().numpy()
    rtc = rtc_physical.cpu().numpy()
    episode_ids = batch["episode_id"].cpu().numpy()
    frames = batch["frame_index"].cpu().numpy()
    action_min, action_max = dataset_action_minmax(cfg.data.prepared_path)
    steps = np.arange(cfg.data.action_horizon)

    fig, axes = plt.subplots(
        len(indices),
        7,
        figsize=(22, max(3.15 * len(indices), 5.5)),
        sharex=True,
        squeeze=False,
    )
    for row in range(len(indices)):
        for dimension in range(7):
            axis = axes[row, dimension]
            axis.axvspan(-0.5, prefix - 0.5, color="#4c78a8", alpha=0.10)
            axis.plot(
                steps,
                target[row, :, dimension],
                color="#2ca02c",
                linestyle="--",
                linewidth=2.2,
                label="ground truth",
            )
            axis.plot(
                steps,
                scratch[row, :, dimension],
                color="#d62728",
                linewidth=1.45,
                alpha=0.90,
                label="from scratch",
            )
            axis.plot(
                steps,
                rtc[row, :, dimension],
                color="#1f77b4",
                linewidth=1.75,
                label="GT-prefix RTC",
            )
            axis.axvline(prefix - 0.5, color="0.25", linewidth=1.0, linestyle=":")
            axis.set_ylim(float(action_min[dimension]), float(action_max[dimension]))
            _mark_outside(
                axis,
                steps,
                scratch[row, :, dimension],
                float(action_min[dimension]),
                float(action_max[dimension]),
                "#d62728",
            )
            _mark_outside(
                axis,
                steps,
                rtc[row, :, dimension],
                float(action_min[dimension]),
                float(action_max[dimension]),
                "#1f77b4",
            )
            axis.grid(alpha=0.22)
            if row == 0:
                axis.set_title(DIMENSION_NAMES[dimension])
            if dimension == 0:
                axis.set_ylabel(f"ep {episode_ids[row]} frame {frames[row]}\naction")
            if row == len(indices) - 1:
                axis.set_xlabel("chunk step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    stage = "ttRTC" if training_type in {"rtc", "ttrtc"} else "base"
    title = (
        f"BSP-UNet FM {cfg.data.action_representation} h{cfg.data.observation_horizon} "
        f"{stage} [{split}] — scratch vs oracle-prefix RTC (prefix={prefix} raw actions)"
    )
    fig.suptitle(title, y=0.997, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.981),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.003,
        "Blue shading is the supplied ground-truth prefix; triangles mark values outside the dataset range.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.018, 1, 0.945))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "scratch_vs_ground_truth_prefix_rtc.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "trajectory_examples.npz",
        target_physical=target,
        scratch_physical=scratch,
        rtc_physical=rtc,
        episode_id=episode_ids,
        frame_index=frames,
        dataset_index=np.asarray(indices, dtype=np.int64),
        action_min=action_min,
        action_max=action_max,
    )
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_training_type": str(payload.get("training_type")),
        "split": split,
        "seed": seed,
        "paired_initial_noise": True,
        "sample_selection": "deterministic episode-balanced complete 30-step chunks",
        "samples": len(indices),
        "dataset_indices": indices,
        "episode_ids": episode_ids.tolist(),
        "frame_indices": frames.tolist(),
        "action_representation": cfg.data.action_representation,
        "observation_horizon": cfg.data.observation_horizon,
        "scratch_generation": "12-step Euler flow integration from Gaussian noise",
        "rtc_generation": rtc_method,
        "pigdm": (
            {
                "enabled": True,
                "conditioning_operator": "binary representation-native prefix mask",
                "max_guidance_weight": 5.0,
                "fm_steps": cfg.policy.fm_steps,
                "prefix_is_exactly_clamped": False,
            }
            if training_type == "base"
            else {"enabled": False}
        ),
        "ground_truth_prefix_raw_actions": prefix,
        "ground_truth_prefix_mapping": mapping,
        "scratch_full_chunk_physical_mse": scratch_mse,
        "rtc_suffix_physical_mse": rtc_suffix_mse,
        "rtc_prefix_physical_mse": rtc_prefix_mse,
        "fixed_control_max_abs": fixed_control_max_abs,
        "plot": str(plot_path.resolve()),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = visualize(
        args.config,
        args.checkpoint,
        args.output_dir,
        split=args.split,
        prefix=args.prefix,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
