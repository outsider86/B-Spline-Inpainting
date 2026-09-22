#!/usr/bin/env python3
"""Plot real RTC inpainting from a previous generated action chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_policy.config import TRAINABLE_ARCHITECTURES, is_continuous_architecture, load_config
from robot_policy.data.dataset import collate_policy_batch, create_policy_dataset
from robot_policy.evaluation.inference_rtc import dataset_action_minmax
from robot_policy.evaluation.rtc import select_rtc_indices
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


DIMENSION_NAMES = ("joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "gripper")


def _select(dataset, delay: int, count: int) -> list[int]:
    selected = select_rtc_indices(
        dataset,
        count,
        delay,
        strategy="episode_balanced_motion",
        seed=20260915,
    )
    if not selected:
        raise RuntimeError("no full-horizon RTC visualization samples are available")
    return selected


def _batch(dataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    batch = collate_policy_batch([dataset[index] for index in indices])
    return {key: value.to(device) for key, value in batch.items()}


def _mark_outside(
    axis,
    steps: np.ndarray,
    values: np.ndarray,
    minimum: float,
    maximum: float,
) -> None:
    below = values < minimum
    above = values > maximum
    if below.any():
        axis.scatter(steps[below], np.full(int(below.sum()), minimum), marker="v", s=18, color="#d62728", zorder=6)
    if above.any():
        axis.scatter(steps[above], np.full(int(above.sum()), maximum), marker="^", s=18, color="#d62728", zorder=6)


def visualize(
    config: str | Path,
    checkpoint: str | Path,
    split: str,
    output: str | Path,
    *,
    architecture: str = "bsp_unet_fm",
    delay: int = 6,
    samples: int = 5,
    seed: int = 20260921,
    device: str = "cuda",
    condition_source: str = "previous_plan",
) -> None:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if condition_source not in {"ground_truth", "previous_plan"}:
        raise ValueError("condition_source must be ground_truth or previous_plan")
    cfg = load_config(config, [f"policy.architecture={architecture}"])
    torch_device = torch.device(device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, torch_device)
    model.eval()
    codec = create_action_codec(cfg, torch_device)
    dataset = create_policy_dataset(cfg, split)
    current_indices = _select(dataset, delay, samples)
    current = _batch(dataset, current_indices, torch_device)

    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    raw_delay = torch.full((len(current_indices),), delay, device=torch_device, dtype=torch.long)
    if condition_source == "ground_truth":
        shifted = current["continuous_target"].float()
    else:
        lookup = {pair: index for index, pair in enumerate(dataset.index)}
        previous_indices = [
            lookup[(dataset.index[index][0], dataset.index[index][1] - delay)]
            for index in current_indices
        ]
        previous = _batch(dataset, previous_indices, torch_device)
        with torch.no_grad():
            prior = model.sample(previous)
        prior_controls = prior.float() if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(prior)
        shifted = codec.shift_and_refit(prior_controls, raw_delay)
    if cfg.data.action_representation == "raw":
        fixed = raw_action_prefix_mask(raw_delay, cfg.data.action_horizon, 7)
        fixed_rows = delay
    else:
        mapping = map_delay(
            delay,
            frequency_hz=cfg.data.frequency_hz,
            span_length_steps=cfg.spline.span_length_steps,
            degree=cfg.spline.degree,
        )
        affected = torch.full_like(raw_delay, mapping.affected_spans)
        fixed = control_support_mask(affected, cfg.spline.num_basis, 7, cfg.spline.degree)
        fixed_rows = mapping.committed_control_count

    training_type = str(payload.get("training_type", "base")).lower()
    if training_type == "base" and is_continuous_architecture(cfg.policy.architecture):
        predicted = model.sample_realtime_pigdm(
            current,
            prefix_values=shifted,
            fixed_mask=fixed,
        )
        method = (
            "base PiGDM binary hard mask, GT prefix"
            if condition_source == "ground_truth"
            else "base PiGDM binary hard mask, previous-plan prefix"
        )
    else:
        prefix_values = (
            shifted
            if is_continuous_architecture(cfg.policy.architecture)
            else codec.encode_tokens(shifted)
        )
        with torch.no_grad():
            predicted = model.sample(
                current,
                prefix_values=prefix_values,
                fixed_mask=fixed,
                use_cache=True,
            )
        method = (
            "finetuned ttRTC direct hard mask"
            if training_type in {"rtc", "ttrtc"}
            else "base discrete direct hard mask"
        )

    stats = json.loads((Path(cfg.data.prepared_path) / "normalization.json").read_text())
    low = torch.as_tensor(stats["action_q01"], device=torch_device)
    high = torch.as_tensor(stats["action_q99"], device=torch_device)
    predicted_controls = (
        predicted.float()
        if is_continuous_architecture(cfg.policy.architecture)
        else codec.decode_tokens(predicted)
    )
    predicted_physical = (codec.decode_controls(predicted_controls) + 1) * 0.5 * (high - low) + low
    condition_physical = (codec.decode_controls(shifted.float()) + 1) * 0.5 * (high - low) + low
    target_physical = current["target_trajectory"].float()
    action_min, action_max = dataset_action_minmax(cfg.data.prepared_path)

    prediction = predicted_physical.detach().cpu().numpy()
    condition = condition_physical.detach().cpu().numpy()
    target = target_physical.detach().cpu().numpy()
    episode_ids = current["episode_id"].cpu().numpy()
    frames = current["frame_index"].cpu().numpy()
    steps = np.arange(cfg.data.action_horizon)
    prefix_steps = steps[:delay]

    fig, axes = plt.subplots(
        len(current_indices),
        7,
        figsize=(22, max(3.2 * len(current_indices), 5.5)),
        sharex=True,
        squeeze=False,
    )
    for row in range(len(current_indices)):
        for dimension in range(7):
            axis = axes[row, dimension]
            axis.plot(
                steps,
                target[row, :, dimension],
                color="#2ca02c",
                linestyle="--",
                linewidth=2.0,
                label="current ground truth",
            )
            axis.plot(
                steps,
                prediction[row, :, dimension],
                color="#ff7f0e",
                linewidth=1.8,
                label="RTC prediction",
            )
            axis.plot(
                prefix_steps,
                condition[row, :delay, dimension],
                color="#1f77b4",
                marker="o",
                markersize=2.5,
                linewidth=2.2,
                label=(
                    "ground-truth prefix condition"
                    if condition_source == "ground_truth"
                    else "previous-plan condition"
                ),
            )
            axis.axvline(delay - 0.5, color="0.35", linewidth=0.9, linestyle=":")
            axis.set_ylim(float(action_min[dimension]), float(action_max[dimension]))
            _mark_outside(
                axis,
                steps,
                prediction[row, :, dimension],
                float(action_min[dimension]),
                float(action_max[dimension]),
            )
            _mark_outside(
                axis,
                prefix_steps,
                condition[row, :delay, dimension],
                float(action_min[dimension]),
                float(action_max[dimension]),
            )
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(DIMENSION_NAMES[dimension])
            if dimension == 0:
                axis.set_ylabel(f"ep {episode_ids[row]} frame {frames[row]}\naction")
            if row == len(current_indices) - 1:
                axis.set_xlabel("chunk step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    representation = cfg.data.action_representation
    fig.suptitle(
        f"{representation} h{cfg.data.observation_horizon} [{split}] — {method}; "
        f"raw delay={delay}, fixed rows={fixed_rows}",
        y=0.997,
        fontsize=15,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.003,
        "Y axes are the full dataset per-channel min/max; red triangles mark clipped out-of-range values.",
        ha="center",
        fontsize=9,
        color="#d62728",
    )
    fig.tight_layout(rect=(0, 0.018, 1, 0.945))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        choices=TRAINABLE_ARCHITECTURES,
        default="bsp_unet_fm",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay", type=int, default=6)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--condition-source",
        choices=("ground_truth", "previous_plan"),
        default="previous_plan",
    )
    args = parser.parse_args()
    visualize(
        args.config,
        args.checkpoint,
        args.split,
        args.output,
        architecture=args.architecture,
        delay=args.delay,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
        condition_source=args.condition_source,
    )


if __name__ == "__main__":
    main()
