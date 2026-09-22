#!/usr/bin/env python3
"""Quantify the VJP correction applied at every base-FM PiGDM Euler step."""

from __future__ import annotations

import argparse
import csv
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
from robot_policy.evaluation.inference_rtc import ground_truth_condition
from robot_policy.evaluation.rtc import select_rtc_indices
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.training import create_action_codec


def _batch(dataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    value = collate_policy_batch([dataset[index] for index in indices])
    return {key: tensor.to(device) for key, tensor in value.items()}


def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    selected = value.float() if mask is None else value.float().masked_select(mask)
    return float(selected.square().mean().sqrt().item()) if selected.numel() else 0.0


def _mae(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    selected = value.float() if mask is None else value.float().masked_select(mask)
    return float(selected.abs().mean().item()) if selected.numel() else 0.0


def _physical(controls, codec, low, high):
    normalized = codec.decode_controls(controls.float())
    return (normalized + 1.0) * 0.5 * (high - low) + low


def diagnose(
    config: str | Path,
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    prefix: int = 6,
    samples: int = 4,
    seed: int = 20260915,
    device: str = "cuda",
) -> dict[str, Any]:
    cfg = load_config(config, ["policy.architecture=bsp_unet_fm"])
    torch_device = torch.device(device)
    dataset = create_policy_dataset(cfg, split)
    indices = select_rtc_indices(
        dataset,
        samples,
        0,
        strategy="episode_balanced_motion",
        seed=seed,
    )
    batch = _batch(dataset, indices, torch_device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, torch_device)
    model.eval()
    if str(payload.get("training_type", "base")).lower() != "base":
        raise ValueError("VJP PiGDM diagnostic requires a base-FM checkpoint")
    codec = create_action_codec(cfg, torch_device)
    stats = json.loads((Path(cfg.data.prepared_path) / "normalization.json").read_text())
    low = torch.as_tensor(stats["action_q01"], dtype=torch.float32, device=torch_device)
    high = torch.as_tensor(stats["action_q99"], dtype=torch.float32, device=torch_device)
    target = batch["target_trajectory"].float()
    prefix_values, fixed_mask, mapping = ground_truth_condition(
        batch, cfg, cfg.policy.architecture, prefix
    )
    mutable_mask = ~fixed_mask

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        scratch_controls = model.sample(batch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rtc_controls, trace = model.sample_realtime_pigdm(
        batch,
        prefix_values=prefix_values,
        fixed_mask=fixed_mask,
        return_trace=True,
    )

    rows: list[dict[str, Any]] = []
    delta = 1.0 / len(trace)
    prefix_action_mask = torch.arange(cfg.data.action_horizon, device=torch_device) < prefix
    prefix_action_mask = prefix_action_mask[None, :, None].expand(samples, -1, 7)
    suffix_action_mask = ~prefix_action_mask
    for item in trace:
        correction = item["vjp_correction"]
        guidance = float(item["guidance_weight"])
        guidance_update = delta * guidance * correction
        unguided_next = item["unguided_next"]
        guided_next = item["guided_next"]
        decoded_delta = _physical(guided_next, codec, low, high) - _physical(
            unguided_next, codec, low, high
        )

        next_time_value = min(1.0, float(item["time"]) + delta)
        if next_time_value >= 1.0:
            unguided_endpoint = unguided_next
            guided_endpoint = guided_next
        else:
            next_time = unguided_next.new_full((len(unguided_next),), next_time_value)
            with torch.no_grad():
                unguided_endpoint = unguided_next + (1.0 - next_time_value) * model.velocity(
                    unguided_next, batch, next_time
                )
                guided_endpoint = guided_next + (1.0 - next_time_value) * model.velocity(
                    guided_next, batch, next_time
                )
        unguided_condition_error = torch.where(
            fixed_mask, prefix_values.float() - unguided_endpoint, torch.zeros_like(unguided_endpoint)
        )
        guided_condition_error = torch.where(
            fixed_mask, prefix_values.float() - guided_endpoint, torch.zeros_like(guided_endpoint)
        )
        velocity = item["predicted_velocity"]
        endpoint_error = item["condition_error"]
        velocity_update = delta * velocity
        rows.append(
            {
                "step": int(item["step"]),
                "time": float(item["time"]),
                "guidance_weight": guidance,
                "condition_endpoint_error_rms_before": _rms(endpoint_error, fixed_mask),
                "next_endpoint_condition_rms_unguided": _rms(
                    unguided_condition_error, fixed_mask
                ),
                "next_endpoint_condition_rms_guided": _rms(
                    guided_condition_error, fixed_mask
                ),
                "velocity_update_control_rms": _rms(velocity_update),
                "vjp_control_rms": _rms(correction),
                "vjp_fixed_control_rms": _rms(correction, fixed_mask),
                "vjp_mutable_control_rms": _rms(correction, mutable_mask),
                "vjp_fixed_gain_over_condition_error": _rms(correction, fixed_mask)
                / max(_rms(endpoint_error, fixed_mask), 1e-12),
                "vjp_cross_gain_over_condition_error": _rms(correction, mutable_mask)
                / max(_rms(endpoint_error, fixed_mask), 1e-12),
                "guidance_update_control_rms": _rms(guidance_update),
                "guidance_update_fixed_control_rms": _rms(guidance_update, fixed_mask),
                "guidance_update_mutable_control_rms": _rms(guidance_update, mutable_mask),
                "guidance_to_velocity_update_ratio": _rms(guidance_update)
                / max(_rms(velocity_update), 1e-12),
                "one_step_decoded_prefix_physical_rms": _rms(
                    decoded_delta, prefix_action_mask
                ),
                "one_step_decoded_suffix_physical_rms": _rms(
                    decoded_delta, suffix_action_mask
                ),
                "one_step_decoded_suffix_physical_mae": _mae(
                    decoded_delta, suffix_action_mask
                ),
            }
        )

    scratch_physical = _physical(scratch_controls, codec, low, high)
    rtc_unstitched = _physical(rtc_controls, codec, low, high)
    rtc_deployed = rtc_unstitched.clone()
    rtc_deployed[:, :prefix] = target[:, :prefix]
    naive_splice = scratch_physical.clone()
    naive_splice[:, :prefix] = target[:, :prefix]
    if not torch.equal(rtc_deployed[:, :prefix], target[:, :prefix]):
        raise AssertionError("deployed prefix must exactly equal GT")
    valid = batch["action_valid_mask"].bool()[..., None].expand_as(target)
    valid_prefix = valid & prefix_action_mask
    valid_suffix = valid & suffix_action_mask
    scratch_error = scratch_physical - target
    rtc_error = rtc_deployed - target
    rtc_minus_naive = rtc_deployed - naive_splice
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "representation": cfg.data.action_representation,
        "split": split,
        "samples": samples,
        "dataset_indices": indices,
        "episode_frame_pairs": [list(dataset.index[index]) for index in indices],
        "seed": seed,
        "paired_initial_noise": True,
        "prefix_actions": prefix,
        "condition_mapping": mapping,
        "reference_equation": (
            "x1=x_t+(1-t)v; e=mask*(y-x1); correction=VJP_x1(e); "
            "x_next=x_t+dt*(v+guidance*correction)"
        ),
        "steps": rows,
        "aggregate": {
            "mean_guidance_update_mutable_control_rms": float(
                np.mean([row["guidance_update_mutable_control_rms"] for row in rows])
            ),
            "mean_vjp_fixed_gain_over_condition_error": float(
                np.mean([row["vjp_fixed_gain_over_condition_error"] for row in rows])
            ),
            "mean_vjp_cross_gain_over_condition_error": float(
                np.mean([row["vjp_cross_gain_over_condition_error"] for row in rows])
            ),
            "mean_one_step_decoded_suffix_physical_rms": float(
                np.mean([row["one_step_decoded_suffix_physical_rms"] for row in rows])
            ),
            "final_rtc_minus_naive_stitch_suffix_physical_rms": _rms(
                rtc_minus_naive, valid_suffix
            ),
            "final_rtc_minus_naive_stitch_suffix_physical_mae": _mae(
                rtc_minus_naive, valid_suffix
            ),
            "scratch_suffix_gt_physical_mse": _rms(scratch_error, valid_suffix) ** 2,
            "rtc_suffix_gt_physical_mse": _rms(rtc_error, valid_suffix) ** 2,
            "rtc_committed_prefix_gt_physical_mse": _rms(rtc_error, valid_prefix) ** 2,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vjp_trace.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "vjp_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "paired_trajectories.npz",
        ground_truth=target.cpu().numpy(),
        from_scratch=scratch_physical.cpu().numpy(),
        naive_stitch=naive_splice.cpu().numpy(),
        pigdm_rtc=rtc_deployed.cpu().numpy(),
        rtc_minus_naive_stitch=rtc_minus_naive.cpu().numpy(),
    )

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, [row["guidance_weight"] for row in rows], marker="o")
    axes[0, 0].set_title("PiGDM guidance coefficient")
    axes[0, 0].set_ylabel("weight")
    axes[0, 1].plot(
        steps,
        [row["guidance_update_fixed_control_rms"] for row in rows],
        marker="o",
        label="fixed controls",
    )
    axes[0, 1].plot(
        steps,
        [row["guidance_update_mutable_control_rms"] for row in rows],
        marker="o",
        label="mutable controls",
    )
    axes[0, 1].set_title("VJP contribution to this Euler update")
    axes[0, 1].set_ylabel("control RMS")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[1, 0].plot(
        steps,
        [row["one_step_decoded_prefix_physical_rms"] for row in rows],
        marker="o",
        label="raw-action prefix",
    )
    axes[1, 0].plot(
        steps,
        [row["one_step_decoded_suffix_physical_rms"] for row in rows],
        marker="o",
        label="raw-action suffix",
    )
    axes[1, 0].set_title("Guided vs unguided one-step decoded change")
    axes[1, 0].set_ylabel("physical action RMS")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 1].plot(
        steps,
        [row["next_endpoint_condition_rms_unguided"] for row in rows],
        marker="o",
        label="without VJP step",
    )
    axes[1, 1].plot(
        steps,
        [row["next_endpoint_condition_rms_guided"] for row in rows],
        marker="o",
        label="with VJP step",
    )
    axes[1, 1].set_title("Next endpoint residual on conditioned controls")
    axes[1, 1].set_ylabel("normalized control RMS")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Euler / denoising step")
        axis.grid(alpha=0.25)
    aggregate = summary["aggregate"]
    fig.suptitle(
        f"{cfg.data.action_representation} FM PiGDM VJP trace [{split}]\n"
        f"final suffix RTC-vs-naive-stitch RMS={aggregate['final_rtc_minus_naive_stitch_suffix_physical_rms']:.5f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "vjp_trace.png", dpi=180)
    plt.close(fig)

    difference = rtc_minus_naive.abs().mean(dim=(0, 2)).cpu().numpy()
    dimension_rms = rtc_minus_naive[:, prefix:].square().mean(dim=(0, 1)).sqrt().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(np.arange(len(difference)), difference, marker="o", ms=3)
    axes[0].axvspan(-0.5, prefix - 0.5, color="#4c78a8", alpha=0.10)
    axes[0].axvline(prefix - 0.5, color="0.35", linestyle=":")
    axes[0].set_title("PiGDM RTC − naive GT-prefix stitch")
    axes[0].set_xlabel("raw action step")
    axes[0].set_ylabel("mean absolute physical difference")
    names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    axes[1].bar(names, dimension_rms)
    axes[1].set_title("Suffix difference by action dimension")
    axes[1].set_ylabel("physical RMS")
    for axis in axes:
        axis.grid(alpha=0.25, axis="y")
    fig.suptitle(
        f"{cfg.data.action_representation} FM [{split}]: does PiGDM differ from direct stitching?"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "rtc_vs_naive_stitch.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = diagnose(
        args.config,
        args.checkpoint,
        args.split,
        args.output_dir,
        prefix=args.prefix,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
