from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_policy.config import (
    TRAINABLE_ARCHITECTURES,
    is_continuous_architecture,
    load_config,
    require_active_architecture,
    require_active_model_size,
)
from robot_policy.data.dataset import PreparedPolicyDataset, collate_policy_batch, create_policy_dataset
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


DIMENSION_NAMES = ("joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "gripper")


def dataset_action_minmax(prepared_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return true physical action extrema across every prepared episode."""
    action_files = sorted((Path(prepared_path) / "actions").glob("episode_*.npz"))
    if not action_files:
        raise FileNotFoundError(f"no prepared action episodes under {Path(prepared_path) / 'actions'}")
    minimum = np.full(7, np.inf, dtype=np.float64)
    maximum = np.full(7, -np.inf, dtype=np.float64)
    for path in action_files:
        with np.load(path) as episode:
            actions = np.asarray(episode["action"], dtype=np.float64)
        minimum = np.minimum(minimum, actions.min(axis=0))
        maximum = np.maximum(maximum, actions.max(axis=0))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(minimum >= maximum):
        raise ValueError("dataset action min/max is non-finite or degenerate")
    return minimum.astype(np.float32), maximum.astype(np.float32)


def select_full_horizon_indices(dataset: PreparedPolicyDataset, count: int, seed: int) -> list[int]:
    """Select deterministic, episode-balanced examples with a complete target chunk."""
    by_episode: dict[int, list[int]] = {}
    for index, (episode_id, _) in enumerate(dataset.index):
        if bool(dataset[index]["action_valid_mask"].all()):
            by_episode.setdefault(episode_id, []).append(index)
    if not by_episode:
        raise ValueError("the requested split contains no complete action chunks")
    rng = np.random.default_rng(seed)
    for values in by_episode.values():
        rng.shuffle(values)
    selected: list[int] = []
    episodes = sorted(by_episode)
    cursor = {episode_id: 0 for episode_id in episodes}
    while len(selected) < count:
        progressed = False
        for episode_id in episodes:
            position = cursor[episode_id]
            values = by_episode[episode_id]
            if position < len(values):
                selected.append(values[position])
                cursor[episode_id] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if not selected:
        raise ValueError("no evaluation samples were selected")
    return selected


def ground_truth_condition(
    batch: dict[str, torch.Tensor], cfg: Any, architecture: str, raw_prefix_actions: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return checkpoint-native prefix values and a representation-aware fixed mask."""
    if raw_prefix_actions <= 0 or raw_prefix_actions >= cfg.data.action_horizon:
        raise ValueError(f"raw_prefix_actions must be in 1..{cfg.data.action_horizon - 1}")
    batch_size = len(batch["state"])
    device = batch["state"].device
    if cfg.data.action_representation == "raw":
        fixed = raw_action_prefix_mask(
            torch.full((batch_size,), raw_prefix_actions, device=device, dtype=torch.long),
            cfg.data.action_horizon,
            7,
        )
        mapping = {
            "raw_prefix_actions": raw_prefix_actions,
            "whole_spans": None,
            "phase_steps": None,
            "affected_spans": None,
            "fixed_control_rows": raw_prefix_actions,
        }
    else:
        delay = map_delay(
            raw_prefix_actions,
            frequency_hz=cfg.data.frequency_hz,
            span_length_steps=cfg.spline.span_length_steps,
            degree=cfg.spline.degree,
            executable_spans=cfg.data.action_horizon // cfg.spline.span_length_steps,
        )
        affected = torch.full((batch_size,), delay.affected_spans, device=device, dtype=torch.long)
        fixed = control_support_mask(affected, cfg.spline.num_basis, 7, cfg.spline.degree)
        mapping = {
            "raw_prefix_actions": raw_prefix_actions,
            "whole_spans": delay.whole_spans,
            "phase_steps": delay.phase_steps,
            "affected_spans": delay.affected_spans,
            "fixed_control_rows": delay.committed_control_count,
        }
    values = batch["continuous_target"].float() if is_continuous_architecture(architecture) else batch["discrete_target"].long()
    return values, fixed, mapping


def _batch(dataset: PreparedPolicyDataset, indices: Iterable[int], device: torch.device) -> dict[str, torch.Tensor]:
    result = collate_policy_batch([dataset[index] for index in indices])
    return {key: value.to(device) for key, value in result.items()}


def _aggregate(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(values.mean())


def _plot_examples(path: Path, artifact: dict[str, np.ndarray], prefix: int, title: str) -> None:
    target = artifact["target_physical"]
    predicted = artifact["predicted_physical"]
    episode_ids = artifact["episode_id"]
    frames = artifact["frame_index"]
    action_min = artifact.get("action_min")
    action_max = artifact.get("action_max")
    rows = len(target)
    fig, axes = plt.subplots(rows, 7, figsize=(22, max(3.2 * rows, 5.5)), sharex=True, squeeze=False)
    steps = np.arange(target.shape[1])
    for row in range(rows):
        for dim in range(7):
            ax = axes[row, dim]
            ax.plot(steps[:prefix], target[row, :prefix, dim], color="#1f77b4", marker="o", ms=2.5,
                    lw=2.2, label="condition action")
            ax.plot(steps[prefix:], target[row, prefix:, dim], color="#2ca02c", ls="--", lw=2.0,
                    label="ground-truth inpainting")
            ax.plot(steps[prefix:], predicted[row, prefix:, dim], color="#ff7f0e", lw=1.8,
                    label="predicted inpainting")
            ax.axvline(prefix - 0.5, color="0.35", lw=0.9, ls=":")
            if action_min is not None and action_max is not None:
                ax.set_ylim(float(action_min[dim]), float(action_max[dim]))
                future_steps = steps[prefix:]
                future_prediction = predicted[row, prefix:, dim]
                below = future_prediction < action_min[dim]
                above = future_prediction > action_max[dim]
                if below.any():
                    ax.scatter(future_steps[below], np.full(int(below.sum()), action_min[dim]),
                               marker="v", s=18, color="#d62728", zorder=5)
                if above.any():
                    ax.scatter(future_steps[above], np.full(int(above.sum()), action_max[dim]),
                               marker="^", s=18, color="#d62728", zorder=5)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(DIMENSION_NAMES[dim])
            if dim == 0:
                ax.set_ylabel(f"ep {episode_ids[row]} frame {frames[row]}\naction")
            if row == rows - 1:
                ax.set_xlabel("chunk step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.997, fontsize=15)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False)
    fig.text(0.5, 0.003, "Red triangles mark predictions outside the full dataset min/max.",
             ha="center", fontsize=9, color="#d62728")
    fig.tight_layout(rect=(0, 0.018, 1, 0.945))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metrics(
    path: Path,
    curves: list[dict[str, Any]],
    title: str,
    *,
    mse_ylim: tuple[float, float] | None = None,
) -> None:
    x = [curve["raw_prefix_actions"] for curve in curves]
    mse = [curve["suffix_physical_mse"] for curve in curves]
    prefix = [curve["prefix_reference_normalized_mse"] for curve in curves]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, mse, marker="o", color="#ff7f0e")
    axes[0].set_title("Inpainted suffix error")
    axes[0].set_ylabel("physical action MSE")
    axes[1].plot(x, prefix, marker="o", color="#1f77b4")
    axes[1].set_title("Condition preservation")
    axes[1].set_ylabel("normalized MSE vs supplied prefix")
    for ax in axes:
        ax.set_xlabel("ground-truth condition length (raw steps)")
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    if mse_ylim is not None:
        axes[0].set_ylim(*mse_ylim)
    if max(abs(value) for value in prefix) == 0.0:
        axes[1].set_ylim(-1e-12, 1e-12)
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1].text(0.5, 0.88, "exactly preserved", transform=axes[1].transAxes,
                     ha="center", color="#1f77b4")
    else:
        axes[1].set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def evaluate(
    cfg: Any,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    prefixes: tuple[int, ...] = (2, 4, 6, 8, 10),
    max_samples: int = 64,
    batch_size: int = 32,
    plot_prefix: int = 6,
    plot_samples: int = 5,
    split: str = "test",
    seed: int = 20260915,
    device: str = "cuda",
) -> dict[str, Any]:
    require_active_architecture(cfg.policy.architecture, "inference-RTC evaluation")
    require_active_model_size(cfg.policy.model_size, "inference-RTC evaluation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    model, payload = load_policy_checkpoint(checkpoint, cfg, torch_device)
    model.eval()
    codec = create_action_codec(cfg, torch_device)
    dataset = create_policy_dataset(cfg, split)
    indices = select_full_horizon_indices(dataset, max_samples, seed)
    plotted_indices = set(indices[: min(plot_samples, len(indices))])
    stats = json.loads((Path(cfg.data.prepared_path) / "normalization.json").read_text())
    dataset_min, dataset_max = dataset_action_minmax(cfg.data.prepared_path)
    action_low = torch.as_tensor(stats["action_q01"], dtype=torch.float32, device=torch_device)
    action_high = torch.as_tensor(stats["action_q99"], dtype=torch.float32, device=torch_device)
    scale = 0.5 * (action_high - action_low)
    offset = 0.5 * (action_high + action_low)
    curves: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    plot_artifact: dict[str, list[np.ndarray]] = {
        "target_physical": [], "predicted_physical": [], "reference_physical": [],
        "episode_id": [], "frame_index": [], "dataset_index": [],
    }
    for raw_prefix in prefixes:
        if raw_prefix <= 0 or raw_prefix >= cfg.data.action_horizon:
            raise ValueError(f"invalid prefix {raw_prefix}; expected 1..{cfg.data.action_horizon - 1}")
        mapping: dict[str, Any] | None = None
        delay_rows: list[dict[str, Any]] = []
        elapsed_ms = 0.0
        torch.manual_seed(seed + raw_prefix)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed + raw_prefix)
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            batch = _batch(dataset, chosen, torch_device)
            prefix_values, fixed_mask, mapping = ground_truth_condition(
                batch, cfg, cfg.policy.architecture, raw_prefix
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            started = time.perf_counter()
            if (
                cfg.data.action_representation == "raw"
                and is_continuous_architecture(cfg.policy.architecture)
                and str(payload.get("training_type", "base")).lower() == "base"
            ):
                prediction = model.sample_realtime_pigdm(
                    batch,
                    prefix_values=prefix_values,
                    fixed_mask=fixed_mask,
                )
            else:
                with torch.no_grad():
                    prediction = model.sample(
                        batch,
                        prefix_values=prefix_values,
                        fixed_mask=fixed_mask,
                        use_cache=True,
                    )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            elapsed_ms += (time.perf_counter() - started) * 1000.0
            predicted_controls = prediction.float() if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(prediction)
            reference_controls = (
                batch["continuous_target"].float()
                if is_continuous_architecture(cfg.policy.architecture)
                else codec.decode_tokens(batch["discrete_target"].long())
            )
            predicted_normalized = codec.decode_controls(predicted_controls)
            reference_normalized = codec.decode_controls(reference_controls)
            predicted_physical = predicted_normalized * scale + offset
            reference_physical = reference_normalized * scale + offset
            target_physical = batch["target_trajectory"].float()
            target_normalized = batch["normalized_target_trajectory"].float()
            for local, dataset_index in enumerate(chosen):
                suffix = slice(raw_prefix, cfg.data.action_horizon)
                prefix_slice = slice(0, raw_prefix)
                suffix_physical_error = predicted_physical[local, suffix] - target_physical[local, suffix]
                suffix_normalized_error = predicted_normalized[local, suffix] - target_normalized[local, suffix]
                suffix_reference_error = predicted_normalized[local, suffix] - reference_normalized[local, suffix]
                prefix_reference_error = predicted_normalized[local, prefix_slice] - reference_normalized[local, prefix_slice]
                prefix_target_error = predicted_physical[local, prefix_slice] - target_physical[local, prefix_slice]
                fixed_error = (predicted_controls[local] - reference_controls[local]).masked_select(fixed_mask[local])
                row = {
                    "dataset_index": int(dataset_index),
                    "episode_id": int(batch["episode_id"][local].item()),
                    "frame_index": int(batch["frame_index"][local].item()),
                    "raw_prefix_actions": raw_prefix,
                    "suffix_physical_mse": float(suffix_physical_error.square().mean().item()),
                    "suffix_physical_mae": float(suffix_physical_error.abs().mean().item()),
                    "suffix_normalized_mse": float(suffix_normalized_error.square().mean().item()),
                    "suffix_reference_normalized_mse": float(suffix_reference_error.square().mean().item()),
                    "prefix_reference_normalized_mse": float(prefix_reference_error.square().mean().item()),
                    "prefix_target_physical_mse": float(prefix_target_error.square().mean().item()),
                    "fixed_control_max_abs": float(fixed_error.abs().max().item()) if fixed_error.numel() else 0.0,
                }
                delay_rows.append(row)
                sample_rows.append(row)
                if raw_prefix == plot_prefix and dataset_index in plotted_indices:
                    plot_artifact["target_physical"].append(target_physical[local].cpu().numpy())
                    plot_artifact["predicted_physical"].append(predicted_physical[local].cpu().numpy())
                    plot_artifact["reference_physical"].append(reference_physical[local].cpu().numpy())
                    plot_artifact["episode_id"].append(np.asarray(row["episode_id"], dtype=np.int64))
                    plot_artifact["frame_index"].append(np.asarray(row["frame_index"], dtype=np.int64))
                    plot_artifact["dataset_index"].append(np.asarray(dataset_index, dtype=np.int64))
        assert mapping is not None
        curves.append({
            **mapping,
            "samples": len(delay_rows),
            "suffix_physical_mse": _aggregate(delay_rows, "suffix_physical_mse"),
            "suffix_physical_mae": _aggregate(delay_rows, "suffix_physical_mae"),
            "suffix_normalized_mse": _aggregate(delay_rows, "suffix_normalized_mse"),
            "suffix_reference_normalized_mse": _aggregate(delay_rows, "suffix_reference_normalized_mse"),
            "prefix_reference_normalized_mse": _aggregate(delay_rows, "prefix_reference_normalized_mse"),
            "prefix_target_physical_mse": _aggregate(delay_rows, "prefix_target_physical_mse"),
            "fixed_control_max_abs": max(row["fixed_control_max_abs"] for row in delay_rows),
            "sampling_ms_per_sample": elapsed_ms / len(delay_rows),
        })
    if plot_prefix not in prefixes:
        raise ValueError("plot_prefix must be included in prefixes")
    artifact_arrays = {key: np.stack(value) for key, value in plot_artifact.items()}
    artifact_arrays["action_min"] = dataset_min
    artifact_arrays["action_max"] = dataset_max
    plotted_prediction = artifact_arrays["predicted_physical"][:, plot_prefix:]
    below_dataset = (plotted_prediction < dataset_min[None, None, :]).sum(axis=(0, 1))
    above_dataset = (plotted_prediction > dataset_max[None, None, :]).sum(axis=(0, 1))
    np.savez_compressed(output_dir / "trajectory_examples.npz", **artifact_arrays)
    checkpoint_training_type = str(payload["training_type"])
    # Older B-spline fine-tuning payloads used `rtc`; organized artifacts and
    # current terminology use `ttrtc`. Preserve the source field separately.
    training_type = "ttrtc" if checkpoint_training_type in {"rtc", "ttrtc"} else checkpoint_training_type
    variant = f"{cfg.data.action_representation}_{cfg.policy.architecture}_{training_type}"
    report = {
        "schema_version": 1,
        "variant": variant,
        "model_size": cfg.policy.model_size,
        "vision_tokens": (
            None
            if cfg.data.observation_source == "rgb"
            else len(cfg.data.camera_keys) * cfg.vision.pooled_grid**2
        ),
        "observation_horizon": cfg.data.observation_horizon,
        "observation_source": cfg.data.observation_source,
        "architecture": cfg.policy.architecture,
        "training_type": training_type,
        "checkpoint_training_type": checkpoint_training_type,
        "action_representation": cfg.data.action_representation,
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split,
        "sample_selection": "deterministic episode-balanced complete 30-step chunks",
        "seed": seed,
        "samples": len(indices),
        "dataset_indices": indices,
        "prefix_source": "current ground-truth action chunk (oracle prefix)",
        "inference_mode": (
            "base flow PiGDM with binary hard-prefix operator"
            if cfg.data.action_representation == "raw"
            and is_continuous_architecture(cfg.policy.architecture)
            and checkpoint_training_type == "base"
            else "training-time RTC direct hard-prefix inpainting"
            if is_continuous_architecture(cfg.policy.architecture)
            and checkpoint_training_type in {"rtc", "ttrtc"}
            else "representation-native fixed-prefix inpainting from scratch"
        ),
        "bspline_conditioning": (
            "freeze the complete cubic control support for every affected raw-action span"
            if cfg.data.action_representation == "bspline" else None
        ),
        "discrete_rounds": cfg.policy.discrete_rounds if not is_continuous_architecture(cfg.policy.architecture) else None,
        "fm_steps": cfg.policy.fm_steps if is_continuous_architecture(cfg.policy.architecture) else None,
        "joint_kv_cache": cfg.policy.architecture == "discrete_joint",
        "prefixes": list(prefixes),
        "plot_prefix": plot_prefix,
        "trajectory_axis_limits": {
            "source": "true per-channel physical min/max across every prepared dataset episode",
            "min": artifact_arrays["action_min"].tolist(),
            "max": artifact_arrays["action_max"].tolist(),
            "plotted_prediction_points_below_min_by_dimension": below_dataset.tolist(),
            "plotted_prediction_points_above_max_by_dimension": above_dataset.tolist(),
        },
        "curves": curves,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    title = f"{variant} [{split} split]: ground-truth-prefix RTC inpainting"
    _plot_examples(output_dir / "trajectory_examples.png", artifact_arrays, plot_prefix, title)
    _plot_metrics(output_dir / "metrics_vs_prefix.png", curves, title)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle-prefix inference RTC inpainting")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--architecture", required=True, choices=TRAINABLE_ARCHITECTURES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefixes", default="2,4,6,8,10")
    parser.add_argument("--plot-prefix", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    prefixes = tuple(int(value) for value in args.prefixes.split(",") if value.strip())
    cfg = load_config(args.config, [*args.set, f"policy.architecture={args.architecture}"])
    report = evaluate(
        cfg, args.checkpoint, args.output_dir, prefixes=prefixes, max_samples=args.max_samples,
        batch_size=args.batch_size, plot_prefix=args.plot_prefix, plot_samples=args.plot_samples,
        split=args.split, seed=args.seed, device=args.device,
    )
    print(json.dumps({"variant": report["variant"], "samples": report["samples"], "output": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
