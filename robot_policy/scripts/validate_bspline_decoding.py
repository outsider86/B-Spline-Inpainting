#!/usr/bin/env python3
"""Numerically validate the exact B-spline decode path used by sweep evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import BSpline

from robot_policy.config import load_config
from robot_policy.encoders.bspline_adapter import BSplineAdapter
from robot_policy.rtc.training import TorchSplineCodec


def stats(values: list[np.ndarray]) -> dict[str, float]:
    x = np.concatenate([item.reshape(-1) for item in values]).astype(np.float64)
    return {"mae": float(np.abs(x).mean()), "rmse": float(np.sqrt(np.mean(x * x))), "max_abs": float(np.abs(x).max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/SWEEP/summary/decoder_validation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    prepared = root / "outputs" / "SWEEP" / "summary" / "cache" / "bspline"
    cfg = load_config(root / "configs" / "model_size_sweep" / "dit_s_bspline.yaml", [f"data.prepared_path={prepared}"])
    encoder_path = prepared / "encoder.json"
    encoder = json.loads(encoder_path.read_text())
    adapter = BSplineAdapter(cfg, encoder["calibration"])
    codec = TorchSplineCodec(prepared, torch.device("cpu"))
    basis = adapter.basis.astype(np.float64)
    knots = np.asarray(adapter.encoder.knots, dtype=np.float64)
    sample_steps = np.arange(cfg.data.action_horizon, dtype=np.float64)
    dense_steps = np.linspace(
        0.0,
        np.nextafter(float(cfg.data.action_horizon), 0.0),
        1201,
    )
    identity_controls = np.eye(cfg.spline.num_basis, dtype=np.float64)
    scipy_basis_spline = BSpline(
        knots,
        identity_controls,
        cfg.spline.degree,
        extrapolate=False,
        axis=0,
    )
    dense_basis = np.asarray(scipy_basis_spline(dense_steps), dtype=np.float64)
    sampled_scipy_basis = np.asarray(scipy_basis_spline(sample_steps), dtype=np.float64)
    sampled_basis_difference = sampled_scipy_basis - basis

    continuous_differences: list[np.ndarray] = []
    quantized_differences: list[np.ndarray] = []
    torch_differences: list[np.ndarray] = []
    fit_differences: list[np.ndarray] = []
    quantization_differences: list[np.ndarray] = []
    worst = None
    split = json.loads((prepared / "splits.json").read_text())
    low = np.asarray(encoder["calibration"]["low"], dtype=np.float64)
    high = np.asarray(encoder["calibration"]["high"], dtype=np.float64)
    for episode_id in split["test"]:
        with np.load(prepared / "actions" / f"episode_{episode_id:06d}.npz") as data:
            controls = data["continuous_target"].astype(np.float64)
            tokens = data["discrete_target"].astype(np.float64)
            stored = data["reconstruction"].astype(np.float64)
            stored_quantized = data["quantized_reconstruction"].astype(np.float64)
            decoded = np.einsum("tn,bnd->btd", basis, controls)
            token_controls = low + tokens / 255.0 * (high - low)
            decoded_quantized = np.einsum("tn,bnd->btd", basis, token_controls)
            torch_decoded = codec.decode_controls(torch.from_numpy(controls).float()).numpy().astype(np.float64)
            continuous_differences.append(decoded - stored)
            quantized_differences.append(decoded_quantized - stored_quantized)
            torch_differences.append(torch_decoded - decoded)

            actions = data["normalized_action"].astype(np.float64)
            indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(cfg.data.action_horizon)[None], len(actions) - 1)
            targets = actions[indices]
            valid = data["action_valid_mask"].astype(bool)[..., None]
            fit = decoded - targets
            quantization = decoded_quantized - decoded
            fit_differences.append(fit[valid.repeat(7, axis=2)])
            quantization_differences.append(quantization[valid.repeat(7, axis=2)])
            per_window = np.mean(np.abs(fit), axis=(1, 2))
            frame = int(per_window.argmax())
            candidate = (float(per_window[frame]), episode_id, frame, targets[frame], decoded[frame], decoded_quantized[frame])
            if worst is None or candidate[0] > worst[0]:
                worst = candidate

    manifests_match = []
    for size in ("dit_s", "dit_b", "dit_l"):
        manifest = json.loads((root / "outputs" / "SWEEP" / size / "bspline" / "checkpoints" / "checkpoint_manifest.json").read_text())
        versions = [entry["encoder_version"] for entry in manifest["checkpoints"]]
        manifests_match.append({
            "model_size": size,
            "entries": len(versions),
            "type_match": all(item["type"] == encoder["type"] for item in versions),
            "config_match": all(item["config"] == encoder["config"] for item in versions),
            "tokenizer_id_match": all(item["tokenizer_id"] == encoder["tokenizer_id"] for item in versions),
            "calibration_match": all(item["calibration"] == encoder["calibration"] for item in versions),
        })

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, cfg.spline.num_basis))
    for index, color in enumerate(colors):
        ax.plot(dense_steps, dense_basis[:, index], color=color, lw=1.8)
        ax.scatter(
            sample_steps,
            basis[:, index],
            color=color,
            s=9,
            alpha=0.8,
            zorder=3,
        )
    ax.set(
        title="Cubic uniform-left B-spline basis (dense evaluation)",
        xlabel="continuous action time (steps)",
        ylabel="basis weight",
        xlim=(0, cfg.data.action_horizon),
    )
    ax.text(
        0.995,
        0.98,
        "curves: dense SciPy BSpline evaluation\ndots: exact 30 decoder samples",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.grid(alpha=.25)
    fig.savefig(output / "bspline_basis.png", dpi=180)
    plt.close(fig)

    _, episode_id, frame, target, decoded, decoded_quantized = worst
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, layout="constrained")
    for dimension, ax in enumerate(axes.flat):
        if dimension == 7:
            ax.axis("off")
            continue
        ax.plot(target[:, dimension], label="normalized target", color="#333333", lw=2)
        ax.plot(decoded[:, dimension], label="continuous spline decode", color="#f58518")
        ax.plot(decoded_quantized[:, dimension], label="token spline decode", color="#4c78a8", ls="--")
        ax.set_title("gripper" if dimension == 6 else f"joint {dimension}")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Worst mean-fit-error test window: episode {episode_id}, frame {frame}")
    fig.savefig(output / "bspline_decode_example.png", dpi=180)
    plt.close(fig)

    action_manifest = json.loads((prepared / "action_manifest.json").read_text())
    report = {
        "passed": all(
            item[key]
            for item in manifests_match
            for key in ("type_match", "config_match", "tokenizer_id_match", "calibration_match")
        )
        and stats(continuous_differences)["max_abs"] < 1e-6
        and stats(quantized_differences)["max_abs"] < 1e-6
        and stats([sampled_basis_difference])["max_abs"] < 1e-12,
        "decoder_order": [
            "discrete policies: uint8 token -> per-control calibration dequantization",
            "all B-spline policies: fixed authoritative 30x18 basis @ 18x7 controls -> 30x7 normalized actions",
            "normalized actions -> training-split q01/q99 scaling -> 30x7 physical joint/gripper commands",
        ],
        "geometry": {
            "degree": cfg.spline.degree,
            "span_length_steps": cfg.spline.span_length_steps,
            "control_shape": [cfg.spline.num_basis, 7],
            "basis_shape": list(basis.shape),
            "decoded_shape": [cfg.data.action_horizon, 7],
            "basis_rank": int(np.linalg.matrix_rank(basis)),
            "basis_row_sum_max_error": float(np.abs(basis.sum(1) - 1).max()),
            "basis_sha256_float64": sha256(basis.tobytes()).hexdigest(),
            "scipy_dense_basis_evaluation_points": len(dense_steps),
            "scipy_at_integer_steps_vs_decoder_basis": stats([sampled_basis_difference]),
        },
        "encoder": {
            "type": encoder["type"],
            "implementation_version": encoder["config"]["implementation_version"],
            "tokenizer_id": encoder["tokenizer_id"],
            "encoder_json_sha256": sha256(encoder_path.read_bytes()).hexdigest(),
            "checkpoint_manifest_matches": manifests_match,
        },
        "recomputed_test_cache_agreement": {
            "continuous_basis_vs_stored_reconstruction": stats(continuous_differences),
            "token_dequant_basis_vs_stored_quantized_reconstruction": stats(quantized_differences),
            "torch_codec_vs_authoritative_numpy_basis": stats(torch_differences),
        },
        "all_dataset_preprocessing_metrics": action_manifest["metrics"],
        "test_split_recomputed_metrics": {
            "continuous_spline_fit_normalized": stats(fit_differences),
            "additional_token_quantization_normalized": stats(quantization_differences),
        },
        "worst_mean_fit_error_test_window": {"episode_id": episode_id, "frame_index": frame, "mean_absolute_error": worst[0]},
        "figures": [str((output / "bspline_basis.png").resolve()), str((output / "bspline_decode_example.png").resolve())],
    }
    (output / "decode_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("B-spline decode validation failed")


if __name__ == "__main__":
    main()
