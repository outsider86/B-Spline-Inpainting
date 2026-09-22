#!/usr/bin/env python3
"""Validate and compare the two 10k BSP-UNet v4 FM checkpoints with v3 50k."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4"
SUMMARY = OUTPUT / "summary"
V3_SUMMARY = ROOT / "outputs" / "BSP_UNET_V3" / "summary" / "evaluation_summary.json"
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")
SPLITS = ("train", "test")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _publish_wandb(summary: dict[str, Any], mode: str) -> dict[str, str] | None:
    if mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        entity="401910710-university-of-california-berkeley",
        project="robot-policy-bsp-unet-v4-evaluation",
        name="bsp-unet-v4-10k-vs-v3-50k",
        job_type="evaluation",
        mode=mode,
        dir=str((OUTPUT / "wandb").resolve()),
        config={
            "v4_updates": 10_000,
            "v3_updates": 50_000,
            "observation_horizon": 2,
            "images_per_example": 4,
            "variants": list(VARIANTS),
        },
    )
    rows = summary["v3_v4_open_loop"]
    columns = list(rows[0])
    rtc = summary["v4_rtc"]
    rtc_columns = list(rtc[0])
    run.log({
        "v3_v4_open_loop": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in rows]
        ),
        "v4_rtc": wandb.Table(
            columns=rtc_columns,
            data=[[row[column] for column in rtc_columns] for row in rtc],
        ),
        "v3_v4_train_test": wandb.Image(str(SUMMARY / "v3_v4_train_test.png")),
        "v3_v4_generalization_gap": wandb.Image(
            str(SUMMARY / "v3_v4_generalization_gap.png")
        ),
        "performance_vs_latency": wandb.Image(str(SUMMARY / "performance_vs_latency.png")),
        "rtc_train_mse_vs_delay": wandb.Image(str(SUMMARY / "rtc_train_mse_vs_delay.png")),
        "rtc_test_mse_vs_delay": wandb.Image(str(SUMMARY / "rtc_test_mse_vs_delay.png")),
    })
    artifact = wandb.Artifact("bsp-unet-v4-evaluation", type="evaluation")
    for name in (
        "evaluation_summary.json", "v3_v4_open_loop.csv", "v4_rtc.csv",
        "v4_latency.csv", "v3_v4_train_test.png",
        "v3_v4_generalization_gap.png", "performance_vs_latency.png",
        "rtc_train_mse_vs_delay.png", "rtc_test_mse_vs_delay.png",
        "SUMMARY_EN.md", "SUMMARY_CN.md",
    ):
        artifact.add_file(str((SUMMARY / name).resolve()))
    artifact.add_file(str((SUMMARY / "deployment" / "websocket_validation.json").resolve()))
    run.log_artifact(artifact)
    info = {"run_id": run.id, "url": run.url, "project": run.project}
    run.finish()
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    args = parser.parse_args()
    v3 = _read(V3_SUMMARY)
    v3_lookup = {
        (row["variant"], row["split"]): row
        for row in v3["open_loop"]
        if row["variant"] in VARIANTS and row["stage"] == "base"
    }
    open_rows: list[dict[str, Any]] = []
    rtc_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for variant in VARIANTS:
        checkpoint = (OUTPUT / "checkpoints" / variant / "base.pt").resolve()
        folder = SUMMARY / variant / "base"
        variant_checks: dict[str, Any] = {
            "checkpoint_exists": checkpoint.is_file(),
            "trajectory_pngs": all(
                (folder / f"rtc_inpainting_{split}.png").is_file() for split in SPLITS
            ),
        }
        for split in SPLITS:
            open_report = _read(folder / f"open_loop_{split}.json")
            rtc_report = _read(folder / f"rtc_{split}.json")
            variant_checks[f"{split}_identity"] = (
                open_report["split"] == split
                and rtc_report["split"] == split
                and open_report["checkpoint_training_type"] == "base"
                and Path(open_report["checkpoint"]).resolve() == checkpoint
                and Path(rtc_report["checkpoint"]).resolve() == checkpoint
            )
            variant_checks[f"{split}_pigdm"] = (
                rtc_report["rtc_inference_method"] == "pigdm_binary_hard_mask"
            )
            variant_checks[f"{split}_all_delays"] = (
                [curve["raw_delay_actions"] for curve in rtc_report["curves"]]
                == list(range(11))
            )
            selection = rtc_report["sample_selection"]
            if split in selections:
                variant_checks[f"{split}_cohort_match"] = selection == selections[split]
            else:
                selections[split] = selection
                variant_checks[f"{split}_cohort_match"] = True
            v4_row = {
                "version": "v4_10k",
                "variant": variant,
                "split": split,
                "samples": open_report["samples"],
                "physical_action_mse": open_report["decoded_all_physical_error"]["mse"],
                "physical_action_mae": open_report["decoded_all_physical_error"]["mae"],
                "normalized_action_mse": open_report["decoded_all_normalized_error"]["mse"],
            }
            old = v3_lookup[(variant, split)]
            v3_row = {
                "version": "v3_50k",
                "variant": variant,
                "split": split,
                "samples": old["samples"],
                "physical_action_mse": old["physical_action_mse"],
                "physical_action_mae": old["physical_action_mae"],
                "normalized_action_mse": old["normalized_action_mse"],
            }
            open_rows.extend((v3_row, v4_row))
            for curve in rtc_report["curves"]:
                rtc_rows.append({
                    "variant": variant,
                    "split": split,
                    "raw_delay_actions": curve["raw_delay_actions"],
                    "delay_ms": curve["delay_ms"],
                    "affected_spans": curve["affected_spans"],
                    "fixed_rows": curve["committed_control_count"],
                    "samples": rtc_report["samples_per_delay"],
                    "physical_action_mse": curve["decoded_physical_error"]["mse"],
                    "physical_action_mae": curve["decoded_physical_error"]["mae"],
                })
        latency = _read(SUMMARY / "latency" / f"{variant}.json")
        variant_checks["latency_identity"] = Path(latency["checkpoint"]).resolve() == checkpoint
        latency_rows.append({
            "variant": variant,
            "representation": "bspline" if "bspline" in variant else "raw",
            "sampling_p50_ms_batch1": latency["sampling_steps-12"]["p50_ms"],
            "sampling_p95_ms_batch1": latency["sampling_steps-12"]["p95_ms"],
            "online_vision_p50_ms": latency["online_vision_total"]["p50_ms"],
            "peak_memory_bytes": latency["peak_memory_bytes"],
        })
        checks[variant] = variant_checks

    failures = {
        variant: [key for key, value in values.items() if isinstance(value, bool) and not value]
        for variant, values in checks.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    deployment = _read(SUMMARY / "deployment" / "websocket_validation.json")
    if failures or not deployment.get("passed"):
        raise RuntimeError(f"v4 evaluation validation failed: {failures}, deployment={deployment}")

    gaps = []
    for version in ("v3_50k", "v4_10k"):
        for variant in VARIANTS:
            train = next(
                row for row in open_rows
                if row["version"] == version and row["variant"] == variant and row["split"] == "train"
            )
            test = next(
                row for row in open_rows
                if row["version"] == version and row["variant"] == variant and row["split"] == "test"
            )
            gaps.append({
                "version": version,
                "variant": variant,
                "train_physical_mse": train["physical_action_mse"],
                "test_physical_mse": test["physical_action_mse"],
                "test_minus_train_mse": test["physical_action_mse"] - train["physical_action_mse"],
                "test_over_train_ratio": test["physical_action_mse"] / train["physical_action_mse"],
            })

    SUMMARY.mkdir(parents=True, exist_ok=True)
    _csv(SUMMARY / "v3_v4_open_loop.csv", open_rows)
    _csv(SUMMARY / "v4_rtc.csv", rtc_rows)
    _csv(SUMMARY / "v4_latency.csv", latency_rows)
    _csv(SUMMARY / "v3_v4_generalization_gap.csv", gaps)

    labels = ("raw h2", "B-spline h2")
    x = list(range(2))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, split in zip(axes, SPLITS):
        for offset, (version, label) in enumerate((("v3_50k", "v3 50k"), ("v4_10k", "v4 10k"))):
            values = [
                next(row for row in open_rows if row["version"] == version and row["variant"] == variant and row["split"] == split)["physical_action_mse"]
                for variant in VARIANTS
            ]
            positions = [value + (offset - 0.5) * width for value in x]
            axis.bar(positions, values, width, label=label)
        axis.set_xticks(x, labels)
        axis.set_yscale("log")
        axis.set_ylabel("physical action MSE")
        axis.set_title(f"{split}: generation from scratch")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(SUMMARY / "v3_v4_train_test.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    gap_labels = [f"{row['variant'].removeprefix('fm_')}\n{row['version']}" for row in gaps]
    axis.bar(range(len(gaps)), [row["test_over_train_ratio"] for row in gaps], color=["#9ecae9", "#fdae6b"] * 2)
    axis.set_xticks(range(len(gaps)), gap_labels)
    axis.set_yscale("log")
    axis.set_ylabel("test MSE / train MSE (log)")
    axis.set_title("Generation-from-scratch generalization gap")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUMMARY / "v3_v4_generalization_gap.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    colors = {"raw": "#4c78a8", "bspline": "#f58518"}
    for latency in latency_rows:
        test = next(row for row in open_rows if row["version"] == "v4_10k" and row["variant"] == latency["variant"] and row["split"] == "test")
        axis.scatter(latency["sampling_p50_ms_batch1"], test["physical_action_mse"], s=110, color=colors[latency["representation"]], label=latency["representation"])
        axis.annotate(latency["variant"], (latency["sampling_p50_ms_batch1"], test["physical_action_mse"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("12-step FM sampling p50 (ms, batch 1; linear)")
    axis.set_ylabel("test physical action MSE (log)")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    handles, names = axis.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    axis.legend(unique.values(), unique.keys())
    axis.set_title("BSP-UNet v4 10k generation-from-scratch frontier")
    fig.tight_layout()
    fig.savefig(SUMMARY / "performance_vs_latency.png", dpi=180)
    plt.close(fig)

    for split in SPLITS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for axis, variant in zip(axes, VARIANTS):
            rows = [row for row in rtc_rows if row["variant"] == variant and row["split"] == split]
            axis.plot([row["raw_delay_actions"] for row in rows], [row["physical_action_mse"] for row in rows], marker="o")
            axis.set_yscale("log")
            axis.set_title(variant)
            axis.set_xlabel("raw action delay")
            axis.set_ylabel("physical action MSE")
            axis.grid(alpha=0.25)
        fig.suptitle(f"V4 base-FM PiGDM RTC on {split}")
        fig.tight_layout()
        fig.savefig(SUMMARY / f"rtc_{split}_mse_vs_delay.png", dpi=180)
        plt.close(fig)

    def metric(version: str, variant: str, split: str) -> float:
        return next(row["physical_action_mse"] for row in open_rows if row["version"] == version and row["variant"] == variant and row["split"] == split)

    table_en = [
        "| Representation | V3 50k train MSE | V3 50k test MSE | V4 10k train MSE | V4 10k test MSE | V4 test change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        old_test = metric("v3_50k", variant, "test")
        new_test = metric("v4_10k", variant, "test")
        table_en.append(
            f"| {variant.removeprefix('fm_')} | {metric('v3_50k', variant, 'train'):.8f} | {old_test:.8f} | {metric('v4_10k', variant, 'train'):.8f} | {new_test:.8f} | {(new_test / old_test - 1) * 100:+.2f}% |"
        )
    common = [
        "", "Both v4 policies use exactly two observation timesteps and two cameras per timestep (four RGB frames). They are 10k base Flow-Matching checkpoints; RTC evaluation uses binary-hard-mask PiGDM, not ttRTC finetuning.",
        "", "Open-loop generation uses 12-step Euler integration from Gaussian noise over all 25,480 train and 3,149 test windows. RTC uses 128 episode-balanced samples per split and raw delays 0..10.",
    ]
    (SUMMARY / "SUMMARY_EN.md").write_text("\n".join([
        "# BSP-UNet v4 10k Flow-Matching evaluation", "", *table_en, *common,
    ]) + "\n")
    (SUMMARY / "SUMMARY_CN.md").write_text("\n".join([
        "# BSP-UNet v4 10k Flow Matching 评估", "", *table_en, "",
        "两个 v4 policy 都固定使用两个观测时刻、每时刻两个相机，共四帧 RGB 图像。它们是仅训练 10k 的 base Flow-Matching checkpoint；RTC 评估使用 binary-hard-mask PiGDM，不包含 ttRTC finetune。",
        "", "Open-loop 从高斯噪声进行 12 步 Euler 生成，覆盖全部 25,480 个 train windows 与 3,149 个 test windows。RTC 每个 split、每个 delay 使用 128 个 episode-balanced 样本，delay 为 0..10。",
    ]) + "\n")

    summary = {
        "scope": "BSP_UNet_V4 raw/B-spline h2 Flow Matching, 10k base only",
        "observation_contract": {"timesteps": 2, "cameras_per_timestep": 2, "images_per_example": 4},
        "training_updates": 10_000,
        "validation_passed": True,
        "checks": checks,
        "sample_selection": selections,
        "deployment": deployment,
        "v3_v4_open_loop": open_rows,
        "generalization_gaps": gaps,
        "v4_rtc": rtc_rows,
        "v4_latency": latency_rows,
    }
    (SUMMARY / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    info = _publish_wandb(summary, args.wandb_mode)
    if info is not None:
        (SUMMARY / "wandb_evaluation.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"validation_passed": True, "wandb": info}, indent=2))


if __name__ == "__main__":
    main()
