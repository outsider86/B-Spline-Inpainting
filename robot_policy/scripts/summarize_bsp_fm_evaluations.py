#!/usr/bin/env python3
"""Validate and summarize the four evaluations for every BSP FM checkpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ("fm_raw_h1", "fm_raw_h2", "fm_bspline_h1", "fm_bspline_h2")
STAGES = ("base", "rtc")
SPLITS = ("train", "test")
EXPECTED_FILES = {
    "open_loop_train.json",
    "open_loop_test.json",
    "rtc_train.json",
    "rtc_test.json",
}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "outputs" / "BSP_UNET_V3" / "summary"
    checkpoint_root = project / "outputs" / "BSP_UNET_V3" / "checkpoints"
    open_rows: list[dict[str, object]] = []
    rtc_rows: list[dict[str, object]] = []
    visualization_rows: list[dict[str, object]] = []
    checks: dict[str, dict[str, object]] = {}

    for variant in VARIANTS:
        for stage in STAGES:
            key = f"{variant}/{stage}"
            folder = root / variant / stage
            files = {path.name for path in folder.glob("*.json")}
            checkpoint = (checkpoint_root / variant / f"{stage}.pt").resolve()
            stage_checks: dict[str, object] = {
                "exactly_four_evaluations": files == EXPECTED_FILES,
                "files": sorted(files),
                "checkpoint_exists": checkpoint.is_file(),
            }
            for split in SPLITS:
                open_report = json.loads((folder / f"open_loop_{split}.json").read_text())
                rtc_report = json.loads((folder / f"rtc_{split}.json").read_text())
                expected_method = (
                    "pigdm_binary_hard_mask"
                    if stage == "base"
                    else "finetuned_ttrtc_direct_hard_mask"
                )
                source_type = str(open_report["checkpoint_training_type"]).lower()
                stage_checks[f"{split}_split_match"] = (
                    open_report["split"] == split and rtc_report["split"] == split
                )
                stage_checks[f"{split}_checkpoint_match"] = (
                    Path(open_report["checkpoint"]).resolve() == checkpoint
                    and Path(rtc_report["checkpoint"]).resolve() == checkpoint
                )
                stage_checks[f"{split}_training_type"] = (
                    source_type == "base"
                    if stage == "base"
                    else source_type in {"rtc", "ttrtc"}
                )
                stage_checks[f"{split}_rtc_method"] = (
                    rtc_report["rtc_inference_method"] == expected_method
                )
                stage_checks[f"{split}_all_delays"] = (
                    [curve["raw_delay_actions"] for curve in rtc_report["curves"]]
                    == list(range(11))
                )
                image_path = folder / f"rtc_inpainting_{split}.png"
                stage_checks[f"{split}_inpainting_png"] = image_path.is_file()
                visualization_rows.append(
                    {
                        "checkpoint": key,
                        "split": split,
                        "raw_delay_actions": 6,
                        "condition_source": "previous generated chunk",
                        "path": str(image_path.resolve()),
                    }
                )
                open_rows.append(
                    {
                        "checkpoint": key,
                        "variant": variant,
                        "stage": stage,
                        "split": split,
                        "samples": open_report["samples"],
                        "physical_action_mse": open_report["decoded_all_physical_error"]["mse"],
                        "physical_action_mae": open_report["decoded_all_physical_error"]["mae"],
                        "normalized_action_mse": open_report["decoded_all_normalized_error"]["mse"],
                    }
                )
                for curve in rtc_report["curves"]:
                    error = curve["decoded_physical_error"]
                    rtc_rows.append(
                        {
                            "checkpoint": key,
                            "variant": variant,
                            "stage": stage,
                            "split": split,
                            "rtc_method": expected_method,
                            "raw_delay_actions": curve["raw_delay_actions"],
                            "delay_ms": curve["delay_ms"],
                            "affected_spans": curve["affected_spans"],
                            "fixed_rows": curve["committed_control_count"],
                            "samples": rtc_report["samples_per_delay"],
                            "physical_action_mse": error["mse"],
                            "physical_action_mae": error["mae"],
                            "fixed_control_max_abs": (
                                None
                                if curve["fixed_control_error"] is None
                                else curve["fixed_control_error"]["max_abs"]
                            ),
                        }
                    )
            checks[key] = stage_checks

    failures = {
        checkpoint: {name: value for name, value in values.items() if isinstance(value, bool) and not value}
        for checkpoint, values in checks.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise RuntimeError(f"evaluation validation failed: {failures}")

    summary = {
        "scope": "eight BSP-UNet flow-matching checkpoints; base and finetuned ttRTC",
        "required_evaluations_per_checkpoint": [
            "generation_from_scratch_train",
            "generation_from_scratch_test",
            "rtc_train",
            "rtc_test",
        ],
        "open_loop_protocol": {
            "train_samples": 25480,
            "test_samples": 3149,
            "generation": "12-step Euler integration from Gaussian noise",
        },
        "rtc_protocol": {
            "samples_per_split_per_delay": 128,
            "raw_action_delays": list(range(11)),
            "condition_source": "previous generated chunk shifted to current observation time",
            "base": "PiGDM with binary hard mask",
            "finetuned_ttrtc": "direct binary hard mask without PiGDM",
            "bspline_fixed_rows": "ceil(raw delay / 2) affected spans + 3 cubic support rows",
        },
        "validation_passed": True,
        "checks": checks,
        "open_loop": open_rows,
        "rtc": rtc_rows,
        "rtc_inpainting_visualizations": visualization_rows,
    }
    (root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for filename, rows in (
        ("open_loop_metrics.csv", open_rows),
        ("rtc_metrics.csv", rtc_rows),
    ):
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = [variant.removeprefix("fm_") for variant in VARIANTS]
    x = list(range(len(VARIANTS)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for axis, split in zip(axes, SPLITS):
        base = [
            next(row for row in open_rows if row["variant"] == variant and row["stage"] == "base" and row["split"] == split)["physical_action_mse"]
            for variant in VARIANTS
        ]
        ttrtc = [
            next(row for row in open_rows if row["variant"] == variant and row["stage"] == "rtc" and row["split"] == split)["physical_action_mse"]
            for variant in VARIANTS
        ]
        axis.bar([value - width / 2 for value in x], base, width, label="base")
        axis.bar([value + width / 2 for value in x], ttrtc, width, label="finetuned ttRTC")
        axis.set_xticks(x, labels)
        axis.set_yscale("log")
        axis.set_ylabel("physical action MSE")
        axis.set_title(f"{split}: generation from scratch")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / "open_loop_train_test.png", dpi=180)
    plt.close(fig)

    for split in SPLITS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for axis, variant in zip(axes.flat, VARIANTS):
            for stage, label in (("base", "base PiGDM"), ("rtc", "finetuned ttRTC")):
                selected = [
                    row for row in rtc_rows
                    if row["variant"] == variant and row["stage"] == stage and row["split"] == split
                ]
                axis.plot(
                    [row["raw_delay_actions"] for row in selected],
                    [row["physical_action_mse"] for row in selected],
                    marker="o",
                    label=label,
                )
            axis.set_yscale("log")
            axis.set_title(variant.removeprefix("fm_"))
            axis.set_xlabel("raw action delay")
            axis.set_ylabel("physical action MSE")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(f"RTC replay on {split} split")
        fig.tight_layout()
        fig.savefig(root / f"rtc_{split}_mse_vs_delay.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
