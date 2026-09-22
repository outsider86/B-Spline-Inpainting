#!/usr/bin/env python3
"""Validate and summarize the FM-equivalent DINO joint-DD evaluations."""

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
EXPERIMENT = ROOT / "outputs" / "DINO_DD_JOINT_H12"
SUMMARY = EXPERIMENT / "summary"
VARIANTS = ("raw_h1", "raw_h2", "bspline_h1", "bspline_h2")
STAGES = ("base", "rtc")
SPLITS = ("train", "test")
EXPECTED_JSON = {
    "open_loop_train.json", "open_loop_test.json", "rtc_train.json", "rtc_test.json"
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wandb(summary: dict[str, Any], mode: str) -> dict[str, str] | None:
    if mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        entity="401910710-university-of-california-berkeley",
        project="robot-policy-dino-dd-joint-h12-evaluation",
        name="dino-dd-joint-h12-final-comparison",
        job_type="evaluation",
        mode=mode,
        dir=str((EXPERIMENT / "wandb").resolve()),
        config={
            "architecture": "discrete_joint",
            "variants": list(VARIANTS),
            "splits": list(SPLITS),
            "rtc_samples_per_delay": 128,
        },
    )
    open_columns = list(summary["open_loop"][0])
    rtc_columns = list(summary["rtc"][0])
    latency_columns = list(summary["latency"][0])
    run.log({
        "open_loop_metrics": wandb.Table(
            columns=open_columns,
            data=[[row[column] for column in open_columns] for row in summary["open_loop"]],
        ),
        "rtc_metrics": wandb.Table(
            columns=rtc_columns,
            data=[[row[column] for column in rtc_columns] for row in summary["rtc"]],
        ),
        "latency_metrics": wandb.Table(
            columns=latency_columns,
            data=[[row[column] for column in latency_columns] for row in summary["latency"]],
        ),
        "open_loop_train_test": wandb.Image(str(SUMMARY / "open_loop_train_test.png")),
        "rtc_train_mse_vs_delay": wandb.Image(str(SUMMARY / "rtc_train_mse_vs_delay.png")),
        "rtc_test_mse_vs_delay": wandb.Image(str(SUMMARY / "rtc_test_mse_vs_delay.png")),
        "performance_vs_latency": wandb.Image(str(SUMMARY / "performance_vs_latency.png")),
    })
    artifact = wandb.Artifact("dino-dd-joint-h12-evaluation", type="evaluation")
    for name in (
        "evaluation_summary.json", "open_loop_metrics.csv", "rtc_metrics.csv",
        "latency_metrics.csv", "open_loop_train_test.png",
        "rtc_train_mse_vs_delay.png", "rtc_test_mse_vs_delay.png",
        "performance_vs_latency.png", "SUMMARY_EN.md", "SUMMARY_CN.md",
    ):
        artifact.add_file(str((SUMMARY / name).resolve()))
    artifact.add_file(str((SUMMARY / "deployment" / "websocket_validation.json").resolve()))
    run.log_artifact(artifact)
    result = {"run_id": run.id, "url": run.url, "project": run.project}
    run.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    args = parser.parse_args()
    open_rows: list[dict[str, Any]] = []
    rtc_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    selections: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        for stage in STAGES:
            key = f"{variant}/{stage}"
            folder = SUMMARY / variant / stage
            checkpoint = (EXPERIMENT / "checkpoints" / variant / f"{stage}.pt").resolve()
            files = {path.name for path in folder.glob("*.json")}
            stage_checks: dict[str, Any] = {
                "exactly_four_evaluations": files == EXPECTED_JSON,
                "files": sorted(files),
                "checkpoint_exists": checkpoint.is_file(),
                "inpainting_pngs": all(
                    (folder / f"rtc_inpainting_{split}.png").is_file() for split in SPLITS
                ),
            }
            for split in SPLITS:
                open_report = _read(folder / f"open_loop_{split}.json")
                rtc_report = _read(folder / f"rtc_{split}.json")
                expected_type = "base" if stage == "base" else {"rtc", "ttrtc"}
                source_type = str(open_report["checkpoint_training_type"]).lower()
                type_ok = source_type == expected_type if isinstance(expected_type, str) else source_type in expected_type
                stage_checks[f"{split}_identity"] = (
                    open_report["split"] == split
                    and rtc_report["split"] == split
                    and Path(open_report["checkpoint"]).resolve() == checkpoint
                    and Path(rtc_report["checkpoint"]).resolve() == checkpoint
                    and open_report["architecture"] == "discrete_joint"
                    and rtc_report["architecture"] == "discrete_joint"
                    and type_ok
                )
                stage_checks[f"{split}_generation_from_scratch"] = (
                    open_report["sampling_protocol"]["decoder"]
                    == "deterministic MaskGIT confidence unmasking"
                )
                stage_checks[f"{split}_rtc_method"] = (
                    rtc_report["rtc_inference_method"] == "discrete_hard_mask"
                )
                stage_checks[f"{split}_all_delays"] = (
                    [curve["raw_delay_actions"] for curve in rtc_report["curves"]]
                    == list(range(11))
                )
                selection = rtc_report["sample_selection"]
                if split in selections:
                    stage_checks[f"{split}_cohort_match"] = selection == selections[split]
                else:
                    selections[split] = selection
                    stage_checks[f"{split}_cohort_match"] = True
                fixed_errors = [
                    curve["fixed_control_error"]
                    for curve in rtc_report["curves"]
                    if curve["raw_delay_actions"] > 0
                ]
                stage_checks[f"{split}_hard_mask_exact"] = all(
                    error is not None and error["max_abs"] == 0.0 for error in fixed_errors
                )
                open_rows.append({
                    "checkpoint": key,
                    "variant": variant,
                    "stage": stage,
                    "split": split,
                    "samples": open_report["samples"],
                    "physical_action_mse": open_report["decoded_all_physical_error"]["mse"],
                    "physical_action_mae": open_report["decoded_all_physical_error"]["mae"],
                    "normalized_action_mse": open_report["decoded_all_normalized_error"]["mse"],
                    "token_accuracy": open_report["token_accuracy"],
                })
                for curve in rtc_report["curves"]:
                    rtc_rows.append({
                        "checkpoint": key,
                        "variant": variant,
                        "stage": stage,
                        "split": split,
                        "rtc_method": "discrete_hard_mask",
                        "raw_delay_actions": curve["raw_delay_actions"],
                        "delay_ms": curve["delay_ms"],
                        "affected_spans": curve["affected_spans"],
                        "fixed_rows": curve["committed_control_count"],
                        "samples": rtc_report["samples_per_delay"],
                        "physical_action_mse": curve["decoded_physical_error"]["mse"],
                        "physical_action_mae": curve["decoded_physical_error"]["mae"],
                        "fixed_control_max_abs": (
                            None if curve["fixed_control_error"] is None
                            else curve["fixed_control_error"]["max_abs"]
                        ),
                    })

            latency = _read(SUMMARY / "latency" / variant / f"{stage}.json")
            latency_key = "sampling_rounds-8_use_cache-True_fuse_cache_transition-True"
            equivalence = latency["cache_equivalence"]
            stage_checks["latency_identity"] = Path(latency["checkpoint"]).resolve() == checkpoint
            stage_checks["kv_cache_exact"] = all(
                equivalence[name]
                for name in (
                    "uncached_vs_legacy_tokens_equal",
                    "uncached_vs_fused_tokens_equal",
                    "legacy_vs_fused_tokens_equal",
                )
            )
            latency_rows.append({
                "checkpoint": key,
                "variant": variant,
                "stage": stage,
                "representation": "bspline" if variant.startswith("bspline") else "raw",
                "observation_horizon": 2 if variant.endswith("h2") else 1,
                "sampling_p50_ms_batch1": latency[latency_key]["p50_ms"],
                "sampling_p95_ms_batch1": latency[latency_key]["p95_ms"],
                "online_vision_p50_ms": latency["online_vision_total"]["p50_ms"],
                "peak_memory_bytes": latency["peak_memory_bytes"],
            })
            checks[key] = stage_checks

    failures = {
        key: {name: value for name, value in values.items() if isinstance(value, bool) and not value}
        for key, values in checks.items()
    }
    failures = {key: values for key, values in failures.items() if values}
    if failures:
        raise RuntimeError(f"evaluation validation failed: {failures}")

    deployment = _read(SUMMARY / "deployment" / "websocket_validation.json")
    if not deployment.get("passed") or deployment.get("server_count") != 8:
        raise RuntimeError(f"deployment validation failed: {deployment}")
    summary = {
        "scope": "eight DINOv2 joint discrete-diffusion checkpoints; base and finetuned ttRTC",
        "required_evaluations_per_checkpoint": [
            "generation_from_scratch_train", "generation_from_scratch_test",
            "rtc_train", "rtc_test", "latency",
        ],
        "open_loop_protocol": {
            "generation": "8-round deterministic confidence MaskGIT from all MASK tokens",
            "train_samples": next(row["samples"] for row in open_rows if row["split"] == "train"),
            "test_samples": next(row["samples"] for row in open_rows if row["split"] == "test"),
        },
        "rtc_protocol": {
            "samples_per_split_per_delay": 128,
            "sample_selection": selections,
            "raw_action_delays": list(range(11)),
            "condition_source": "previous generated chunk shifted to current observation time",
            "base": "native categorical hard-mask inpainting",
            "finetuned_ttrtc": "same hard-mask sampler after delay-conditioned finetuning",
            "bspline_fixed_rows": "ceil(raw delay / 2) affected spans + 3 cubic support rows",
        },
        "validation_passed": True,
        "deployment_validation": deployment,
        "checks": checks,
        "open_loop": open_rows,
        "rtc": rtc_rows,
        "latency": latency_rows,
    }
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(SUMMARY / "open_loop_metrics.csv", open_rows)
    _write_csv(SUMMARY / "rtc_metrics.csv", rtc_rows)
    _write_csv(SUMMARY / "latency_metrics.csv", latency_rows)

    labels = [variant for variant in VARIANTS]
    x = list(range(len(VARIANTS)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, split in zip(axes, SPLITS):
        base = [next(row for row in open_rows if row["variant"] == variant and row["stage"] == "base" and row["split"] == split)["physical_action_mse"] for variant in VARIANTS]
        rtc = [next(row for row in open_rows if row["variant"] == variant and row["stage"] == "rtc" and row["split"] == split)["physical_action_mse"] for variant in VARIANTS]
        axis.bar([value - width / 2 for value in x], base, width, label="base")
        axis.bar([value + width / 2 for value in x], rtc, width, label="finetuned ttRTC")
        axis.set_xticks(x, labels, rotation=15)
        axis.set_yscale("log")
        axis.set_ylabel("physical action MSE")
        axis.set_title(f"{split}: generation from scratch")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(SUMMARY / "open_loop_train_test.png", dpi=180)
    plt.close(fig)

    for split in SPLITS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for axis, variant in zip(axes.flat, VARIANTS):
            for stage, label in (("base", "base hard mask"), ("rtc", "finetuned ttRTC")):
                selected = [row for row in rtc_rows if row["variant"] == variant and row["stage"] == stage and row["split"] == split]
                axis.plot([row["raw_delay_actions"] for row in selected], [row["physical_action_mse"] for row in selected], marker="o", label=label)
            axis.set_yscale("log")
            axis.set_title(variant)
            axis.set_xlabel("raw action delay")
            axis.set_ylabel("physical action MSE")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(f"Discrete RTC replay on {split} split")
        fig.tight_layout()
        fig.savefig(SUMMARY / f"rtc_{split}_mse_vs_delay.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"raw": "#4c78a8", "bspline": "#f58518"}
    for horizon, axis in zip((1, 2), axes):
        for latency in latency_rows:
            if latency["stage"] != "base" or latency["observation_horizon"] != horizon:
                continue
            open_row = next(row for row in open_rows if row["variant"] == latency["variant"] and row["stage"] == "base" and row["split"] == "test")
            axis.scatter(latency["sampling_p50_ms_batch1"], open_row["physical_action_mse"], color=colors[latency["representation"]], s=100, label=latency["representation"])
            axis.annotate(latency["variant"], (latency["sampling_p50_ms_batch1"], open_row["physical_action_mse"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        handles, names = axis.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        axis.legend(unique.values(), unique.keys())
        axis.set_title(f"h{horizon}")
        axis.set_xlabel("cached sampling p50 (ms, batch 1; linear)")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("test physical action MSE (log)")
    fig.suptitle("DINOv2 joint-DD generation-from-scratch latency frontier")
    fig.tight_layout()
    fig.savefig(SUMMARY / "performance_vs_latency.png", dpi=180)
    plt.close(fig)

    ordered = sorted(open_rows, key=lambda row: (row["split"], row["physical_action_mse"]))
    table = [
        "| Variant | Stage | Split | Physical action MSE | Token accuracy |",
        "|---|---|---|---:|---:|",
        *[
            f"| `{row['variant']}` | {row['stage']} | {row['split']} | {row['physical_action_mse']:.8f} | {row['token_accuracy']:.6f} |"
            for row in ordered
        ],
    ]
    (SUMMARY / "SUMMARY_EN.md").write_text("\n".join([
        "# DINOv2 joint discrete-diffusion evaluation", "",
        "Every checkpoint has full train/test generation-from-scratch evaluation, 128-sample train/test RTC delay curves, dataset-min/max-aligned prefix-inpainting PNGs, batch-1 latency, and exact KV-cache equivalence checks.", "",
        *table, "",
        "Base and ttRTC use identical categorical hard-mask inference. ttRTC differs only by delay-conditioned finetuning from the exact base-parent prediction cache.",
        "All eight checkpoints also passed the real Piper-compatible WebSocket standard/RTC inference audit.",
    ]) + "\n")
    (SUMMARY / "SUMMARY_CN.md").write_text("\n".join([
        "# DINOv2 joint 离散扩散评估", "",
        "每个 checkpoint 都包含完整 train/test 从零生成评估、每个 delay 128 个样本的 train/test RTC 曲线、按全数据集 min/max 对齐坐标轴的 prefix-inpainting PNG、batch-1 延迟，以及严格的 KV-cache 等价性检查。", "",
        *table, "",
        "Base 与 ttRTC 使用完全相同的 categorical hard-mask 推理；ttRTC 的唯一区别是使用对应 base checkpoint 的精确 parent prediction cache 进行 delay-conditioned finetune。",
        "全部 8 个 checkpoint 也通过了真实 Piper-compatible WebSocket 标准推理与 RTC 推理审计。",
    ]) + "\n")

    info = _wandb(summary, args.wandb_mode)
    if info is not None:
        (SUMMARY / "wandb_evaluation.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"validation_passed": True, "checkpoints": len(checks), "wandb": info}, indent=2))


if __name__ == "__main__":
    main()
