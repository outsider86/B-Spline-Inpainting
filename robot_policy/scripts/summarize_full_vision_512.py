#!/usr/bin/env python3
"""Aggregate the 16-checkpoint active full-vision experiment into reports and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from robot_policy.config import ACTIVE_ARCHITECTURES


SIZES = (("DiT-S", "dit_s"), ("DiT-B", "dit_b"))
REPRESENTATIONS = ("raw", "bspline")
STAGES = ("base", "ttrtc")


def read(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pct(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def latency_key(architecture: str) -> str:
    return (
        "sampling_steps-12" if architecture == "fm"
        else "sampling_rounds-8_use_cache-True_fuse_cache_transition-True"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join((
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ))


def log_wandb(summary: Path, result: dict, records: list[dict], *, entity: str,
              project: str, mode: str) -> dict[str, str]:
    """Publish the complete 16-checkpoint comparison as one W&B evaluation run."""
    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        name="dit-s-b-512vision-fm-vs-joint-dd",
        job_type="evaluation",
        mode=mode,
        config=result["protocol"],
    )
    columns = list(records[0])
    table = wandb.Table(
        columns=columns,
        data=[[record[column] for column in columns] for record in records],
    )
    payload = {
        "checkpoint_comparison": table,
        "performance_vs_latency": wandb.Image(str(summary / "performance_vs_latency.png")),
        "validation_vs_test_generation": wandb.Image(str(summary / "validation_vs_test_generation.png")),
        **{f"findings/{key}": value for key, value in result["findings"].items()
           if isinstance(value, (int, float, str, bool))},
    }
    for record in records:
        prefix = "/".join((
            record["model_size"], record["representation"],
            record["architecture"], record["stage"],
        ))
        payload[f"{prefix}/test_physical_action_mse"] = record["test_decoded_physical_mse"]
        payload[f"{prefix}/validation_generation_action_mse"] = record["final_validation_generation_action_mse_normalized"]
        payload[f"{prefix}/sampling_p50_ms"] = record["sampling_p50_ms"]
    run.log(payload)
    artifact = wandb.Artifact("full-vision-512-comparison", type="evaluation")
    for filename in (
        "summary.json", "checkpoint_metrics.csv", "ranking_by_physical_mse.csv",
        "pairwise_effects.csv", "performance_vs_latency.png",
        "validation_vs_test_generation.png", "SUMMARY_EN.md", "SUMMARY_CN.md",
    ):
        artifact.add_file(str((summary / filename).resolve()))
    run.log_artifact(artifact)
    info = {
        "entity": str(run.entity), "project": str(run.project),
        "run_id": str(run.id), "url": str(run.url), "state": "finished",
    }
    run.finish()
    (summary / "wandb_evaluation.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/FULL_VISION_512"))
    parser.add_argument("--wandb-entity", default="401910710-university-of-california-berkeley")
    parser.add_argument("--wandb-project", default="robot-policy-full-vision-512-evaluation")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args()
    root = args.root.resolve()
    summary = root / "summary"
    audit = read(root / "completion_audit.json")
    if audit["status"] != "passed":
        raise RuntimeError("completion audit must pass before summary generation")
    deployment = {
        size: read(summary / f"deployment_validation_{size}.json")
        for _, size in SIZES
    }
    invalid_deployment = [
        size for size, report in deployment.items()
        if not report.get("passed")
        or report.get("checkpoint_inventory_count") != 16
        or report.get("runtime_case_count") != 8
        or report.get("vision_encoder_output_shape") != [2, 256, 2176]
    ]
    if invalid_deployment:
        raise RuntimeError(f"deployment validation did not pass for {invalid_deployment}")
    rtc_summary = read(summary / "inference_rtc" / "summary.json")
    inference_index = {}
    for report in rtc_summary["reports"]:
        curve = next(item for item in report["curves"] if item["raw_prefix_actions"] == rtc_summary["primary_prefix"])
        report_stage = "ttrtc" if report["training_type"] in {"rtc", "ttrtc"} else report["training_type"]
        inference_index[(report["model_size"], report["action_representation"], report["architecture"], report_stage)] = curve

    records = []
    for model_size, size in SIZES:
        for representation in REPRESENTATIONS:
            for architecture in ACTIVE_ARCHITECTURES:
                for stage in STAGES:
                    filename = f"{architecture}_{stage}.json"
                    evaluation = read(summary / "open_loop" / size / representation / filename)
                    latency = read(summary / "latency" / size / representation / filename)
                    rtc = read(summary / "rtc" / size / representation / filename)
                    checkpoint = root / size / representation / "checkpoints" / filename.replace(".json", ".pt")
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    source_type = str(payload["training_type"])
                    normalized_stage = "ttrtc" if source_type in {"rtc", "ttrtc"} else source_type
                    curve = inference_index[(model_size, representation, architecture, normalized_stage)]
                    sample = latency[latency_key(architecture)]
                    gradients = [float(item["grad_norm"]) for item in payload["loss_trace"]]
                    applied_gradients = [
                        float(item["grad_norm"]) for item in payload["loss_trace"]
                        if not bool(item.get("optimizer_step_skipped", 0.0))
                    ]
                    final_validation = payload["history"][-1]["validation"]
                    record = {
                        "model_size": model_size,
                        "representation": representation,
                        "architecture": architecture,
                        "stage": stage,
                        "vision_tokens": 2 * payload["config"]["vision"]["pooled_grid"]**2,
                        "parameters": payload["parameter_counts"]["total_trainable"],
                        "learning_rate": (
                            payload["config"]["train"].get("rtc_learning_rate")
                            if stage == "ttrtc"
                            else payload["config"]["train"]["learning_rate"]
                        ),
                        "fm_math_sdp_training": (
                            payload["config"]["train"].get("fm_math_sdp_training")
                            if architecture == "fm"
                            else None
                        ),
                        "final_train_action_mse_normalized": payload["loss_trace"][-1]["action_mse"],
                        "final_validation_generation_action_mse_normalized": final_validation["generation_action_mse"],
                        "best_validation_generation_action_mse_normalized": payload["best_validation"],
                        "max_gradient_norm_before_clip": max(gradients),
                        "max_applied_gradient_norm_before_clip": max(applied_gradients),
                        "skipped_optimizer_updates": sum(
                            int(bool(item.get("optimizer_step_skipped", 0.0)))
                            for item in payload["loss_trace"]
                        ),
                        "test_decoded_normalized_mse": evaluation["decoded_all_normalized_error"]["mse"],
                        "test_decoded_physical_mse": evaluation["decoded_all_physical_error"]["mse"],
                        "test_decoded_physical_mae": evaluation["decoded_all_physical_error"]["mae"],
                        "token_accuracy": evaluation["token_accuracy"],
                        "sampling_p50_ms": sample["p50_ms"],
                        "sampling_p95_ms": sample["p95_ms"],
                        "online_vision_p50_ms": latency["online_vision_total"]["p50_ms"],
                        "projector_state_p50_ms": latency["projector_state"]["p50_ms"],
                        "action_decode_p50_ms": latency["action_decode"]["p50_ms"],
                        "estimated_online_pipeline_p50_ms": latency["online_vision_total"]["p50_ms"] + latency["projector_state"]["p50_ms"] + sample["p50_ms"] + latency["action_decode"]["p50_ms"],
                        "peak_memory_bytes": latency["peak_memory_bytes"],
                        "delay_mean_physical_mse_d0_d10": float(np.mean([item["decoded_physical_error"]["mse"] for item in rtc["curves"]])),
                        "inference_rtc_suffix_physical_mse_prefix6": curve["suffix_physical_mse"],
                        "inference_rtc_fixed_control_max_abs": curve["fixed_control_max_abs"],
                        "checkpoint": str(checkpoint.resolve()),
                        "wandb_url": payload["wandb"]["url"],
                    }
                    if architecture == "discrete_joint":
                        equivalence = latency["cache_equivalence"]
                        record["joint_legacy_vs_fused_tokens_equal"] = equivalence["legacy_vs_fused_tokens_equal"]
                        legacy = latency["sampling_rounds-8_use_cache-True_fuse_cache_transition-False"]["p50_ms"]
                        record["joint_fused_cache_speedup_percent"] = pct(legacy, sample["p50_ms"])
                    else:
                        record["joint_legacy_vs_fused_tokens_equal"] = None
                        record["joint_fused_cache_speedup_percent"] = None
                    records.append(record)

    if len(records) != 16:
        raise RuntimeError(f"expected 16 summary records, got {len(records)}")
    write_csv(summary / "checkpoint_metrics.csv", records)
    ranked = sorted(records, key=lambda item: item["test_decoded_physical_mse"])
    write_csv(summary / "ranking_by_physical_mse.csv", ranked)
    lookup = {(r["model_size"], r["representation"], r["architecture"], r["stage"]): r for r in records}
    pairs = []
    for model_size, _ in SIZES:
        for architecture in ACTIVE_ARCHITECTURES:
            for stage in STAGES:
                raw = lookup[(model_size, "raw", architecture, stage)]
                spline = lookup[(model_size, "bspline", architecture, stage)]
                pairs.append({
                    "comparison": "bspline_vs_raw", "model_size": model_size,
                    "architecture": architecture, "stage": stage, "representation": "bspline",
                    "physical_mse_reduction_percent": pct(raw["test_decoded_physical_mse"], spline["test_decoded_physical_mse"]),
                    "latency_reduction_percent": pct(raw["sampling_p50_ms"], spline["sampling_p50_ms"]),
                })
    for model_size, _ in SIZES:
        for representation in REPRESENTATIONS:
            for architecture in ACTIVE_ARCHITECTURES:
                base = lookup[(model_size, representation, architecture, "base")]
                child = lookup[(model_size, representation, architecture, "ttrtc")]
                pairs.append({
                    "comparison": "ttrtc_vs_base", "model_size": model_size,
                    "architecture": architecture, "stage": "ttrtc", "representation": representation,
                    "physical_mse_reduction_percent": pct(base["test_decoded_physical_mse"], child["test_decoded_physical_mse"]),
                    "latency_reduction_percent": pct(base["sampling_p50_ms"], child["sampling_p50_ms"]),
                })
    write_csv(summary / "pairwise_effects.csv", pairs)

    colors = {"raw": "#4c78a8", "bspline": "#f58518"}
    markers = {"fm": "o", "discrete_joint": "^"}
    fig, axis = plt.subplots(figsize=(11, 7), layout="constrained")
    for record in (item for item in records if item["stage"] == "base"):
        axis.scatter(
            record["sampling_p50_ms"], record["test_decoded_physical_mse"],
            color=colors[record["representation"]], marker=markers[record["architecture"]],
            s={"DiT-S": 50, "DiT-B": 90}[record["model_size"]], alpha=.82,
        )
    axis.set(
        xlabel="default policy sampling p50 (ms, batch 1)",
        ylabel="from-scratch generated physical action MSE",
        yscale="log",
        title="512-token base-policy from-scratch generation / latency frontier",
    )
    axis.grid(alpha=.25, which="both")
    axis.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=colors["raw"], label="raw"),
        Line2D([], [], marker="o", linestyle="", color=colors["bspline"], label="B-spline"),
        Line2D([], [], marker="o", linestyle="", color="black", label="FM"),
        Line2D([], [], marker="^", linestyle="", color="black", label="joint DD"),
        Line2D([], [], marker="o", linestyle="", color="gray", markersize=7, label="DiT-S"),
        Line2D([], [], marker="o", linestyle="", color="gray", markersize=10, label="DiT-B"),
    ])
    fig.savefig(summary / "performance_vs_latency.png", dpi=180)
    fig.savefig(summary / "latency_vs_generation_from_scratch.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    for record in records:
        axis.scatter(record["final_validation_generation_action_mse_normalized"], record["test_decoded_normalized_mse"],
                     color=colors[record["representation"]], marker=markers[record["architecture"]], alpha=.8)
    low = min(min(r["final_validation_generation_action_mse_normalized"], r["test_decoded_normalized_mse"]) for r in records)
    high = max(max(r["final_validation_generation_action_mse_normalized"], r["test_decoded_normalized_mse"]) for r in records)
    axis.plot([low, high], [low, high], color="gray", linestyle="--")
    axis.set(xlabel="final from-scratch validation action MSE", ylabel="full-test open-loop normalized MSE",
             xscale="log", yscale="log", title="Validation generation metric vs held-out generation")
    axis.grid(alpha=.25, which="both")
    fig.savefig(summary / "validation_vs_test_generation.png", dpi=180); plt.close(fig)

    spline_pairs = [item for item in pairs if item["comparison"] == "bspline_vs_raw"]
    rtc_pairs = [item for item in pairs if item["comparison"] == "ttrtc_vs_base"]
    joint = [item for item in records if item["architecture"] == "discrete_joint"]
    findings = {
        "best": {key: ranked[0][key] for key in ("model_size", "representation", "architecture", "stage", "test_decoded_physical_mse", "sampling_p50_ms")},
        "bspline_wins": sum(item["physical_mse_reduction_percent"] > 0 for item in spline_pairs),
        "bspline_pairs": len(spline_pairs),
        "bspline_median_reduction_percent": median(item["physical_mse_reduction_percent"] for item in spline_pairs),
        "ttrtc_wins": sum(item["physical_mse_reduction_percent"] > 0 for item in rtc_pairs),
        "ttrtc_pairs": len(rtc_pairs),
        "ttrtc_median_reduction_percent": median(item["physical_mse_reduction_percent"] for item in rtc_pairs),
        "joint_cache_exact": sum(bool(item["joint_legacy_vs_fused_tokens_equal"]) for item in joint),
        "joint_cache_comparisons": len(joint),
        "fm_skipped_optimizer_updates": sum(
            item["skipped_optimizer_updates"] for item in records
            if item["architecture"] == "fm"
        ),
    }
    result = {
        "protocol": {
            "checkpoints": 16, "vision_tokens": 512, "tokens_per_camera": 256,
            "training": "50,000 base + 5,000 ttRTC updates, effective batch 32",
            "validation": "from-scratch generation using vision+state only; fixed 128 validation examples per checkpoint",
            "evaluation": "open-loop full test, delay RTC, oracle-prefix inference RTC, batch-1 latency",
            "rtceval_plot_axes": "shared log physical-MSE limits per DiT size across raw/B-spline and shared full-dataset physical trajectory min/max",
        },
        "findings": findings,
        "completion_audit": {
            "status": audit["status"],
            "checkpoint_count": audit["checkpoint_count"],
            "expected_checkpoint_count": audit["expected_checkpoint_count"],
            "active_architectures": audit["active_architectures"],
            "vision": audit["vision"],
            "validation_contract": audit["validation_contract"],
        },
        "deployment_validation": deployment,
        "decoder_validation": read(summary / "decoder_validation" / "decode_validation.json"),
        "records": records,
        "pairwise_effects": pairs,
    }
    (summary / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    table = markdown_table(
        ["size", "repr.", "policy", "stage", "val gen MSE", "test physical MSE", "p50 ms", "W&B"],
        [[r["model_size"], r["representation"], r["architecture"], r["stage"],
          f"{r['final_validation_generation_action_mse_normalized']:.6f}", f"{r['test_decoded_physical_mse']:.6f}",
          f"{r['sampling_p50_ms']:.2f}", f"[run]({r['wandb_url']})"] for r in records],
    )
    english = f"""# Full-Vision FM and Joint-DD Results

## Protocol

- 16 checkpoints: DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC.
- Every observation contains two complete 16×16 patch grids: **512 vision tokens**, plus one state token.
- Base training uses 50,000 updates; ttRTC uses 5,000 updates; effective batch size is 32.
- `validation/action_mse` is decoded generation from scratch. The sampler receives only vision and state; it never receives target actions, corrupted ground truth, or teacher assistance.
- Evaluation includes full held-out open-loop generation, delay RTC, oracle-prefix inference RTC, and synchronized batch-1 latency.
- Within each DiT size, raw and B-spline RTCEVAL metric plots share one log physical-MSE y-axis; all trajectory plots share the same full-dataset physical min/max for each action dimension.

## Findings

- Best decoded physical MSE: **{ranked[0]['test_decoded_physical_mse']:.6f}** ({ranked[0]['model_size']} / {ranked[0]['representation']} / {ranked[0]['architecture']} / {ranked[0]['stage']}).
- B-spline wins {findings['bspline_wins']}/{findings['bspline_pairs']} matched comparisons; median physical-MSE reduction is {findings['bspline_median_reduction_percent']:.2f}%.
- ttRTC wins {findings['ttrtc_wins']}/{findings['ttrtc_pairs']} matching zero-delay open-loop comparisons; median reduction is {findings['ttrtc_median_reduction_percent']:.2f}%.
- Joint cached/fused decoding is token-identical in {findings['joint_cache_exact']}/{findings['joint_cache_comparisons']} checkpoints.
- FM rejected {findings['fm_skipped_optimizer_updates']} anomalous optimizer updates after the 1,000-update warm-up; every rejected step remains recorded with its pre-clip norm.
- Deployment validation passes all 8 runtime cases for each of DiT-S and DiT-B; each audit inventories all 16 checkpoints and produces finite 30×7 actions from two 256-token camera streams.

## Evidence

- `completion_audit.json`: checkpoint identity, exact update counts, hashes, parent lineage, finite gradients, from-scratch validation, all four evaluation families, and local/remote W&B state.
- `summary/deployment_validation_dit_{{s,b}}.json`: real-GPU load and inference for every raw/B-spline × FM/joint-DD × base/ttRTC case.
- `summary/checkpoint_metrics.csv`: complete 16-row metric table; `ranking_by_physical_mse.csv` and `pairwise_effects.csv`: rankings and matched effects.
- Results are open-loop action-generation diagnostics, not closed-loop task-success measurements.

## All checkpoints

{table}
"""
    chinese = f"""# 全视觉 Token FM 与 Joint-DD 结果

## 实验协议

- 共 16 个检查点：DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC。
- 每个观测保留两个完整的 16×16 patch 网格，即 **512 个视觉 token**，并追加 1 个状态 token。
- 基础训练 50,000 次更新，ttRTC 训练 5,000 次更新，有效 batch size 为 32。
- `validation/action_mse` 来自从零开始的完整生成。采样器只接收视觉与状态，不接收目标动作、被扰动的真值或教师辅助信息。
- 后续评测包含完整测试集开环生成、延迟 RTC、oracle-prefix inference RTC 和 batch-1 同步延迟。
- 在每个 DiT 尺寸内，raw 与 B-spline 的 RTCEVAL 指标图共享同一组对数物理 MSE 纵轴范围；全部轨迹图按动作维度共享完整数据集的物理最小值/最大值。

## 主要结果

- 最佳物理 MSE 为 **{ranked[0]['test_decoded_physical_mse']:.6f}**（{ranked[0]['model_size']} / {ranked[0]['representation']} / {ranked[0]['architecture']} / {ranked[0]['stage']}）。
- 在严格配对中，B-spline 有 {findings['bspline_wins']}/{findings['bspline_pairs']} 组更优；物理 MSE 降幅中位数为 {findings['bspline_median_reduction_percent']:.2f}%。
- ttRTC 在零延迟开环对比中有 {findings['ttrtc_wins']}/{findings['ttrtc_pairs']} 组更优；降幅中位数为 {findings['ttrtc_median_reduction_percent']:.2f}%。
- Joint cache 融合解码在 {findings['joint_cache_exact']}/{findings['joint_cache_comparisons']} 个检查点上逐 token 一致。
- FM 在 1,000 次 warm-up 更新之后共拒绝了 {findings['fm_skipped_optimizer_updates']} 次异常优化器更新；每次被拒绝的更新及其裁剪前梯度范数都完整保留在记录中。
- DiT-S、DiT-B 的部署验证均通过全部 8 个运行时用例；每份审计都清点全部 16 个检查点，并从两路各 256-token 的相机输入生成有限的 30×7 动作。

## 证据

- `completion_audit.json`：检查点身份、精确更新数、哈希、父模型谱系、有限梯度、从零验证、四类评测以及本地/远端 W&B 状态。
- `summary/deployment_validation_dit_{{s,b}}.json`：对全部 raw/B-spline × FM/joint-DD × base/ttRTC 组合执行真实 GPU 加载与推理。
- `summary/checkpoint_metrics.csv`：完整 16 行指标；`ranking_by_physical_mse.csv` 与 `pairwise_effects.csv`：排名和严格配对效应。
- 这些结果属于开环动作生成诊断，不代表闭环任务成功率。

## 全部检查点

{table}
"""
    (summary / "SUMMARY_EN.md").write_text(english)
    (summary / "SUMMARY_CN.md").write_text(chinese)
    codex_reports = Path(__file__).resolve().parents[2] / "CodexDoc" / "reports"
    codex_reports.mkdir(parents=True, exist_ok=True)
    (codex_reports / "FULL_VISION_512_16_EN.md").write_text(english)
    (codex_reports / "FULL_VISION_512_16_CN.md").write_text(chinese)
    wandb_info = None
    if args.wandb_mode != "disabled":
        wandb_info = log_wandb(
            summary, result, records, entity=args.wandb_entity,
            project=args.wandb_project, mode=args.wandb_mode,
        )
    print(json.dumps({"records": len(records), "findings": findings, "wandb": wandb_info}, indent=2))


if __name__ == "__main__":
    main()
