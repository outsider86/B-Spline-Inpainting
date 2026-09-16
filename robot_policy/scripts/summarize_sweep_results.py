#!/usr/bin/env python3
"""Aggregate all sweep evaluation artifacts into CSV/JSON/plots/EN+CN reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SIZES = (("DiT-S", "dit_s"), ("DiT-B", "dit_b"), ("DiT-L", "dit_l"))
ARCHITECTURES = ("fm", "discrete_layerwise", "discrete_joint")
STAGES = ("base", "ttrtc")
REPRESENTATIONS = ("raw", "bspline")


def read(path: Path):
    return json.loads(path.read_text())


def default_latency_key(architecture: str) -> str:
    if architecture == "fm":
        return "sampling_steps-12"
    if architecture == "discrete_layerwise":
        return "sampling_rounds-8"
    return "sampling_rounds-8_use_cache-True_fuse_cache_transition-True"


def baseline_latency_key(architecture: str) -> str:
    return "sampling_rounds-8_use_cache-True" if architecture == "discrete_joint" else default_latency_key(architecture)


def pct_reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join((
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = root / "outputs" / "SWEEP" / "summary"
    audit = read(root / "outputs" / "SWEEP" / "completion_audit.json")
    audit_index = {(x["model_size"], x["representation"], x["architecture"], x["stage"]): x for x in audit["records"]}
    records = []
    for model_size, size_dir in SIZES:
        for representation in REPRESENTATIONS:
            for architecture in ARCHITECTURES:
                for stage in STAGES:
                    name = f"{architecture}_{stage}.json"
                    evaluation = read(summary / "open_loop" / size_dir / representation / name)
                    latency = read(summary / "latency" / size_dir / representation / name)
                    baseline_latency = read(summary / "latency_baseline" / size_dir / representation / name)
                    rtc_path = summary / "rtc" / size_dir / representation / name
                    rtc = read(rtc_path) if rtc_path.exists() else None
                    metadata = audit_index[(model_size, representation, architecture, stage)]
                    key = default_latency_key(architecture)
                    sampling = latency[key]
                    joint_mse = float(np.mean([evaluation["decoded_per_dimension"][f"joint_{i}"]["mse"] for i in range(6)]))
                    record = {
                        "model_size": model_size,
                        "representation": representation,
                        "architecture": architecture,
                        "stage": stage,
                        "parameters": metadata["parameters"],
                        "final_train_action_mse_normalized": metadata["final_train_action_mse"],
                        "final_validation_action_mse_normalized": metadata["final_validation_action_mse"],
                        "test_decoded_normalized_mse": evaluation["decoded_all_normalized_error"]["mse"],
                        "test_decoded_physical_mse": evaluation["decoded_all_physical_error"]["mse"],
                        "test_decoded_physical_mae": evaluation["decoded_all_physical_error"]["mae"],
                        "test_decoded_physical_rmse": evaluation["decoded_all_physical_error"]["rmse"],
                        "test_joint_physical_mse": joint_mse,
                        "test_gripper_physical_mse": evaluation["decoded_per_dimension"]["gripper"]["mse"],
                        "encoded_control_mse": evaluation["encoded_control_error"]["mse"],
                        "token_accuracy": evaluation["token_accuracy"],
                        "boundary_physical_mse": evaluation["boundary_first_sample_error"]["mse"],
                        "velocity_p95_physical_per_s": evaluation["velocity_abs_quantiles"]["0.95"],
                        "acceleration_p95_physical_per_s2": evaluation["acceleration_abs_quantiles"]["0.95"],
                        "sampling_p50_ms": sampling["p50_ms"],
                        "sampling_p95_ms": sampling["p95_ms"],
                        "sampling_p99_ms": sampling["p99_ms"],
                        "online_vision_p50_ms": latency["online_vision_total"]["p50_ms"],
                        "projector_state_p50_ms": latency["projector_state"]["p50_ms"],
                        "action_decode_p50_ms": latency["action_decode"]["p50_ms"],
                        "estimated_online_pipeline_p50_ms": latency["online_vision_total"]["p50_ms"] + latency["projector_state"]["p50_ms"] + sampling["p50_ms"] + latency["action_decode"]["p50_ms"],
                        "peak_memory_bytes": latency["peak_memory_bytes"],
                        "delay_mean_physical_mse_d0_d10": None if rtc is None else float(np.mean([x["decoded_physical_error"]["mse"] for x in rtc["curves"]])),
                        "delay_d10_physical_mse": None if rtc is None else rtc["curves"][-1]["decoded_physical_error"]["mse"],
                        "checkpoint_sha256": metadata["sha256"],
                        "checkpoint": str((root / "outputs" / "SWEEP" / size_dir / representation / "checkpoints" / name.replace(".json", ".pt")).resolve()),
                        "wandb_url": metadata["wandb"]["url"],
                    }
                    if architecture == "discrete_joint":
                        legacy = baseline_latency[baseline_latency_key(architecture)]["p50_ms"]
                        record["joint_legacy_cached_p50_ms"] = legacy
                        record["joint_fused_cache_speedup_percent"] = pct_reduction(legacy, sampling["p50_ms"])
                        record["joint_legacy_vs_fused_tokens_equal"] = latency["cache_equivalence"]["legacy_vs_fused_tokens_equal"]
                        record["joint_legacy_vs_fused_max_token_difference"] = latency["cache_equivalence"]["legacy_vs_fused_max_token_difference"]
                    else:
                        record["joint_legacy_cached_p50_ms"] = None
                        record["joint_fused_cache_speedup_percent"] = None
                        record["joint_legacy_vs_fused_tokens_equal"] = None
                        record["joint_legacy_vs_fused_max_token_difference"] = None
                    records.append(record)

    lookup = {(x["model_size"], x["representation"], x["architecture"], x["stage"]): x for x in records}
    pairwise = []
    for model_size, _ in SIZES:
        for architecture in ARCHITECTURES:
            for stage in STAGES:
                raw = lookup[(model_size, "raw", architecture, stage)]
                spline = lookup[(model_size, "bspline", architecture, stage)]
                pairwise.append({
                    "comparison": "bspline_vs_raw",
                    "model_size": model_size,
                    "architecture": architecture,
                    "stage": stage,
                    "representation": "bspline",
                    "physical_mse_reduction_percent": pct_reduction(raw["test_decoded_physical_mse"], spline["test_decoded_physical_mse"]),
                    "normalized_mse_reduction_percent": pct_reduction(raw["test_decoded_normalized_mse"], spline["test_decoded_normalized_mse"]),
                    "sampling_latency_reduction_percent": pct_reduction(raw["sampling_p50_ms"], spline["sampling_p50_ms"]),
                    "delay_mean_mse_reduction_percent": None if raw["delay_mean_physical_mse_d0_d10"] is None else pct_reduction(raw["delay_mean_physical_mse_d0_d10"], spline["delay_mean_physical_mse_d0_d10"]),
                })
    for model_size, _ in SIZES:
        for representation in REPRESENTATIONS:
            for architecture in ARCHITECTURES:
                base = lookup[(model_size, representation, architecture, "base")]
                rtc = lookup[(model_size, representation, architecture, "ttrtc")]
                pairwise.append({
                    "comparison": "ttrtc_vs_base",
                    "model_size": model_size,
                    "architecture": architecture,
                    "stage": "ttrtc",
                    "representation": representation,
                    "physical_mse_reduction_percent": pct_reduction(base["test_decoded_physical_mse"], rtc["test_decoded_physical_mse"]),
                    "normalized_mse_reduction_percent": pct_reduction(base["test_decoded_normalized_mse"], rtc["test_decoded_normalized_mse"]),
                    "sampling_latency_reduction_percent": pct_reduction(base["sampling_p50_ms"], rtc["sampling_p50_ms"]),
                    "delay_mean_mse_reduction_percent": None if base["delay_mean_physical_mse_d0_d10"] is None else pct_reduction(base["delay_mean_physical_mse_d0_d10"], rtc["delay_mean_physical_mse_d0_d10"]),
                })

    write_csv(summary / "checkpoint_metrics.csv", records)
    write_csv(summary / "pairwise_effects.csv", pairwise)
    ranked = sorted(records, key=lambda x: x["test_decoded_physical_mse"])
    write_csv(summary / "ranking_by_physical_mse.csv", ranked)
    decoder = read(summary / "decoder_validation" / "decode_validation.json")
    spline_pairs = [x for x in pairwise if x["comparison"] == "bspline_vs_raw"]
    rtc_pairs = [x for x in pairwise if x["comparison"] == "ttrtc_vs_base"]
    joint = [x for x in records if x["architecture"] == "discrete_joint"]
    findings = {
        "best_checkpoint": {key: ranked[0][key] for key in ("model_size", "representation", "architecture", "stage", "test_decoded_physical_mse", "sampling_p50_ms")},
        "worst_checkpoint": {key: ranked[-1][key] for key in ("model_size", "representation", "architecture", "stage", "test_decoded_physical_mse", "sampling_p50_ms")},
        "bspline_physical_mse_wins": sum(x["physical_mse_reduction_percent"] > 0 for x in spline_pairs),
        "bspline_pairs": len(spline_pairs),
        "bspline_median_physical_mse_reduction_percent": median(x["physical_mse_reduction_percent"] for x in spline_pairs),
        "bspline_delay_mean_mse_wins": sum(x["delay_mean_mse_reduction_percent"] > 0 for x in spline_pairs),
        "bspline_median_delay_mean_mse_reduction_percent": median(x["delay_mean_mse_reduction_percent"] for x in spline_pairs),
        "ttrtc_zero_delay_physical_mse_wins": sum(x["physical_mse_reduction_percent"] > 0 for x in rtc_pairs),
        "ttrtc_pairs": len(rtc_pairs),
        "ttrtc_median_zero_delay_physical_mse_reduction_percent": median(x["physical_mse_reduction_percent"] for x in rtc_pairs),
        "ttrtc_delay_mean_mse_wins": sum(x["delay_mean_mse_reduction_percent"] > 0 for x in rtc_pairs),
        "ttrtc_median_delay_mean_mse_reduction_percent": median(x["delay_mean_mse_reduction_percent"] for x in rtc_pairs),
        "joint_fused_cache_median_speedup_percent": median(x["joint_fused_cache_speedup_percent"] for x in joint),
        "joint_fused_cache_min_speedup_percent": min(x["joint_fused_cache_speedup_percent"] for x in joint),
        "joint_fused_cache_max_speedup_percent": max(x["joint_fused_cache_speedup_percent"] for x in joint),
        "joint_fused_cache_exact_token_matches": sum(bool(x["joint_legacy_vs_fused_tokens_equal"]) for x in joint),
        "joint_fused_cache_comparisons": len(joint),
    }

    # Performance/latency frontier.  Keep this focused on the independently
    # trained base checkpoints; ttRTC effects have their own paired tables.
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    colors = {"raw": "#4c78a8", "bspline": "#f58518"}
    markers = {"fm": "o", "discrete_layerwise": "s", "discrete_joint": "^"}
    frontier_records = [record for record in records if record["stage"] == "base"]
    for record in frontier_records:
        color=colors[record["representation"]]
        ax.scatter(record["sampling_p50_ms"], record["test_decoded_physical_mse"], facecolors=color, edgecolors=color, marker=markers[record["architecture"]], s={"DiT-S":45,"DiT-B":85,"DiT-L":130}[record["model_size"]], alpha=.8, linewidths=1.5)
    ax.set(yscale="log", xlabel="default policy sampling p50 (ms, batch 1)", ylabel="decoded physical action MSE", title="18 base-checkpoint open-loop performance / latency frontier")
    ax.grid(alpha=.25, which="both")
    handles=[
        Line2D([],[],marker="o",linestyle="",color=colors["raw"],label="raw"),
        Line2D([],[],marker="o",linestyle="",color=colors["bspline"],label="B-spline"),
        Line2D([],[],marker="o",linestyle="",color="black",label="FM"),
        Line2D([],[],marker="s",linestyle="",color="black",label="layerwise"),
        Line2D([],[],marker="^",linestyle="",color="black",label="joint"),
        Line2D([],[],marker="o",linestyle="",color="gray",markersize=5,label="DiT-S"),
        Line2D([],[],marker="o",linestyle="",color="gray",markersize=8,label="DiT-B"),
        Line2D([],[],marker="o",linestyle="",color="gray",markersize=11,label="DiT-L"),
    ]
    ax.legend(handles=handles,ncol=2,fontsize=8,loc="upper right")
    fig.savefig(summary / "performance_vs_latency.png", dpi=180)
    plt.close(fig)

    labels=[]; matrix=[]
    for model_size, _ in SIZES:
        for architecture in ARCHITECTURES:
            for stage in STAGES:
                labels.append(f"{model_size} {architecture.replace('discrete_','')} {stage}")
                matrix.append([lookup[(model_size, rep, architecture, stage)]["test_decoded_physical_mse"] for rep in REPRESENTATIONS])
    matrix=np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 10), layout="constrained")
    image=ax.imshow(np.log10(matrix), aspect="auto", cmap="viridis_r")
    ax.set(xticks=[0,1],xticklabels=["raw","B-spline"],yticks=np.arange(len(labels)),yticklabels=labels,title="Decoded physical MSE (log10; lower is better)")
    for i in range(len(labels)):
        for j in range(2): ax.text(j,i,f"{matrix[i,j]:.4f}",ha="center",va="center",fontsize=7,color="white" if np.log10(matrix[i,j]) > np.median(np.log10(matrix)) else "black")
    fig.colorbar(image,ax=ax,label="log10 MSE")
    fig.savefig(summary / "physical_mse_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    joint_sorted=sorted(joint,key=lambda x:(x["model_size"],x["representation"],x["stage"]))
    joint_labels=[f"{x['model_size']}/{x['representation']}/{x['stage']}" for x in joint_sorted]
    speedups=[x["joint_fused_cache_speedup_percent"] for x in joint_sorted]
    ax.bar(np.arange(len(speedups)),speedups,color=[colors[x["representation"]] for x in joint_sorted])
    ax.axhline(0,color="black",lw=1); ax.set(xticks=np.arange(len(speedups)),xticklabels=joint_labels,ylabel="p50 speedup (%)",title="D2F-aligned fused inter-block K/V transition")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    ax.grid(axis="y",alpha=.25)
    fig.savefig(summary / "joint_cache_acceleration.png", dpi=180)
    plt.close(fig)

    result = {
        "protocol": {
            "checkpoints": 36,
            "test_windows_per_checkpoint": 3149,
            "open_loop_seed": 20260915,
            "open_loop_batch_size": 64,
            "default_sampling": "FM 12 steps; layerwise 8 rounds; joint 8 rounds with D2F-aligned fused K/V transition",
            "latency": "RTX PRO 6000 Blackwell, batch 1, 10 warmups + 100 timed iterations, PyTorch compile disabled",
            "scope": "open-loop recorded observations; not closed-loop task success",
        },
        "findings": findings,
        "decoder_validation": decoder,
        "records": records,
        "pairwise_effects": pairwise,
    }
    (summary / "summary.json").write_text(json.dumps(result,indent=2)+"\n")

    rows=[]
    for r in records:
        rows.append([r["model_size"],"B-spline" if r["representation"]=="bspline" else "raw",r["architecture"].replace("discrete_",""),r["stage"],f"{r['parameters']/1e6:.2f}",f"{r['final_validation_action_mse_normalized']:.5f}",f"{r['test_decoded_normalized_mse']:.5f}",f"{r['test_decoded_physical_mse']:.5f}",f"{r['sampling_p50_ms']:.2f}",f"{r['estimated_online_pipeline_p50_ms']:.2f}",f"[run]({r['wandb_url']})"])
    table=markdown_table(["size","repr.","policy","stage","M params","val action MSE","test norm. MSE","test physical MSE","sample p50 ms","estimated online p50 ms","W&B"],rows)
    top_rows=[[str(i+1),r["model_size"],r["representation"],r["architecture"],r["stage"],f"{r['test_decoded_physical_mse']:.6f}",f"{r['sampling_p50_ms']:.2f}"] for i,r in enumerate(ranked[:10])]
    top_table=markdown_table(["rank","size","repr.","policy","stage","physical MSE","sample p50 ms"],top_rows)
    delay_note = "Delay curves are included for all checkpoints." if all(x["delay_mean_physical_mse_d0_d10"] is not None for x in records) else "Delay-curve evaluation was not present when this summary was generated."
    en=f"""# Comprehensive 36-Checkpoint Sweep Summary

Generated from fresh evaluation artifacts in this directory on 2026-09-16.

## Evaluation contract

- All 36 checkpoints were evaluated on the same 3,149 held-out windows (five test episodes), using recorded observations and seed 20260915. These are open-loop action errors, not task-success rates.
- Primary quality metrics are decoded 30×7 trajectories: normalized MSE for representation-fair comparison and physical joint/gripper MSE after training-split q01/q99 scaling.
- Default sampling is FM 12 steps, layerwise diffusion 8 rounds, and joint diffusion 8 rounds with fused inter-block K/V transitions.
- Latency uses batch 1 on RTX PRO 6000 Blackwell: 10 warmups and 100 synchronized trials. `estimated online p50` is the sum of separately measured online vision, observation projector, policy sampling, and action decode medians; it is not a directly timed end-to-end percentile.
- {delay_note}

## Main findings

- Best physical MSE: **{ranked[0]['test_decoded_physical_mse']:.6f}**, {ranked[0]['model_size']} {ranked[0]['representation']} {ranked[0]['architecture']} {ranked[0]['stage']}.
- B-spline beats its matched raw checkpoint in **{findings['bspline_physical_mse_wins']}/{findings['bspline_pairs']}** pairs; median physical-MSE reduction is **{findings['bspline_median_physical_mse_reduction_percent']:.2f}%** (negative means worse).
- Across d=0…10, B-spline has lower mean delayed MSE in **{findings['bspline_delay_mean_mse_wins']}/{findings['bspline_pairs']}** matched pairs; the median reduction is **{findings['bspline_median_delay_mean_mse_reduction_percent']:.2f}%**.
- ttRTC improves zero-delay physical MSE in **{findings['ttrtc_zero_delay_physical_mse_wins']}/{findings['ttrtc_pairs']}** pairs and mean d=0…10 MSE in **{findings['ttrtc_delay_mean_mse_wins']}/{findings['ttrtc_pairs']}** pairs. Median delayed-MSE reduction is **{findings['ttrtc_median_delay_mean_mse_reduction_percent']:.2f}%**, so this fixed fine-tuning protocol does not reliably improve delay robustness across capacities.
- The D2F-aligned fused cache transition is exactly token-equivalent to the prior cached sampler in **{findings['joint_fused_cache_exact_token_matches']}/{findings['joint_fused_cache_comparisons']}** real checkpoints. Median p50 speedup is **{findings['joint_fused_cache_median_speedup_percent']:.2f}%** (range {findings['joint_fused_cache_min_speedup_percent']:.2f}% to {findings['joint_fused_cache_max_speedup_percent']:.2f}%).
- The known DiT-B raw layerwise instability is visible in open-loop error; capacity scaling is therefore not monotonic across every policy family.

## B-spline decode proof

- Every B-spline checkpoint uses the same manifest-matched `UniformLeftBSplineConfig`, implementation `uniform_left_direct_fit_v1`, tokenizer `{decoder['encoder']['tokenizer_id']}`, and calibration.
- Decode order is token dequantization (discrete only) → fixed **30×18** basis × **18×7** controls → **30×7** normalized actions → physical scaling.
- Across every held-out window, recomputed continuous/stored decode max error is **{decoder['recomputed_test_cache_agreement']['continuous_basis_vs_stored_reconstruction']['max_abs']:.3e}**; recomputed token/stored decode max error is **{decoder['recomputed_test_cache_agreement']['token_dequant_basis_vs_stored_quantized_reconstruction']['max_abs']:.3e}**.
- The basis is rank {decoder['geometry']['basis_rank']}; maximum partition-of-unity error is {decoder['geometry']['basis_row_sum_max_error']:.3e}. The basis figure uses a dense {decoder['geometry']['scipy_dense_basis_evaluation_points']:,}-point SciPy cubic evaluation, with the exact 30 decoder samples overlaid as dots; those integer samples match the decoder matrix to {decoder['geometry']['scipy_at_integer_steps_vs_decoder_basis']['max_abs']:.3e}. Full evidence is in `decoder_validation/decode_validation.json`.

## Top 10 by decoded physical MSE

{top_table}

## All checkpoints

{table}

## Artifacts

- `checkpoint_metrics.csv`: one row per checkpoint with training, test, latency, memory, smoothness, decoder, and W&B fields.
- `pairwise_effects.csv`: matched B-spline/raw and ttRTC/base effects.
- `summary.json`: complete machine-readable aggregation.
- `performance_vs_latency.png`, `physical_mse_matrix.png`, `joint_cache_acceleration.png`: comparison figures.
- `decoder_validation/`: numerical decode audit and inspected basis/trajectory figures.
- `open_loop/`, `latency/`, `latency_baseline/`, and `rtc/` (when present): raw per-checkpoint reports.

## Limitations

The dataset exposes absolute joints and gripper but no calibrated Cartesian pose, so position/rotation error cannot be reported honestly. Velocity/acceleration statistics describe predicted open-loop trajectory smoothness, not executed robot dynamics. No simulator or physical robot success claim is made.
"""
    (summary/"SUMMARY_EN.md").write_text(en)
    cn=f"""# 36 个检查点综合汇总

本报告由本目录中的重新评测结果生成，日期为 2026-09-16。

## 评测协议

- 36 个检查点均使用相同的 3,149 个测试窗口（5 个测试 episode）与随机种子 20260915。这里报告的是开环动作误差，不是任务成功率。
- 主要质量指标来自解码后的 30×7 轨迹：归一化 MSE 用于公平比较动作表示；物理 MSE 使用训练集 q01/q99 还原关节与夹爪单位。
- 默认采样：FM 12 步、layerwise diffusion 8 轮、joint diffusion 8 轮并启用融合的跨 block K/V cache 转换。
- 延迟在 RTX PRO 6000 Blackwell 上以 batch 1 测量：10 次预热、100 次同步计时。`estimated online p50` 是在线视觉、投影器、策略采样和动作解码各自 p50 的加和，并非直接测得的端到端分位数。

## 主要结论

- 最佳物理 MSE 为 **{ranked[0]['test_decoded_physical_mse']:.6f}**：{ranked[0]['model_size']} / {ranked[0]['representation']} / {ranked[0]['architecture']} / {ranked[0]['stage']}。
- 在 18 组严格配对中，B-spline 有 **{findings['bspline_physical_mse_wins']}/{findings['bspline_pairs']}** 组优于 raw；物理 MSE 降幅中位数为 **{findings['bspline_median_physical_mse_reduction_percent']:.2f}%**（负值表示变差）。
- ttRTC 在零延迟下有 **{findings['ttrtc_zero_delay_physical_mse_wins']}/{findings['ttrtc_pairs']}** 组改善；ttRTC 的核心目标是延迟重规划，因此不能只凭零延迟指标判断。
- 在 d=0…10 的平均延迟 MSE 上，B-spline 有 **{findings['bspline_delay_mean_mse_wins']}/{findings['bspline_pairs']}** 组优于 raw，中位降幅为 **{findings['bspline_median_delay_mean_mse_reduction_percent']:.2f}%**。
- ttRTC 在平均延迟 MSE 上仅有 **{findings['ttrtc_delay_mean_mse_wins']}/{findings['ttrtc_pairs']}** 组改善，中位降幅为 **{findings['ttrtc_median_delay_mean_mse_reduction_percent']:.2f}%**；当前固定微调协议不能稳定提升不同容量模型的延迟鲁棒性。
- 参考 `dd-openvla` D2F 实现的融合 cache 转换在 **{findings['joint_fused_cache_exact_token_matches']}/{findings['joint_fused_cache_comparisons']}** 个真实检查点上与原缓存采样器逐 token 完全一致；p50 加速中位数 **{findings['joint_fused_cache_median_speedup_percent']:.2f}%**，范围 {findings['joint_fused_cache_min_speedup_percent']:.2f}%–{findings['joint_fused_cache_max_speedup_percent']:.2f}%。

## B-spline 解码正确性

- 18 个 B-spline 检查点的 manifest 均与重新生成的 encoder 类型、配置、tokenizer `{decoder['encoder']['tokenizer_id']}` 和 calibration 完全一致。
- 解码顺序：离散 token 反量化（仅离散策略）→ 固定 30×18 basis × 18×7 控制点 → 30×7 归一化动作 → 物理尺度还原。
- 全部测试窗口中，连续重算与缓存重建的最大差值为 **{decoder['recomputed_test_cache_agreement']['continuous_basis_vs_stored_reconstruction']['max_abs']:.3e}**；离散 token 重算的最大差值为 **{decoder['recomputed_test_cache_agreement']['token_dequant_basis_vs_stored_quantized_reconstruction']['max_abs']:.3e}**。
- basis 秩为 {decoder['geometry']['basis_rank']}，行和最大误差为 {decoder['geometry']['basis_row_sum_max_error']:.3e}。basis 图使用 SciPy 在连续时间上密集计算 {decoder['geometry']['scipy_dense_basis_evaluation_points']:,} 个点，并以圆点叠加解码器实际使用的 30 个离散采样；离散采样与解码矩阵的最大差值为 {decoder['geometry']['scipy_at_integer_steps_vs_decoder_basis']['max_abs']:.3e}。

## 全部检查点

{table}

## 文件索引

- `checkpoint_metrics.csv`：36 个检查点的训练、测试、延迟、显存、平滑性、解码和 W&B 字段。
- `pairwise_effects.csv`：B-spline/raw 与 ttRTC/base 严格配对差异。
- `summary.json`：完整机器可读汇总。
- `performance_vs_latency.png`、`physical_mse_matrix.png`、`joint_cache_acceleration.png`：可视化比较。
- `decoder_validation/`：B-spline 数值解码审计与图形。

## 限制

数据集只提供绝对关节与夹爪量，没有经过标定的笛卡尔位姿，因此不能可靠报告位置/旋转误差。速度和加速度仅描述开环预测轨迹平滑性，不代表机器人真实执行动力学。本报告不声称仿真器或真实机器人任务成功率。
"""
    (summary/"SUMMARY_CN.md").write_text(cn)
    readme="""# SWEEP Evaluation Summary

- [English comprehensive report](SUMMARY_EN.md)
- [中文综合报告](SUMMARY_CN.md)
- [Machine-readable summary](summary.json)
- [All checkpoint metrics](checkpoint_metrics.csv)
- [Pairwise effects](pairwise_effects.csv)
- [B-spline decode validation](decoder_validation/decode_validation.json)
- [Dense cubic B-spline basis visualization](decoder_validation/bspline_basis.png)
- [dd-openvla acceleration audit](REFERENCE_ACCELERATION_AUDIT.md)
- [Performance/latency frontier](performance_vs_latency.png)
- [Physical-MSE matrix](physical_mse_matrix.png)
- [Joint-cache acceleration](joint_cache_acceleration.png)

Raw per-checkpoint evidence is retained in `open_loop/`, `latency/`, `latency_baseline/`, and `rtc/` when delay evaluation is present. Regenerable evaluation caches are under `cache/`.
"""
    (summary/"README.md").write_text(readme)
    print(json.dumps({"records":len(records),"pairwise":len(pairwise),"findings":findings},indent=2))


if __name__ == "__main__":
    main()
