#!/usr/bin/env python3
"""Render final English and Chinese reports from the audited 36-run sweep."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any


ARCHITECTURE_LABELS = {
    "fm": "FM",
    "discrete_layerwise": "Layerwise DD",
    "discrete_joint": "Joint DD",
}


def metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.8g}"


def result_rows(records: list[dict[str, Any]], chinese: bool = False) -> list[str]:
    rows: list[str] = []
    for item in records:
        wandb = item.get("wandb") or {}
        run = f"[{wandb.get('run_id', '—')}]({wandb.get('url', '')})" if wandb.get("url") else "—"
        representation_labels = {"raw": "Raw", "bspline": "B-spline"}
        architecture_labels = ARCHITECTURE_LABELS
        stage_labels = {"base": "Base", "ttrtc": "ttRTC"}
        if chinese:
            representation_labels = {"raw": "Raw", "bspline": "B-spline"}
            architecture_labels = {"fm": "流匹配", "discrete_layerwise": "逐层离散扩散", "discrete_joint": "联合离散扩散"}
            stage_labels = {"base": "基础训练", "ttrtc": "ttRTC"}
        representation = representation_labels.get(item["representation"], item["representation"])
        stage = stage_labels.get(item["stage"], item["stage"])
        path = Path(item["path"])
        rows.append(
            "| " + " | ".join([
                item["model_size"], representation, architecture_labels[item["architecture"]], stage,
                f"{int(item['parameters']):,}", metric(item.get("final_train_action_mse")),
                metric(item.get("final_validation_action_mse")), f"`{path}`", f"`{item['sha256']}`", run,
            ]) + " |"
        )
    return rows


def capacity_rows(records: list[dict[str, Any]], chinese: bool = False) -> list[str]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[str] = []
    for item in records:
        key = (item["model_size"], item["representation"], item["architecture"])
        if item["stage"] != "base" or key in seen:
            continue
        seen.add(key)
        representation = "Raw" if item["representation"] == "raw" else "B-spline"
        architecture = ARCHITECTURE_LABELS[item["architecture"]]
        if chinese:
            architecture = {"fm": "流匹配", "discrete_layerwise": "逐层离散扩散", "discrete_joint": "联合离散扩散"}[item["architecture"]]
        rows.append(
            f"| {item['model_size']} | {representation} | {architecture} | "
            f"{int(item['parameters']):,} |"
        )
    return rows


def english(audit: dict[str, Any], records: list[dict[str, Any]]) -> str:
    return "\n".join([
        "# DiT-S/B/L 36-Checkpoint Capacity Sweep",
        "",
        f"**Report date:** {date.today().isoformat()}",
        "",
        "## Outcome",
        "",
        f"The capacity sweep is complete: **{audit['found_checkpoints']}/36** final checkpoints passed the requirement-level local and W&B audit. "
        "The experiment covers DiT-S, DiT-B, and DiT-L across raw and B-spline actions, FM, layerwise discrete diffusion, and joint discrete diffusion, each with a 50,000-update base stage and 5,000-update ttRTC stage.",
        "",
        "These are open-loop validation metrics, not closed-loop robot success rates. `action_mse` is the common cross-architecture metric; architecture-specific training objectives are not directly comparable.",
        "",
        "## Numerical stability warning",
        "",
        "Post-sweep gradient forensics found FP32 gradient-norm overflow in six layerwise-DD runs: DiT-B raw and B-spline base/ttRTC, and DiT-L B-spline base/ttRTC. Their forward losses remained finite, but clipping erased the new gradient signal after the overflow. These checkpoints are valid reproduction artifacts, not trustworthy converged capacity comparisons. DiT-S layerwise and raw DiT-L layerwise remained finite. See `CodexDoc/reports/GRADIENT_INSTABILITY_INVESTIGATION_EN.md` for evidence and corrective recommendations.",
        "",
        "## Fixed protocol",
        "",
        "- Seed: 7 with deterministic CUDA/data loading.",
        "- Micro batch = effective batch = 32.",
        "- Base training: exactly 50,000 updates in `robot-policy-50k-bs32-basic`.",
        "- ttRTC fine-tuning: exactly 5,000 updates in `robot-policy-5k-bs32-ttRTC`.",
        "- Shapes: DiT-S 6×384/4 heads; DiT-B 12×768/12 heads; DiT-L 24×1024/16 heads.",
        "- Each ttRTC checkpoint has an exact local base parent; joint-DD parent-prediction caches are keyed by the parent SHA-256.",
        "",
        "## Trainable policy sizes",
        "",
        "| Size | Representation | Architecture | Parameters |",
        "|---|---|---|---:|",
        *capacity_rows(records),
        "",
        "## Checkpoint results",
        "",
        "| Size | Representation | Architecture | Stage | Parameters | Final train action MSE | Final validation action MSE | Checkpoint | SHA-256 | W&B |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
        *result_rows(records),
        "",
        "## Verification",
        "",
        "The completion audit verifies exactly 36 expected `.pt` files and no extras; exact update counts and model shapes; batch size 32; parameter counts; local train/validation `action_mse`; independent state-dict reload; manifest uniqueness, hashes, and parent lineage; W&B finished state, update summaries, action-MSE history, and model artifacts.",
        "",
        "Machine-readable evidence: `robot_policy/outputs/model_size_sweep_50k_bs32/completion_audit.json`.",
        "",
    ])


def chinese(audit: dict[str, Any], records: list[dict[str, Any]]) -> str:
    return "\n".join([
        "# DiT-S/B/L 36 个检查点容量扫描",
        "",
        f"**报告日期：** {date.today().isoformat()}",
        "",
        "## 结果",
        "",
        f"容量扫描已完成：**{audit['found_checkpoints']}/36** 个最终检查点全部通过本地与 W&B 的逐项审计。实验覆盖 DiT-S、DiT-B、DiT-L，Raw 与 B-spline 两种动作表示，FM、逐层离散扩散和联合离散扩散三种架构；每组均包含 50,000 updates 的 base 阶段和 5,000 updates 的 ttRTC 阶段。",
        "",
        "这些是开环验证指标，不代表闭环机器人成功率。`action_mse` 是跨架构统一对比指标；各架构自身的训练 objective 不能直接横向比较。",
        "",
        "## 数值稳定性警告",
        "",
        "扫描后的梯度取证发现 6 个逐层离散扩散运行发生 FP32 梯度范数溢出：DiT-B Raw 与 B-spline 的 base/ttRTC，以及 DiT-L B-spline 的 base/ttRTC。其前向 loss 仍保持有限，但溢出后裁剪会清除新的梯度信号。这些检查点是有效的复现实验产物，但不能视为可信的收敛后容量对比。DiT-S 逐层模型和 Raw DiT-L 逐层模型保持有限。证据与修复建议见 `CodexDoc/reports/GRADIENT_INSTABILITY_INVESTIGATION_CN.md`。",
        "",
        "## 固定实验协议",
        "",
        "- 随机种子为 7，并启用确定性 CUDA 与数据加载。",
        "- Micro batch 与 effective batch 均为 32。",
        "- Base：严格 50,000 updates，记录到 `robot-policy-50k-bs32-basic`。",
        "- ttRTC：严格 5,000 updates，记录到 `robot-policy-5k-bs32-ttRTC`。",
        "- 结构：DiT-S 6×384/4 heads；DiT-B 12×768/12 heads；DiT-L 24×1024/16 heads。",
        "- 每个 ttRTC 检查点均有精确的本地 base 父检查点；联合 DD 的父策略预测缓存按父检查点 SHA-256 隔离。",
        "",
        "## 可训练策略规模",
        "",
        "| 规模 | 动作表示 | 架构 | 参数量 |",
        "|---|---|---|---:|",
        *capacity_rows(records, chinese=True),
        "",
        "## 检查点结果",
        "",
        "| 规模 | 动作表示 | 架构 | 阶段 | 参数量 | 最终训练 action MSE | 最终验证 action MSE | 检查点 | SHA-256 | W&B |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
        *result_rows(records, chinese=True),
        "",
        "## 验证范围",
        "",
        "完成审计验证：恰好存在 36 个预期 `.pt` 文件且无额外文件；update 数、模型结构、batch size 32 与参数量准确；本地训练/验证 `action_mse` 完整；state dict 可独立重载；manifest 条目唯一、哈希和父子关系正确；W&B run 已结束、update summary 正确、包含 action-MSE 历史并上传模型 artifact。",
        "",
        "机器可读证据：`robot_policy/outputs/model_size_sweep_50k_bs32/completion_audit.json`。",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audit_path = args.audit or project_root / "outputs/model_size_sweep_50k_bs32/completion_audit.json"
    report_root = args.reports or project_root.parent / "CodexDoc/reports"
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("found_checkpoints") != 36:
        raise SystemExit("completion audit must pass for all 36 checkpoints before rendering final reports")
    records = sorted(audit["records"], key=lambda x: (
        ("DiT-S", "DiT-B", "DiT-L").index(x["model_size"]),
        ("raw", "bspline").index(x["representation"]),
        ("fm", "discrete_layerwise", "discrete_joint").index(x["architecture"]),
        ("base", "ttrtc").index(x["stage"]),
    ))
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "MODEL_SIZE_SWEEP_36_EN.md").write_text(english(audit, records))
    (report_root / "MODEL_SIZE_SWEEP_36_CN.md").write_text(chinese(audit, records))


if __name__ == "__main__":
    main()
