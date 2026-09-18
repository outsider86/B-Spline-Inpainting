from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_policy.config import ACTIVE_ARCHITECTURES

ARCHITECTURES = ACTIVE_ARCHITECTURES


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _default_latency(architecture: str, report: dict[str, Any]) -> float:
    if architecture == "fm":
        key = "sampling_steps-12"
    else:
        key = "sampling_rounds-8_use_cache-True"
    return float(report[key]["p50_ms"])


def _manifest_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    entries = _read(root / "checkpoints" / "checkpoint_manifest.json")["checkpoints"]
    result = {}
    for entry in entries:
        stage = "base" if entry["training_type"] == "base" else "ttrtc"
        result[(entry["architecture"], stage)] = entry
    return result


def build(raw_root: Path, bspline_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    roots = {"raw": raw_root, "bspline": bspline_root}
    manifests = {name: _manifest_index(root) for name, root in roots.items()}
    records = []
    for representation, root in roots.items():
        for architecture in ARCHITECTURES:
            for stage in ("base", "ttrtc"):
                file_stage = stage if representation == "raw" else ("base" if stage == "base" else "rtc")
                name = f"{architecture}_{file_stage}"
                evaluation = _read(root / "evaluation" / f"{name}.json")
                delays = _read(root / "evaluation" / f"{name}_delays.json")
                latency = _read(root / "latency" / f"{name}.json")
                entry = manifests[representation][(architecture, stage)]
                delay_mse = [float(curve["decoded_physical_error"]["mse"]) for curve in delays["curves"]]
                records.append({
                    "representation": representation,
                    "architecture": architecture,
                    "stage": stage,
                    "validation_action_mse_normalized": float(entry["validation_final"]["action_mse"]),
                    "test_physical_action_mse": float(evaluation["decoded_all_physical_error"]["mse"]),
                    "test_physical_action_mae": float(evaluation["decoded_all_physical_error"]["mae"]),
                    "test_physical_action_rmse": float(evaluation["decoded_all_physical_error"]["rmse"]),
                    "token_accuracy": evaluation["token_accuracy"],
                    "delay_mean_physical_mse_d0_d10": float(np.mean(delay_mse)),
                    "delay_d0_physical_mse": delay_mse[0],
                    "delay_d10_physical_mse": delay_mse[-1],
                    "default_sampling_p50_ms": _default_latency(architecture, latency),
                    "action_scalar_tokens": 210 if representation == "raw" else 126,
                    "trainable_parameters": int(entry["parameter_counts"]["total_trainable"]),
                    "wandb_project": entry["wandb"]["project"],
                    "wandb_run_id": entry["wandb"]["run_id"],
                    "wandb_url": entry["wandb"]["url"],
                })

    lookup = {(r["representation"], r["architecture"], r["stage"]): r for r in records}
    effects = []
    for representation in roots:
        for architecture in ARCHITECTURES:
            base = lookup[(representation, architecture, "base")]["test_physical_action_mse"]
            rtc = lookup[(representation, architecture, "ttrtc")]["test_physical_action_mse"]
            delay_base = lookup[(representation, architecture, "base")]["delay_mean_physical_mse_d0_d10"]
            delay_rtc = lookup[(representation, architecture, "ttrtc")]["delay_mean_physical_mse_d0_d10"]
            effects.append({
                "representation": representation,
                "architecture": architecture,
                "test_mse_reduction_percent": 100 * (base - rtc) / base,
                "delay_mean_mse_reduction_percent": 100 * (delay_base - delay_rtc) / delay_base,
            })

    fields = list(records[0])
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)

    labels = ["FM", "Joint DD"]
    combinations = (("raw", "base"), ("raw", "ttrtc"), ("bspline", "base"), ("bspline", "ttrtc"))
    legend = {("raw", "base"): "raw base", ("raw", "ttrtc"): "raw ttRTC", ("bspline", "base"): "B-spline base", ("bspline", "ttrtc"): "B-spline ttRTC"}
    colors = {("raw", "base"): "#4c78a8", ("raw", "ttrtc"): "#72b7b2", ("bspline", "base"): "#f58518", ("bspline", "ttrtc"): "#e45756"}
    metrics = (
        ("test_physical_action_mse", "Held-out physical action MSE ↓"),
        ("validation_action_mse_normalized", "Final normalized validation action MSE ↓"),
        ("delay_mean_physical_mse_d0_d10", "Mean delayed physical MSE, d=0…10 ↓"),
    )
    x = np.arange(len(ARCHITECTURES)); width = 0.19
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, (metric, title) in zip(axes, metrics):
        for offset, combination in enumerate(combinations):
            values = [lookup[(combination[0], architecture, combination[1])][metric] for architecture in ARCHITECTURES]
            ax.bar(x + (offset - 1.5) * width, values, width, label=legend[combination], color=colors[combination])
        ax.set(xticks=x, xticklabels=labels, title=title); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Raw actions vs B-spline controls under the same deterministic training protocol")
    figure_path = output / "action_mse_comparison.png"
    fig.savefig(figure_path, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(ARCHITECTURES)); width = .34
    for offset, representation in enumerate(("raw", "bspline")):
        values = [next(x["test_mse_reduction_percent"] for x in effects if x["representation"] == representation and x["architecture"] == architecture) for architecture in ARCHITECTURES]
        ax.bar(positions + (offset - .5) * width, values, width, label=representation)
    ax.axhline(0, color="black", lw=1); ax.set(xticks=positions, xticklabels=labels, ylabel="physical action MSE reduction (%)", title="ttRTC effect relative to matching base checkpoint")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    effect_path = output / "ttrtc_effect_comparison.png"
    fig.savefig(effect_path, dpi=180); plt.close(fig)

    report = {
        "scope": "same dataset split, architectures, optimizer schedule, update budget, seed, evaluation windows, and hardware class",
        "metric_contract": {
            "training_validation_action_mse": "normalized 30x7 decoded action MSE on corruption-supervised support; B-spline non-supervised controls are ground-truth-filled before decoding; identity raw representation reduces exactly to the original raw metric",
            "primary_evaluation": "decoded physical action MSE on all 3,149 held-out windows",
        },
        "records": records,
        "ttrtc_effects": effects,
        "figures": [str(figure_path.resolve()), str(effect_path.resolve())],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def log_wandb(report: dict[str, Any], output: Path, entity: str, project: str, mode: str) -> dict[str, str]:
    import wandb

    run = wandb.init(entity=entity, project=project, name="raw-vs-bspline-action-mse", job_type="comparison", mode=mode,
                     config={"raw_project_base": "raw-actions-basic", "raw_project_ttrtc": "raw-actions-ttRTC",
                             "bspline_project_base": "bspline-actions-basic", "bspline_project_ttrtc": "bspline-actions-ttRTC"})
    columns = list(report["records"][0])
    table = wandb.Table(columns=columns, data=[[record[column] for column in columns] for record in report["records"]])
    payload: dict[str, Any] = {
        "comparison_table": table,
        "action_mse_comparison": wandb.Image(str(output / "action_mse_comparison.png")),
        "ttrtc_effect_comparison": wandb.Image(str(output / "ttrtc_effect_comparison.png")),
    }
    for record in report["records"]:
        prefix = f"{record['representation']}/{record['architecture']}/{record['stage']}"
        payload[f"{prefix}/test_physical_action_mse"] = record["test_physical_action_mse"]
        payload[f"{prefix}/validation_action_mse_normalized"] = record["validation_action_mse_normalized"]
        payload[f"{prefix}/delay_mean_physical_mse"] = record["delay_mean_physical_mse_d0_d10"]
        payload[f"{prefix}/sampling_p50_ms"] = record["default_sampling_p50_ms"]
    run.log(payload)
    artifact = wandb.Artifact("raw-vs-bspline-comparison", type="evaluation")
    artifact.add_file(str((output / "comparison.json").resolve()))
    artifact.add_file(str((output / "comparison.csv").resolve()))
    artifact.add_file(str((output / "action_mse_comparison.png").resolve()))
    artifact.add_file(str((output / "ttrtc_effect_comparison.png").resolve()))
    run.log_artifact(artifact)
    info = {"entity": run.entity, "project": run.project, "run_id": run.id, "url": run.url, "state": "finished"}
    run.finish()
    (output / "wandb_comparison.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="outputs/raw_actions")
    parser.add_argument("--bspline-root", default="outputs/bspline_reproduction")
    parser.add_argument("--output", default="outputs/action_representation_comparison")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default="robot-policy-action-representation-comparison")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    args = parser.parse_args(argv)
    output = Path(args.output)
    report = build(Path(args.raw_root), Path(args.bspline_root), output)
    info = None
    if args.wandb_mode != "disabled":
        if not args.wandb_entity:
            raise ValueError("--wandb-entity is required when W&B logging is enabled")
        info = log_wandb(report, output, args.wandb_entity, args.wandb_project, args.wandb_mode)
    print(json.dumps({"output": str(output.resolve()), "records": len(report["records"]), "wandb": info}, indent=2))


if __name__ == "__main__":
    main()
