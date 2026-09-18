#!/usr/bin/env python3
"""Run and summarize the active DiT-S ground-truth-prefix RTC evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPRESENTATIONS = ("raw", "bspline")
ARCHITECTURES = ("fm", "discrete_joint")
STAGES = ("base", "ttrtc")


def build_tasks(project: Path, output: Path) -> list[dict[str, str | Path]]:
    tasks = []
    for representation in REPRESENTATIONS:
        for architecture in ARCHITECTURES:
            for stage in STAGES:
                variant = f"{representation}_{architecture}_{stage}"
                tasks.append({
                    "variant": variant,
                    "representation": representation,
                    "architecture": architecture,
                    "stage": stage,
                    "config": project / "configs" / "model_size_sweep" / f"dit_s_{representation}.yaml",
                    "prepared": project / "outputs" / "SWEEP" / "summary" / "cache" / representation,
                    "vision": project / "outputs" / "SWEEP" / "summary" / "cache" / "vision",
                    "checkpoint": project / "outputs" / "SWEEP" / "dit_s" / representation / "checkpoints" / f"{architecture}_{stage}.pt",
                    "output": output / variant,
                })
    return tasks


def summarize(output: Path, tasks: list[dict[str, str | Path]], primary_prefix: int) -> None:
    reports = []
    rows = []
    for task in tasks:
        path = Path(task["output"]) / "report.json"
        if not path.exists():
            raise FileNotFoundError(f"missing RTC evaluation report: {path}")
        report = json.loads(path.read_text())
        curve = next((row for row in report["curves"] if row["raw_prefix_actions"] == primary_prefix), None)
        if curve is None:
            raise ValueError(f"{report['variant']} has no prefix={primary_prefix} curve")
        reports.append(report)
        rows.append({
            "variant": report["variant"],
            "representation": report["action_representation"],
            "architecture": report["architecture"],
            "training_type": report["training_type"],
            "samples": report["samples"],
            "primary_prefix": primary_prefix,
            "suffix_physical_mse": curve["suffix_physical_mse"],
            "suffix_physical_mae": curve["suffix_physical_mae"],
            "suffix_normalized_mse": curve["suffix_normalized_mse"],
            "prefix_reference_normalized_mse": curve["prefix_reference_normalized_mse"],
            "fixed_control_max_abs": curve["fixed_control_max_abs"],
            "sampling_ms_per_sample": curve["sampling_ms_per_sample"],
        })
    expected_variants = len(REPRESENTATIONS) * len(ARCHITECTURES) * len(STAGES)
    if len(reports) != expected_variants or len({report["variant"] for report in reports}) != expected_variants:
        raise ValueError(f"summary requires exactly {expected_variants} unique DiT-S variants")
    (output / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "scope": "DiT-S oracle-ground-truth-prefix inference RTC on the held-out test split",
        "variants": len(reports),
        "primary_prefix": primary_prefix,
        "reports": reports,
    }, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    colors = {"fm": "#1f77b4", "discrete_joint": "#2ca02c"}
    labels = {"fm": "FM", "discrete_joint": "joint DD"}
    for ax, representation in zip(axes, REPRESENTATIONS):
        for report in reports:
            if report["action_representation"] != representation:
                continue
            x = [curve["raw_prefix_actions"] for curve in report["curves"]]
            y = [curve["suffix_physical_mse"] for curve in report["curves"]]
            stage = report["training_type"]
            ax.plot(x, y, marker="o", color=colors[report["architecture"]],
                    ls="-" if stage == "base" else "--",
                    label=f"{labels[report['architecture']]} {stage}")
        ax.set_title(representation)
        ax.set_xlabel("ground-truth condition length (raw steps)")
        ax.set_ylabel("inpainted suffix physical MSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DiT-S inference-RTC: ground-truth-prefix inpainting")
    fig.tight_layout()
    fig.savefig(output / "inpainting_comparison.png", dpi=180)
    plt.close(fig)

    ordered = sorted(rows, key=lambda row: (row["representation"], row["suffix_physical_mse"]))
    lines = [
        "# DiT-S inference-RTC evaluation",
        "",
        "All 8 raw/B-spline × FM/joint-DD × base/ttRTC checkpoints were evaluated on the same",
        "episode-balanced held-out test samples. Each prediction receives the current observation and an",
        "oracle ground-truth action prefix, then generates the remaining chunk from scratch.",
        "",
        "For B-splines, conditioning freezes the complete cubic control support needed by the requested",
        "raw-action prefix. The primary table below uses a six-action (200 ms at 30 Hz) prefix.",
        "",
        "| Variant | Suffix physical MSE | Suffix physical MAE | Prefix preservation MSE | ms/sample |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| `{row['variant']}` | {row['suffix_physical_mse']:.8f} | {row['suffix_physical_mae']:.8f} | "
            f"{row['prefix_reference_normalized_mse']:.3e} | {row['sampling_ms_per_sample']:.2f} |"
        )
    keyed = {(row["representation"], row["architecture"], row["training_type"]): row for row in rows}
    lines += [
        "",
        "## Paired ttRTC effect at the primary prefix",
        "",
        "Positive values mean that ttRTC reduced suffix physical MSE relative to its matching base checkpoint.",
        "",
        "| Representation | Architecture | Base MSE | ttRTC MSE | ttRTC improvement |",
        "|---|---|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        for architecture in ARCHITECTURES:
            base = keyed[(representation, architecture, "base")]["suffix_physical_mse"]
            ttrtc = keyed[(representation, architecture, "ttrtc")]["suffix_physical_mse"]
            improvement = 100.0 * (base - ttrtc) / base
            lines.append(
                f"| {representation} | {labels[architecture]} | {base:.8f} | {ttrtc:.8f} | {improvement:+.2f}% |"
            )
    lines += [
        "",
        f"Best primary-prefix result: `{ordered[0]['variant']}` at physical suffix MSE "
        f"`{ordered[0]['suffix_physical_mse']:.8f}`.",
        "",
        "Each variant folder contains `report.json`, `per_sample_metrics.csv`, `trajectory_examples.npz`,",
        "`trajectory_examples.png`, and `metrics_vs_prefix.png`.",
        "",
        "The evaluation is open-loop inpainting, not simulator or physical-robot task success.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output")
    parser.add_argument("--prefixes", default="2,4,6,8,10")
    parser.add_argument("--plot-prefix", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve() if args.output else project / "outputs" / "RTCEVAL"
    output.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(project, output)
    pending: Queue[dict[str, str | Path]] = Queue()
    for task in tasks:
        if args.force or not (Path(task["output"]) / "report.json").exists():
            pending.put(task)
    failures: list[tuple[str, int]] = []
    lock = Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                task = pending.get_nowait()
            except Empty:
                return
            variant = str(task["variant"])
            command = [
                sys.executable, "-m", "robot_policy.cli", "evaluate_inference_rtc",
                "--config", str(task["config"]),
                "--set", f"data.prepared_path={task['prepared']}",
                "--set", f"data.vision_cache_path={task['vision']}",
                "--architecture", str(task["architecture"]),
                "--checkpoint", str(task["checkpoint"]),
                "--output-dir", str(task["output"]),
                "--prefixes", args.prefixes,
                "--plot-prefix", str(args.plot_prefix),
                "--max-samples", str(args.max_samples),
                "--batch-size", str(args.batch_size),
                "--plot-samples", str(args.plot_samples),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(project / "src")
            with lock:
                print(f"[gpu {gpu}] start {variant}", flush=True)
            with (output / f"{variant}.log").open("w") as handle:
                completed = subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
            with lock:
                print(f"[gpu {gpu}] {'done' if completed.returncode == 0 else 'FAILED'} {variant}", flush=True)
            if completed.returncode:
                failures.append((variant, completed.returncode))
            pending.task_done()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    threads = [Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit("; ".join(f"{variant}: exit {code}" for variant, code in failures))
    summarize(output, tasks, args.plot_prefix)
    print(f"completed and summarized 12 variants in {output}")


if __name__ == "__main__":
    main()
