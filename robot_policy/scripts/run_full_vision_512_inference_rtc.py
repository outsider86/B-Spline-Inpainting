#!/usr/bin/env python3
"""Evaluate conditional from-scratch suffix generation for 16 active checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_policy.config import ACTIVE_ARCHITECTURES
from robot_policy.evaluation.inference_rtc import _plot_metrics


SIZES = ("dit_s", "dit_b")
REPRESENTATIONS = ("raw", "bspline")
STAGES = ("base", "ttrtc")


def build_tasks(project: Path, root: Path, output: Path,
                sizes: tuple[str, ...], representations: tuple[str, ...],
                architectures: tuple[str, ...], stages: tuple[str, ...]) -> list[dict[str, str | Path]]:
    result = []
    for size in sizes:
        for representation in representations:
            for architecture in architectures:
                for stage in stages:
                    variant = f"{representation}_{architecture}_{stage}"
                    result.append({
                        "size": size,
                        "variant": variant,
                        "representation": representation,
                        "architecture": architecture,
                        "stage": stage,
                        "config": project / "configs" / "full_vision_512" / f"{size}_{representation}.yaml",
                        "checkpoint": root / size / representation / "checkpoints" / f"{architecture}_{stage}.pt",
                        "output": output / size / variant,
                    })
    return result


def summarize(output: Path, tasks: list[dict[str, str | Path]], primary_prefix: int) -> None:
    rows = []
    reports = []
    for task in tasks:
        report = json.loads((Path(task["output"]) / "report.json").read_text())
        curve = next(item for item in report["curves"] if item["raw_prefix_actions"] == primary_prefix)
        reports.append(report)
        rows.append({
            "model_size": report["model_size"],
            "vision_tokens": report["vision_tokens"],
            "representation": report["action_representation"],
            "architecture": report["architecture"],
            "stage": "ttrtc" if report["training_type"] in {"rtc", "ttrtc"} else report["training_type"],
            "samples": report["samples"],
            "primary_prefix": primary_prefix,
            "suffix_physical_mse": curve["suffix_physical_mse"],
            "suffix_physical_mae": curve["suffix_physical_mae"],
            "suffix_normalized_mse": curve["suffix_normalized_mse"],
            "fixed_control_max_abs": curve["fixed_control_max_abs"],
            "sampling_ms_per_sample": curve["sampling_ms_per_sample"],
        })
    identities = {
        (row["model_size"], row["representation"], row["architecture"], row["stage"])
        for row in rows
    }
    if len(rows) != len(tasks) or len(identities) != len(tasks):
        raise ValueError(f"expected {len(tasks)} unique inference-RTC reports")
    output.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "schema_version": 1,
        "scope": "512-vision-token oracle-prefix suffix generation on held-out test data",
        "variants": len(tasks),
        "primary_prefix": primary_prefix,
        "reports": reports,
    }
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    colors = {"fm": "#1f77b4", "discrete_joint": "#2ca02c"}
    labels = {"fm": "FM", "discrete_joint": "joint DD"}
    slug_by_name = {"DiT-S": "dit_s", "DiT-B": "dit_b"}
    for size in SIZES:
        selected = [report for report in reports if slug_by_name[report["model_size"]] == size]
        if not selected:
            continue
        mse_values = [
            float(curve["suffix_physical_mse"])
            for report in selected
            for curve in report["curves"]
        ]
        if any(value <= 0 or not math.isfinite(value) for value in mse_values):
            raise ValueError(f"{size} contains invalid physical MSE values")
        log_low, log_high = math.log(min(mse_values)), math.log(max(mse_values))
        padding = max(0.08 * (log_high - log_low), 0.04)
        shared_mse_ylim = (math.exp(log_low - padding), math.exp(log_high + padding))
        for report in selected:
            report["metric_plot_axis_limits"] = {
                "scope": f"{size} shared across raw and B-spline, FM and joint-DD, base and ttRTC",
                "yscale": "log",
                "suffix_physical_mse_min": shared_mse_ylim[0],
                "suffix_physical_mse_max": shared_mse_ylim[1],
            }
            report_path = output / size / report["variant"] / "report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            title = (
                f"{report['variant']} [{report['split']} split]: "
                "ground-truth-prefix RTC inpainting"
            )
            _plot_metrics(
                report_path.parent / "metrics_vs_prefix.png",
                report["curves"],
                title,
                mse_ylim=shared_mse_ylim,
            )
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5), layout="constrained", sharey=True
        )
        for axis, representation in zip(axes, REPRESENTATIONS):
            for report in selected:
                if report["action_representation"] != representation:
                    continue
                axis.plot(
                    [item["raw_prefix_actions"] for item in report["curves"]],
                    [item["suffix_physical_mse"] for item in report["curves"]],
                    marker="o", color=colors[report["architecture"]],
                    linestyle="-" if report["training_type"] == "base" else "--",
                    label=f"{labels[report['architecture']]} "
                          f"{'ttrtc' if report['training_type'] in {'rtc', 'ttrtc'} else report['training_type']}",
                )
            axis.set(title=representation, xlabel="ground-truth prefix (raw actions)",
                     ylabel="suffix physical MSE", yscale="log")
            axis.set_ylim(*shared_mse_ylim)
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(alpha=.3)
            handles, legend_labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, legend_labels, fontsize=8)
        fig.suptitle(f"{size.upper()} 512-token inference RTC (shared physical-MSE y-axis)")
        fig.savefig(output / f"{size}_inference_rtc.png", dpi=180)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--architectures", default=",".join(ACTIVE_ARCHITECTURES))
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--root", type=Path, default=Path("outputs/FULL_VISION_512"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefixes", default="2,4,6,8,10")
    parser.add_argument("--plot-prefix", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sizes = tuple(item.strip() for item in args.sizes.split(",") if item.strip())
    unknown = set(sizes).difference(SIZES)
    if unknown or not sizes:
        parser.error(f"invalid sizes: {sorted(unknown)}")
    representations = tuple(item.strip() for item in args.representations.split(",") if item.strip())
    unknown = set(representations).difference(REPRESENTATIONS)
    if unknown or not representations:
        parser.error(f"invalid representations: {sorted(unknown)}")
    architectures = tuple(item.strip() for item in args.architectures.split(",") if item.strip())
    unknown = set(architectures).difference(ACTIVE_ARCHITECTURES)
    if unknown or not architectures:
        parser.error(f"invalid architectures: {sorted(unknown)}")
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown = set(stages).difference(STAGES)
    if unknown or not stages:
        parser.error(f"invalid stages: {sorted(unknown)}")
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus:
        parser.error("at least one GPU is required")
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "summary" / "inference_rtc"
    all_tasks = build_tasks(project, root, output, sizes, representations, architectures, stages)
    # A sweep can be evaluated while training is still in flight.  Only atomic
    # final checkpoints are eligible; missing combinations remain pending for a
    # later restart-safe invocation rather than becoming false failures.
    ready_tasks = [task for task in all_tasks if Path(task["checkpoint"]).is_file()]
    pending: Queue[dict[str, str | Path]] = Queue()
    for task in ready_tasks:
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
            destination = Path(task["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "robot_policy.cli", "evaluate_inference_rtc",
                "--config", str(task["config"]),
                "--architecture", str(task["architecture"]),
                "--checkpoint", str(task["checkpoint"]),
                "--output-dir", str(destination),
                "--prefixes", args.prefixes,
                "--plot-prefix", str(args.plot_prefix),
                "--max-samples", str(args.max_samples),
                "--batch-size", str(args.batch_size),
                "--plot-samples", str(args.plot_samples),
            ]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu; env["PYTHONPATH"] = str(project / "src")
            log = output / "logs" / f"{task['size']}__{task['variant']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                print(f"[gpu {gpu}] start {task['size']}/{task['variant']}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with lock:
                print(f"[gpu {gpu}] {'done' if completed.returncode == 0 else 'FAILED'} {task['size']}/{task['variant']}", flush=True)
            if completed.returncode:
                failures.append((f"{task['size']}/{task['variant']}", completed.returncode))
            pending.task_done()

    threads = [Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        raise SystemExit(f"inference-RTC failures: {failures}")
    if ready_tasks:
        summarize(output, ready_tasks, args.plot_prefix)


if __name__ == "__main__":
    main()
