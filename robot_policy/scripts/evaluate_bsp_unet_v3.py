#!/usr/bin/env python3
"""Evaluate all v3 checkpoints and build the base-only latency/accuracy plot."""

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


VARIANTS = (
    "fm_raw_h1", "fm_raw_h2", "fm_bspline_h1", "fm_bspline_h2",
    "dd_raw_h1", "dd_raw_h2", "dd_bspline_h1", "dd_bspline_h2",
)


def _architecture(variant: str) -> str:
    return "bsp_unet_fm" if variant.startswith("fm_") else "bsp_unet_discrete"


def _run(command: list[str], root: Path, gpu: str, log: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(root / "src")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        result = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"evaluation failed ({result.returncode}); see {log}")


def _evaluate_job(root: Path, output: Path, variant: str, stage: str, gpu: str, force: bool) -> None:
    python = sys.executable
    config = root / "configs" / "bsp_unet_v3" / f"{variant}.yaml"
    checkpoint = root / "outputs" / "BSP_UNET_V3" / "checkpoints" / variant / f"{stage}.pt"
    architecture = _architecture(variant)
    target = output / variant / stage
    target.mkdir(parents=True, exist_ok=True)
    log = output / "logs" / f"{variant}_{stage}.log"
    commands = []
    if force or not (target / "open_loop.json").exists():
        commands.append([
            python, "-m", "robot_policy.cli", "evaluate_open_loop",
            "--config", str(config), "--architecture", architecture,
            "--checkpoint", str(checkpoint), "--batch-size", "64",
            "--output", str(target / "open_loop.json"),
        ])
    if force or not (target / "latency.json").exists():
        commands.append([
            python, "-m", "robot_policy.cli", "benchmark_inference",
            "--config", str(config), "--architecture", architecture,
            "--checkpoint", str(checkpoint), "--warmup", "10", "--iterations", "100",
            "--output", str(target / "latency.json"),
        ])
    for split in ("test", "train"):
        rtc_dir = target / f"inference_rtc_{split}"
        if force or not (rtc_dir / "report.json").exists():
            commands.append([
                python, "-m", "robot_policy.cli", "evaluate_inference_rtc",
                "--config", str(config), "--architecture", architecture,
                "--checkpoint", str(checkpoint), "--output-dir", str(rtc_dir),
                "--prefixes", "2,4,6,8,10", "--plot-prefix", "6",
                "--max-samples", "64", "--batch-size", "32", "--plot-samples", "5",
                "--split", split,
            ])
    for command in commands:
        _run(command, root, gpu, log)


def _summarize(output: Path) -> None:
    rows = []
    for variant in VARIANTS:
        for stage in ("base", "rtc"):
            target = output / variant / stage
            open_loop = json.loads((target / "open_loop.json").read_text())
            latency = json.loads((target / "latency.json").read_text())
            architecture = _architecture(variant)
            latency_key = "sampling_steps-12" if architecture.endswith("fm") else "sampling_rounds-8"
            rows.append({
                "variant": variant,
                "stage": stage,
                "architecture": architecture,
                "representation": "bspline" if "bspline" in variant else "raw",
                "observation_horizon": 2 if variant.endswith("h2") else 1,
                "normalized_generation_action_mse": open_loop["decoded_all_normalized_error"]["mse"],
                "physical_generation_action_mse": open_loop["decoded_all_physical_error"]["mse"],
                "sampling_p50_ms_batch1": latency[latency_key]["p50_ms"],
                "sampling_p95_ms_batch1": latency[latency_key]["p95_ms"],
                "parameters": latency.get("parameter_counts"),
            })
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({
        "scope": "BSP scratch-vision U-Net v3; generation from scratch on held-out test observations",
        "frontier_scope": "base checkpoints only; ttRTC intentionally excluded from visualization",
        "rows": rows,
    }, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"raw": "#4c78a8", "bspline": "#f58518"}
    markers = {"bsp_unet_fm": "o", "bsp_unet_discrete": "^"}
    for horizon, axis in zip((1, 2), axes):
        for row in rows:
            if row["stage"] != "base" or row["observation_horizon"] != horizon:
                continue
            label = f"{row['representation']} / {'FM' if row['architecture'].endswith('fm') else 'discrete DD'}"
            axis.scatter(
                row["sampling_p50_ms_batch1"], row["normalized_generation_action_mse"],
                c=colors[row["representation"]], marker=markers[row["architecture"]],
                s=100, label=label,
            )
            axis.annotate(row["variant"], (row["sampling_p50_ms_batch1"], row["normalized_generation_action_mse"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set_title(f"{horizon}-frame observation")
        axis.set_xlabel("default policy sampling p50 (ms, batch 1; linear)")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("normalized decoded generation action MSE (log)")
    fig.suptitle("BSP-UNet v3: base checkpoint generation-from-scratch frontier")
    fig.tight_layout()
    fig.savefig(output / "performance_vs_latency.png", dpi=180)
    plt.close(fig)

    ordered = sorted(rows, key=lambda row: (row["stage"], row["normalized_generation_action_mse"]))
    lines = [
        "# BSP-UNet v3 evaluation summary", "",
        "Open-loop metrics are generation from scratch on the held-out test split. The latency/accuracy plot",
        "uses a linear latency x-axis, logarithmic action-MSE y-axis, and base checkpoints only.", "",
        "| Variant | Stage | Normalized action MSE | Physical action MSE | p50 ms |",
        "|---|---|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| `{row['variant']}` | {row['stage']} | {row['normalized_generation_action_mse']:.8f} | "
            f"{row['physical_generation_action_mse']:.8f} | {row['sampling_p50_ms_batch1']:.2f} |"
        )
    lines += [
        "", "Each checkpoint folder also contains test- and train-split RTC trajectory examples whose y-axis",
        "limits are the true per-channel min/max over the complete prepared dataset.", "",
        "These are open-loop diagnostics, not simulator or physical-robot task success rates.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "BSP_UNET_V3" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    queue: Queue[tuple[str, str]] = Queue()
    for variant in VARIANTS:
        for stage in ("base", "rtc"):
            queue.put((variant, stage))
    failures = []
    lock = Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                variant, stage = queue.get_nowait()
            except Empty:
                return
            try:
                _evaluate_job(root, output, variant, stage, gpu, args.force)
            except Exception as exc:
                with lock:
                    failures.append((variant, stage, str(exc)))
            finally:
                queue.task_done()

    threads = [Thread(target=worker, args=(gpu.strip(),)) for gpu in args.gpus.split(",") if gpu.strip()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    _summarize(output)


if __name__ == "__main__":
    main()
