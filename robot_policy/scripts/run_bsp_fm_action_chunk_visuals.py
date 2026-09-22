#!/usr/bin/env python3
"""Run the standard scratch-vs-oracle-RTC visualization matrix for BSP FM."""

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


ROOT = Path(__file__).resolve().parents[1]
VISUALIZER = ROOT / "scripts" / "visualize_bsp_from_scratch_vs_oracle_rtc.py"
VARIANTS = ("fm_raw_h1", "fm_raw_h2", "fm_bspline_h1", "fm_bspline_h2")
STAGES = ("base", "rtc")


def _run_job(
    output_root: Path,
    variant: str,
    stage: str,
    gpu: str,
    prefix: int,
    samples: int,
    seed: int,
    split: str,
    force: bool,
) -> None:
    target = output_root / variant / stage
    report = target / "report.json"
    if report.is_file() and not force:
        return
    target.mkdir(parents=True, exist_ok=True)
    log = output_root / "logs" / f"{variant}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(VISUALIZER),
        "--config",
        str(ROOT / "configs" / "bsp_unet_v3" / f"{variant}.yaml"),
        "--checkpoint",
        str(ROOT / "outputs" / "BSP_UNET_V3" / "checkpoints" / variant / f"{stage}.pt"),
        "--output-dir",
        str(target),
        "--split",
        split,
        "--prefix",
        str(prefix),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(ROOT / "src")
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"visual evaluation failed; see {log}")


def _summarize(output_root: Path, prefix: int, samples: int, seed: int, split: str) -> None:
    rows: list[dict[str, object]] = []
    sample_identity: tuple[list[int], list[int]] | None = None
    for variant in VARIANTS:
        for stage in STAGES:
            folder = output_root / variant / stage
            report = json.loads((folder / "report.json").read_text())
            identity = (report["episode_ids"], report["frame_indices"])
            if sample_identity is None:
                sample_identity = identity
            elif identity != sample_identity:
                raise RuntimeError(f"sample mismatch in {variant}/{stage}")
            rows.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "representation": report["action_representation"],
                    "observation_horizon": report["observation_horizon"],
                    "split": report["split"],
                    "samples": report["samples"],
                    "prefix_raw_actions": report["ground_truth_prefix_raw_actions"],
                    "scratch_full_chunk_physical_mse": report[
                        "scratch_full_chunk_physical_mse"
                    ],
                    "rtc_suffix_physical_mse": report["rtc_suffix_physical_mse"],
                    "rtc_prefix_physical_mse": report["rtc_prefix_physical_mse"],
                    "fixed_control_max_abs": report["fixed_control_max_abs"],
                    "plot": str(
                        (folder / "scratch_vs_ground_truth_prefix_rtc.png").resolve()
                    ),
                }
            )

    with (output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "BSP-UNet v3 FM raw/B-spline action-chunk visual evaluation",
        "split": split,
        "samples": samples,
        "prefix_raw_actions": prefix,
        "seed": seed,
        "sample_identity": {
            "episode_ids": sample_identity[0],
            "frame_indices": sample_identity[1],
        },
        "validation_passed": True,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = [f"{row['variant'].removeprefix('fm_')}\n{row['stage']}" for row in rows]
    x = list(range(len(rows)))
    width = 0.37
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.bar(
        [value - width / 2 for value in x],
        [row["scratch_full_chunk_physical_mse"] for row in rows],
        width,
        label="from scratch: full chunk",
        color="#d62728",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [row["rtc_suffix_physical_mse"] for row in rows],
        width,
        label=f"GT-prefix RTC: suffix after step {prefix}",
        color="#1f77b4",
    )
    axis.set_xticks(x, labels)
    axis.set_yscale("log")
    axis.set_ylabel("physical action MSE (log scale)")
    axis.set_title(f"BSP-UNet FM action-chunk generation [{split} split, {samples} samples]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_root / "metric_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# BSP-UNet FM action-chunk visual evaluation",
        "",
        f"Held-out `{split}` split; {samples} deterministic episode-balanced samples; ",
        f"ground-truth prefix length `{prefix}` raw actions; seed `{seed}`.",
        "",
        "Each trajectory plot overlays the ground-truth chunk, generation from scratch, and ",
        "RTC generation conditioned on the ground-truth prefix.",
        "",
        "| Variant | Stage | Scratch full MSE | RTC suffix MSE | Visualization |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        relative_plot = Path(row["plot"]).relative_to(output_root)
        lines.append(
            f"| `{row['variant']}` | {row['stage']} | "
            f"{row['scratch_full_chunk_physical_mse']:.8f} | "
            f"{row['rtc_suffix_physical_mse']:.8f} | "
            f"[plot]({relative_plot.as_posix()}) |"
        )
    lines += [
        "",
        "The shaded area in every trajectory plot is the supplied oracle prefix. This is an ",
        "open-loop diagnostic, not a physical-robot success rate.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="4,5,6")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prefix", type=int, default=6)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = (
        args.output_root
        or ROOT / "outputs" / "BSP_UNET_V3" / "action_chunk_visual_evaluation"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")

    queue: Queue[tuple[str, str]] = Queue()
    for variant in VARIANTS:
        for stage in STAGES:
            queue.put((variant, stage))
    failures: list[tuple[str, str, str]] = []
    lock = Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                variant, stage = queue.get_nowait()
            except Empty:
                return
            try:
                _run_job(
                    output_root,
                    variant,
                    stage,
                    gpu,
                    args.prefix,
                    args.samples,
                    args.seed,
                    args.split,
                    args.force,
                )
            except Exception as exc:
                with lock:
                    failures.append((variant, stage, str(exc)))
            finally:
                queue.task_done()

    threads = [Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    _summarize(output_root, args.prefix, args.samples, args.seed, args.split)


if __name__ == "__main__":
    main()
