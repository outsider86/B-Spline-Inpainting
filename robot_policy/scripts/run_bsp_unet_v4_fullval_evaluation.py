#!/usr/bin/env python3
"""Wait for the clean batch-4 rerun, then execute its authoritative evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4"
CONFIG_ROOT = ROOT / "configs" / "bsp_unet_v4"
SUMMARY = OUTPUT / "summary" / "authoritative_fullsplit" / "batch4"
VARIANTS = ("fm_raw_h2_batch4", "fm_bspline_h2_batch4")
BATCH = 4
SPLITS = ("train", "val")


def _environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(ROOT / "src")
    cache = OUTPUT / "cache" / "model_weights"
    environment["HF_HOME"] = str(cache / "huggingface")
    environment["TORCH_HOME"] = str(cache / "torch")
    environment["XDG_CACHE_HOME"] = str(cache / "xdg")
    return environment


def _run(command: list[str], gpu: int, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _wait_for_training(completion: Path, timeout_hours: float) -> None:
    deadline = time.time() + timeout_hours * 3600.0
    while not completion.is_file():
        status_path = OUTPUT / f"status_batch{BATCH}.json"
        if status_path.is_file():
            status = _read(status_path)
            failed = {
                key: value
                for key, value in status.items()
                if isinstance(value, dict) and value.get("state") == "failed"
            }
            if failed:
                raise RuntimeError(f"training failed: {failed}")
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for {completion}")
        time.sleep(30)


def _evaluate_variant(variant: str, gpu: int, force: bool) -> None:
    config = CONFIG_ROOT / f"{variant}.yaml"
    checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
    target = SUMMARY / variant
    log = SUMMARY / "logs" / f"{variant}.log"
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        open_loop = target / f"open_loop_{split}.json"
        if force or not open_loop.is_file():
            _run(
                [
                    sys.executable,
                    "-m",
                    "robot_policy.cli",
                    "evaluate_open_loop",
                    "--config",
                    str(config),
                    "--architecture",
                    "bsp_unet_fm",
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                    "--batch-size",
                    "64",
                    "--output",
                    str(open_loop),
                ],
                gpu,
                log,
            )
        rtc = target / f"gt_prefix_rtc_{split}.json"
        if force or not rtc.is_file():
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_fm_ground_truth_prefix.py",
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                    "--prefix",
                    "6",
                    "--batch-size",
                    "32",
                    "--seed",
                    "20260915",
                    "--output",
                    str(rtc),
                ],
                gpu,
                log,
            )


def _write_summary() -> None:
    expected_samples = {"train": 28_557, "val": 3_149}
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for split in SPLITS:
            folder = SUMMARY / variant
            open_loop = _read(folder / f"open_loop_{split}.json")
            rtc = _read(folder / f"gt_prefix_rtc_{split}.json")
            if open_loop["samples"] != expected_samples[split]:
                raise RuntimeError(
                    f"{variant} {split} open-loop covered {open_loop['samples']}, "
                    f"expected {expected_samples[split]}"
                )
            if rtc["samples"] != expected_samples[split]:
                raise RuntimeError(
                    f"{variant} {split} RTC covered {rtc['samples']}, "
                    f"expected {expected_samples[split]}"
                )
            if rtc["condition_source"] != "ground-truth prefix from the current action chunk":
                raise RuntimeError(f"{variant} {split} did not use the GT prefix")
            rows.append(
                {
                    "variant": variant,
                    "representation": open_loop["action_representation"],
                    "split": split,
                    "samples": open_loop["samples"],
                    "open_loop_from_scratch_physical_mse": open_loop[
                        "decoded_all_physical_error"
                    ]["mse"],
                    "open_loop_from_scratch_physical_mae": open_loop[
                        "decoded_all_physical_error"
                    ]["mae"],
                    "gt_prefix_actions": rtc["raw_prefix_actions"],
                    "gt_prefix_rtc_suffix_physical_mse": rtc[
                        "generated_suffix_physical_error"
                    ]["mse"],
                    "gt_prefix_rtc_suffix_physical_mae": rtc[
                        "generated_suffix_physical_error"
                    ]["mae"],
                    "rtc_samples": rtc["samples"],
                }
            )
    with (SUMMARY / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "variants": list(VARIANTS),
        "split_contract": {
            "train": 28_557,
            "val": 3_149,
            "test": "not used",
        },
        "open_loop": "generation from scratch over every window in each split",
        "rtc": (
            "base-FM PiGDM, binary representation-native hard mask, current-chunk "
            "ground-truth prefix=6; metric covers only the generated suffix"
        ),
        "visualizations": {
            "train": str((SUMMARY / "trajectory_comparison_train.png").resolve()),
            "val": str((SUMMARY / "trajectory_comparison_val.png").resolve()),
            "lines": [
                "ground truth",
                "from-scratch prediction",
                "GT-prefix RTC prediction",
            ],
        },
        "metrics": rows,
    }
    (SUMMARY / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    header = (
        "| representation | split | samples | open-loop MSE | GT-prefix RTC suffix MSE |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    table = "".join(
        f"| {row['representation']} | {row['split']} | {row['samples']} | "
        f"{row['open_loop_from_scratch_physical_mse']:.8f} | "
        f"{row['gt_prefix_rtc_suffix_physical_mse']:.8f} |\n"
        for row in rows
    )
    (SUMMARY / "SUMMARY_EN.md").write_text(
        f"# Batch-{BATCH} full-split FM evaluation\n\n"
        "Open-loop generation covers every train/validation window. GT-prefix RTC uses "
        "the current ground-truth first six actions with representation-native binary "
        "hard-mask PiGDM; its primary error is computed only on the generated suffix.\n\n"
        + header
        + table
        + "\nThe two primary figures are `trajectory_comparison_train.png` and "
        "`trajectory_comparison_val.png`. Each subplot contains exactly GT, "
        "from-scratch prediction, and GT-prefix RTC prediction.\n"
    )
    (SUMMARY / "SUMMARY_CN.md").write_text(
        f"# Batch={BATCH} 全 split FM 评估\n\n"
        "Open-loop generation 覆盖 train/validation 的全部窗口。GT-prefix RTC 使用当前 "
        "ground-truth chunk 的前 6 个 action，并通过 representation-native binary hard-mask "
        "PiGDM 生成；主指标只在未提供的 suffix 上计算。\n\n"
        + header
        + table
        + "\n两张主图为 `trajectory_comparison_train.png` 和 "
        "`trajectory_comparison_val.png`。每个子图严格包含 GT、from-scratch prediction、"
        "GT-prefix RTC prediction 三条曲线。\n"
    )


def main() -> None:
    global BATCH, SUMMARY, VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--batch", type=int, choices=(4, 64), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("--gpus must contain exactly two distinct GPU indices")
    BATCH = args.batch
    suffix = "" if BATCH == 64 else f"_batch{BATCH}"
    VARIANTS = (f"fm_raw_h2{suffix}", f"fm_bspline_h2{suffix}")
    SUMMARY = OUTPUT / "summary" / "authoritative_fullsplit" / f"batch{BATCH}"
    completion = OUTPUT / ("COMPLETE.json" if BATCH == 64 else f"COMPLETE_BATCH{BATCH}.json")
    if args.wait:
        _wait_for_training(completion, args.timeout_hours)
    elif not completion.is_file():
        raise FileNotFoundError(f"completed batch-{BATCH} checkpoint pair is missing: {completion}")
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(variant: str, gpu: int) -> None:
        try:
            _evaluate_variant(variant, gpu, args.force)
        except BaseException as exc:
            with lock:
                failures.append((variant, str(exc)))

    threads = [
        threading.Thread(target=worker, args=(variant, gpu))
        for variant, gpu in zip(VARIANTS, gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))

    raw, bspline = VARIANTS
    figure_log = SUMMARY / "logs" / "trajectory_comparisons.log"
    for split in SPLITS:
        image = SUMMARY / f"trajectory_comparison_{split}.png"
        if args.force or not image.is_file():
            _run(
                [
                    sys.executable,
                    "scripts/visualize_fm_fullval_comparison.py",
                    "--raw-config",
                    str(CONFIG_ROOT / f"{raw}.yaml"),
                    "--raw-checkpoint",
                    str(OUTPUT / "checkpoints" / raw / "base.pt"),
                    "--bspline-config",
                    str(CONFIG_ROOT / f"{bspline}.yaml"),
                    "--bspline-checkpoint",
                    str(OUTPUT / "checkpoints" / bspline / "base.pt"),
                    "--split",
                    split,
                    "--prefix",
                    "6",
                    "--samples",
                    "4",
                    "--output",
                    str(image),
                ],
                gpus[0],
                figure_log,
            )
    _write_summary()
    (SUMMARY / "COMPLETE.json").write_text(
        json.dumps(
            {
                "completed_unix": time.time(),
                "variants": list(VARIANTS),
                "splits": list(SPLITS),
                "full_split_metrics": True,
                "gt_prefix_actions": 6,
                "figures": [
                    "trajectory_comparison_train.png",
                    "trajectory_comparison_val.png",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
