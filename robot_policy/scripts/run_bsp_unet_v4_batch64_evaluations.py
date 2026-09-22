#!/usr/bin/env python3
"""Evaluate a completed V4 raw/B-spline Flow-Matching batch pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4"
CONFIG_ROOT = ROOT / "configs" / "bsp_unet_v4"
def _environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src")
    cache = OUTPUT / "cache" / "model_weights"
    env["HF_HOME"] = str(cache / "huggingface")
    env["TORCH_HOME"] = str(cache / "torch")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")
    return env


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
        raise RuntimeError(f"evaluation failed ({result.returncode}); see {log}")


def _evaluate(variant: str, gpu: int, force: bool, summary: Path) -> None:
    config = CONFIG_ROOT / f"{variant}.yaml"
    checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
    target = summary / variant
    log = summary / "logs" / f"{variant}.log"
    target.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    # Validation comes first because it is the authoritative held-out split.
    for split in ("val", "train"):
        open_loop = target / f"open_loop_{split}.json"
        if force or not open_loop.is_file():
            _run([
                python, "-m", "robot_policy.cli", "evaluate_open_loop",
                "--config", str(config), "--architecture", "bsp_unet_fm",
                "--checkpoint", str(checkpoint), "--split", split,
                "--batch-size", "64", "--output", str(open_loop),
            ], gpu, log)
        rtc = target / f"rtc_{split}.json"
        if force or not rtc.is_file():
            _run([
                python, "-m", "robot_policy.cli", "evaluate_rtc",
                "--config", str(config), "--architecture", "bsp_unet_fm",
                "--checkpoint", str(checkpoint), "--split", split,
                "--max-samples", "128", "--batch-size", "32",
                "--sample-strategy", "episode_balanced_motion",
                "--sample-seed", "20260915", "--output", str(rtc),
                "--condition-source", "ground_truth",
            ], gpu, log)
        image = target / f"rtc_inpainting_{split}.png"
        if force or not image.is_file():
            _run([
                python, "scripts/visualize_bsp_fm_rtc_inpainting.py",
                "--config", str(config), "--architecture", "bsp_unet_fm",
                "--checkpoint", str(checkpoint), "--split", split,
                "--output", str(image), "--delay", "6", "--samples", "5",
                "--condition-source", "ground_truth",
            ], gpu, log)

    latency = target / "latency.json"
    if force or not latency.is_file():
        _run([
            python, "-m", "robot_policy.cli", "benchmark_inference",
            "--config", str(config), "--architecture", "bsp_unet_fm",
            "--checkpoint", str(checkpoint), "--split", "val",
            "--warmup", "10", "--iterations", "100", "--output", str(latency),
        ], gpu, log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch", type=int, choices=(2, 4, 64), default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("--gpus must contain exactly two distinct GPU indices")
    completion = OUTPUT / ("COMPLETE.json" if args.batch == 64 else f"COMPLETE_BATCH{args.batch}.json")
    if not completion.is_file():
        raise RuntimeError(f"V4 batch-{args.batch} training is incomplete")
    suffix = "" if args.batch == 64 else f"_batch{args.batch}"
    variants = (f"fm_raw_h2{suffix}", f"fm_bspline_h2{suffix}")
    summary = OUTPUT / "summary" / f"batch{args.batch}"

    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(variant: str, gpu: int) -> None:
        try:
            _evaluate(variant, gpu, args.force, summary)
        except BaseException as exc:
            with lock:
                failures.append((variant, str(exc)))

    threads = [
        threading.Thread(target=worker, args=(variant, gpus[index]))
        for index, variant in enumerate(variants)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    complete = {
        "completed_unix": time.time(),
        "variants": list(variants),
        "splits": ["train", "val"],
        "open_loop": "full split generation from scratch",
        "rtc": "128 episode-balanced motion-rich samples per split and delay; hard-mask PiGDM",
    }
    path = summary / "COMPLETE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
