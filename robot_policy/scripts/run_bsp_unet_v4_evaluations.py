#!/usr/bin/env python3
"""Wait for and evaluate the two BSP-UNet v4 Flow-Matching checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4"
CONFIG_ROOT = ROOT / "configs" / "bsp_unet_v4"
SUMMARY = OUTPUT / "summary"
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")


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
            command, cwd=ROOT, env=_environment(gpu),
            stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"evaluation failed ({result.returncode}); see {log}")


def _evaluate(variant: str, gpu: int, force: bool) -> None:
    config = CONFIG_ROOT / f"{variant}.yaml"
    checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
    target = SUMMARY / variant / "base"
    latency = SUMMARY / "latency" / f"{variant}.json"
    log = SUMMARY / "logs" / f"{variant}.log"
    target.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    for split in ("train", "test"):
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
                "--sample-strategy", "episode_balanced",
                "--sample-seed", "20260915", "--output", str(rtc),
            ], gpu, log)
        image = target / f"rtc_inpainting_{split}.png"
        if force or not image.is_file():
            _run([
                python, "scripts/visualize_bsp_fm_rtc_inpainting.py",
                "--config", str(config), "--architecture", "bsp_unet_fm",
                "--checkpoint", str(checkpoint), "--split", split,
                "--output", str(image), "--delay", "6", "--samples", "5",
            ], gpu, log)
    if force or not latency.is_file():
        _run([
            python, "-m", "robot_policy.cli", "benchmark_inference",
            "--config", str(config), "--architecture", "bsp_unet_fm",
            "--checkpoint", str(checkpoint), "--warmup", "10",
            "--iterations", "100", "--output", str(latency),
        ], gpu, log)


def _wait_for_training(poll_seconds: int) -> None:
    while not (OUTPUT / "COMPLETE.json").is_file():
        status_path = OUTPUT / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            failed = {
                key: value for key, value in status.items()
                if isinstance(value, dict) and value.get("state") == "failed"
            }
            if failed:
                raise RuntimeError(f"v4 training failed: {failed}")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) < 1 or len(set(gpus)) != len(gpus) or len(gpus) > 4:
        parser.error("--gpus must contain one to four distinct GPU indices")
    if args.wait_for_training:
        _wait_for_training(args.poll_seconds)
    elif not (OUTPUT / "COMPLETE.json").is_file():
        raise RuntimeError("v4 training is incomplete; use --wait-for-training")

    queue: Queue[str] = Queue()
    for variant in VARIANTS:
        queue.put(variant)
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                variant = queue.get_nowait()
            except Empty:
                return
            try:
                _evaluate(variant, gpu, args.force)
            except BaseException as exc:
                with lock:
                    failures.append((variant, str(exc)))
            finally:
                queue.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))

    deployment = subprocess.run([
        sys.executable, "scripts/validate_bsp_unet_v4_deployment.py",
        "--gpus", ",".join(str(gpu) for gpu in (gpus * 2)[:2]),
    ], cwd=ROOT, env=_environment(gpus[0]))
    if deployment.returncode:
        raise SystemExit(deployment.returncode)
    summary = subprocess.run([
        sys.executable, "scripts/summarize_bsp_unet_v4.py",
        "--wandb-mode", args.wandb_mode,
    ], cwd=ROOT, env=_environment(gpus[0]))
    if summary.returncode:
        raise SystemExit(summary.returncode)
    audit = subprocess.run([
        sys.executable, "scripts/audit_bsp_unet_v4.py", "--remote-wandb",
    ], cwd=ROOT, env=_environment(gpus[0]))
    if audit.returncode:
        raise SystemExit(audit.returncode)


if __name__ == "__main__":
    main()
