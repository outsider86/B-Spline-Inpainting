#!/usr/bin/env python3
"""Run the FM-equivalent evaluation matrix for the eight DINO DD checkpoints."""

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
EXPERIMENT = ROOT / "outputs" / "DINO_DD_JOINT_H12"
CONFIG_ROOT = ROOT / "configs" / "dino_dd_joint_h12"
SUMMARY = EXPERIMENT / "summary"
VARIANTS = ("raw_h1", "raw_h2", "bspline_h1", "bspline_h2")
STAGES = ("base", "rtc")


def _environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src")
    cache = EXPERIMENT / "cache" / "model_weights"
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


def _evaluate(variant: str, stage: str, gpu: int, force: bool) -> None:
    config = CONFIG_ROOT / f"{variant}.yaml"
    checkpoint = EXPERIMENT / "checkpoints" / variant / f"{stage}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    target = SUMMARY / variant / stage
    latency = SUMMARY / "latency" / variant / f"{stage}.json"
    target.mkdir(parents=True, exist_ok=True)
    log = SUMMARY / "logs" / f"{variant}_{stage}.log"
    python = sys.executable

    for split in ("train", "test"):
        open_loop = target / f"open_loop_{split}.json"
        if force or not open_loop.is_file():
            _run(
                [
                    python, "-m", "robot_policy.cli", "evaluate_open_loop",
                    "--config", str(config), "--architecture", "discrete_joint",
                    "--checkpoint", str(checkpoint), "--split", split,
                    "--batch-size", "64", "--output", str(open_loop),
                ],
                gpu,
                log,
            )

        rtc = target / f"rtc_{split}.json"
        if force or not rtc.is_file():
            _run(
                [
                    python, "-m", "robot_policy.cli", "evaluate_rtc",
                    "--config", str(config), "--architecture", "discrete_joint",
                    "--checkpoint", str(checkpoint), "--split", split,
                    "--max-samples", "128", "--batch-size", "32",
                    "--sample-strategy", "episode_balanced",
                    "--sample-seed", "20260915", "--output", str(rtc),
                ],
                gpu,
                log,
            )

        image = target / f"rtc_inpainting_{split}.png"
        if force or not image.is_file():
            _run(
                [
                    python, "scripts/visualize_bsp_fm_rtc_inpainting.py",
                    "--config", str(config), "--architecture", "discrete_joint",
                    "--checkpoint", str(checkpoint), "--split", split,
                    "--output", str(image), "--delay", "6", "--samples", "5",
                ],
                gpu,
                log,
            )

    if force or not latency.is_file():
        _run(
            [
                python, "-m", "robot_policy.cli", "benchmark_inference",
                "--config", str(config), "--architecture", "discrete_joint",
                "--checkpoint", str(checkpoint), "--warmup", "10",
                "--iterations", "100", "--output", str(latency),
            ],
            gpu,
            log,
        )


def _wait_for_training(poll_seconds: int) -> None:
    while not (EXPERIMENT / "COMPLETE.json").is_file():
        status_path = EXPERIMENT / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            failed = {
                key: value
                for key, value in status.items()
                if isinstance(value, dict) and value.get("state") == "failed"
            }
            if failed:
                raise RuntimeError(f"training failed before evaluation: {failed}")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or len(gpus) > 4:
        parser.error("--gpus must contain one to four distinct GPU indices")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if args.wait_for_training:
        _wait_for_training(args.poll_seconds)
    elif not (EXPERIMENT / "COMPLETE.json").is_file():
        raise RuntimeError("training is incomplete; pass --wait-for-training to wait")

    queue: Queue[tuple[str, str]] = Queue()
    for variant in VARIANTS:
        for stage in STAGES:
            queue.put((variant, stage))
    failures: list[tuple[str, str, str]] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                variant, stage = queue.get_nowait()
            except Empty:
                return
            try:
                _evaluate(variant, stage, gpu, args.force)
            except BaseException as exc:
                with lock:
                    failures.append((variant, stage, str(exc)))
            finally:
                queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))

    deployment_gpus = ",".join(str(gpu) for gpu in (gpus * 8)[:8])
    deployment = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dino_dd_joint_h12_deployment.py",
            "--gpus",
            deployment_gpus,
        ],
        cwd=ROOT,
        env=_environment(gpus[0]),
    )
    if deployment.returncode:
        raise SystemExit(deployment.returncode)

    command = [
        sys.executable,
        "scripts/summarize_dino_dd_joint_h12.py",
        "--wandb-mode",
        args.wandb_mode,
    ]
    result = subprocess.run(command, cwd=ROOT, env=_environment(gpus[0]))
    if result.returncode:
        raise SystemExit(result.returncode)
    audit = subprocess.run(
        [
            sys.executable,
            "scripts/audit_dino_dd_joint_h12.py",
            "--remote-wandb",
        ],
        cwd=ROOT,
        env=_environment(gpus[0]),
    )
    if audit.returncode:
        raise SystemExit(audit.returncode)


if __name__ == "__main__":
    main()
