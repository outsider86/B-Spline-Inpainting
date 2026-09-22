#!/usr/bin/env python3
"""Train the two h2 Flow-Matching BSP-UNet v4 checkpoints for 50k updates."""

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
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src")
    cache = OUTPUT / "cache" / "model_weights"
    env["HF_HOME"] = str(cache / "huggingface")
    env["TORCH_HOME"] = str(cache / "torch")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")
    return env


def _resume_arguments(output: Path) -> list[str]:
    recovery = output.with_suffix(output.suffix + ".resume")
    if not recovery.is_file():
        return []
    result = ["--resume", str(recovery)]
    wandb = output.with_suffix(output.suffix + ".wandb.json")
    if wandb.is_file():
        result.extend(["--wandb-resume", str(wandb)])
    return result


def _prepare_actions(config: Path) -> None:
    """Build split-dependent targets before any training worker starts."""
    command = [
        sys.executable,
        "-m",
        "robot_policy.cli",
        "prepare_actions",
        "--config",
        str(config),
    ]
    result = subprocess.run(command, cwd=ROOT, env=_environment(0))
    if result.returncode:
        raise RuntimeError(f"action preparation failed for {config}")


def _worker(variant: str, gpu: int, status: dict, lock: threading.Lock) -> None:
    checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log = OUTPUT / "logs" / f"{variant}.log"
    config = CONFIG_ROOT / f"{variant}.yaml"
    try:
        if checkpoint.is_file():
            state = "already_complete"
        else:
            with lock:
                status[variant] = {
                    "gpu": gpu,
                    "stage": "base",
                    "state": "running",
                    "started_unix": time.time(),
                }
                _atomic_json(OUTPUT / "status.json", status)
            command = [
                sys.executable,
                "-m",
                "robot_policy.cli",
                "train_base",
                "--config",
                str(config),
                "--architecture",
                "bsp_unet_fm",
                "--output",
                str(checkpoint),
                *_resume_arguments(checkpoint),
            ]
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
                raise RuntimeError(f"training failed ({result.returncode}); see {log}")
            state = "complete"
        with lock:
            status[variant] = {
                "gpu": gpu,
                "stage": "base",
                "state": state,
                "completed_unix": time.time(),
            }
            _atomic_json(OUTPUT / "status.json", status)
    except BaseException as exc:
        with lock:
            status[variant] = {
                "gpu": gpu,
                "stage": "base",
                "state": "failed",
                "error": str(exc),
                "failed_unix": time.time(),
            }
            _atomic_json(OUTPUT / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or len(gpus) > 4:
        parser.error("--gpus must contain one to four distinct GPU indices")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or len(variants) != len(set(variants)):
        parser.error(f"invalid variants: {unknown or variants}")
    if len(gpus) < len(variants):
        parser.error("each concurrently trained variant requires one GPU")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        config = CONFIG_ROOT / f"{variant}.yaml"
        # The V4 split changes both train-only normalization and, for B-spline,
        # control calibration. Never reuse V3's 42/5/5 prepared targets.
        prepared = OUTPUT / "cache" / "prepared" / (
            "raw" if variant == "fm_raw_h2" else "bspline"
        )
        if not (prepared / "action_manifest.json").is_file():
            _prepare_actions(config)
    status: dict = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "scope": "BSP-UNet Flow Matching h2 only; raw and B-spline; 50k base",
        "observation_contract": {
            "timesteps": 2,
            "cameras_per_timestep": 2,
            "images_per_example": 4,
        },
        "split_contract": {"train_episodes": 47, "val_episodes": 5, "test_episodes": 0},
        "maximum_reserved_gpus": 4,
        "reserved_gpus": gpus[: len(variants)],
    }
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(variant, gpus[index], status, lock),
            daemon=False,
        )
        for index, variant in enumerate(variants)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [
        variant for variant in variants
        if status.get(variant, {}).get("state") == "failed"
    ]
    if failures:
        raise SystemExit(f"failed variants: {failures}")
    _atomic_json(
        OUTPUT / "COMPLETE.json",
        {"completed_unix": time.time(), "variants": variants, "updates": 50_000},
    )


if __name__ == "__main__":
    main()
