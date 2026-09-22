#!/usr/bin/env python3
"""Train a two-variant V4 h2 Flow-Matching small-batch ablation."""

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


def _worker(
    variant: str,
    gpu: int,
    status: dict,
    lock: threading.Lock,
    status_path: Path,
) -> None:
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
                    "state": "running",
                    "started_unix": time.time(),
                }
                _atomic_json(status_path, status)
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
                "state": state,
                "completed_unix": time.time(),
            }
            _atomic_json(status_path, status)
    except BaseException as exc:
        with lock:
            status[variant] = {
                "gpu": gpu,
                "state": "failed",
                "error": str(exc),
                "failed_unix": time.time(),
            }
            _atomic_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--batch", type=int, choices=(2, 4), default=4)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("--gpus must contain exactly two distinct GPU indices")
    tag = args.run_tag.strip().replace("-", "_")
    if tag and not tag.replace("_", "").isalnum():
        parser.error("--run-tag may contain only letters, numbers, underscores, and hyphens")
    suffix = f"_batch{args.batch}" + (f"_{tag}" if tag else "")
    variants = (f"fm_raw_h2{suffix}", f"fm_bspline_h2{suffix}")
    run_label = f"batch{args.batch}" + (f"_{tag}" if tag else "")
    status_path = OUTPUT / f"status_{run_label}.json"
    for representation in ("raw", "bspline"):
        prepared = OUTPUT / "cache" / "prepared" / representation
        if not (prepared / "action_manifest.json").is_file():
            raise FileNotFoundError(f"missing prepared V4 targets: {prepared}")

    status: dict = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "scope": f"V4 BSP-UNet FM h2 batch-{args.batch} ablation; raw and B-spline; 50k",
        "batch_size": args.batch,
        "effective_batch_size": args.batch,
        "approximate_epochs_at_50k": 50_000 * args.batch / 28_557,
        "split_contract": {"train_episodes": 47, "val_episodes": 5, "test_episodes": 0},
        "validation_contract": {
            "batch_size": 64 if tag == "fullval" else args.batch,
            "samples": 3149 if tag == "fullval" else min(3149, 64 * args.batch),
            "complete_split": tag == "fullval",
        },
        "reserved_gpus": gpus,
    }
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(variant, gpus[index], status, lock, status_path),
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
        OUTPUT / f"COMPLETE_{run_label.upper()}.json",
        {"completed_unix": time.time(), "variants": list(variants), "updates": 50_000},
    )


if __name__ == "__main__":
    main()
