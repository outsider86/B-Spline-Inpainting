#!/usr/bin/env python3
"""Train raw and B-spline V4 FM policies for exactly 600 loader epochs."""

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
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4_STACKING_CUP_600E_EPOCHVAL"
CACHE_ROOT = ROOT / "outputs" / "BSP_UNET_V4_STACKING_CUP_600E" / "cache"
CONFIG_ROOT = ROOT / "configs" / "bsp_unet_v4_stacking_cup_600e"
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")
UPDATES_PER_EPOCH = 636
TOTAL_EPOCHS = 600
SAVE_EPOCHS = (100, 200, 300, 400, 500, 600)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(ROOT / "src")
    cache = OUTPUT / "cache" / "model_weights"
    environment["HF_HOME"] = str(cache / "huggingface")
    environment["TORCH_HOME"] = str(cache / "torch")
    environment["XDG_CACHE_HOME"] = str(cache / "xdg")
    environment["WANDB_DATA_DIR"] = str(cache / "wandb_data")
    environment["WANDB_CACHE_DIR"] = str(cache / "wandb_cache")
    return environment


def _resume_arguments(checkpoint: Path) -> list[str]:
    recovery = checkpoint.with_suffix(checkpoint.suffix + ".resume")
    if not recovery.is_file():
        return []
    result = ["--resume", str(recovery)]
    wandb = checkpoint.with_suffix(checkpoint.suffix + ".wandb.json")
    if wandb.is_file():
        result.extend(["--wandb-resume", str(wandb)])
    return result


def _publish_epoch_aliases(checkpoint: Path) -> None:
    for epoch in SAVE_EPOCHS:
        update = epoch * UPDATES_PER_EPOCH
        source = checkpoint.with_name(f"{checkpoint.stem}.step_{update:06d}{checkpoint.suffix}")
        if not source.is_file():
            raise FileNotFoundError(f"missing epoch-{epoch} snapshot: {source}")
        destination = checkpoint.with_name(f"{checkpoint.stem}.epoch_{epoch:03d}{checkpoint.suffix}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        os.replace(temporary, destination)


def _worker(variant: str, gpu: int, status: dict, lock: threading.Lock) -> None:
    checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
    log = OUTPUT / "logs" / f"{variant}.log"
    status_path = OUTPUT / "status.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        if checkpoint.is_file():
            state = "already_complete"
        else:
            with lock:
                status[variant] = {"gpu": gpu, "state": "running", "started_unix": time.time()}
                _atomic_json(status_path, status)
            command = [
                sys.executable,
                "-m",
                "robot_policy.cli",
                "train_base",
                "--config",
                str(CONFIG_ROOT / f"{variant}.yaml"),
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
        _publish_epoch_aliases(checkpoint)
        with lock:
            status[variant] = {"gpu": gpu, "state": state, "completed_unix": time.time()}
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
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("--gpus must contain exactly two distinct GPU indices")
    for representation in ("raw", "bspline"):
        prepared = CACHE_ROOT / "prepared" / representation
        if not (prepared / "action_manifest.json").is_file():
            raise FileNotFoundError(f"missing prepared targets: {prepared}")
    rgb = CACHE_ROOT / "rgb84" / "manifest.json"
    if not rgb.is_file():
        raise FileNotFoundError(f"missing complete RGB cache manifest: {rgb}")

    status = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "dataset": "/scratch/wangpc/B-Spline-Inpainting/Data/stacking_cup_30hz",
        "split": {"train_episodes": 55, "val_episodes": 6, "test_episodes": 0},
        "windows": {"train": 40748, "val": 4537},
        "batch_size": 64,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "epochs": TOTAL_EPOCHS,
        "updates": UPDATES_PER_EPOCH * TOTAL_EPOCHS,
        "checkpoint_epochs": list(SAVE_EPOCHS),
        "full_validation_every_epochs": 1,
        "gpus": gpus,
    }
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker, args=(variant, gpu, status, lock))
        for variant, gpu in zip(VARIANTS, gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [name for name in VARIANTS if status.get(name, {}).get("state") == "failed"]
    if failures:
        raise SystemExit(f"failed variants: {failures}")
    _atomic_json(
        OUTPUT / "COMPLETE.json",
        {
            "completed_unix": time.time(),
            "variants": list(VARIANTS),
            "epochs": TOTAL_EPOCHS,
            "updates": UPDATES_PER_EPOCH * TOTAL_EPOCHS,
            "checkpoint_epochs": list(SAVE_EPOCHS),
        },
    )


if __name__ == "__main__":
    main()
