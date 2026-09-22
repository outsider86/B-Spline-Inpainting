#!/usr/bin/env python3
"""Train raw and B-spline h2 V4 FM policies on two datasets using four GPUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SAVE_EPOCHS = (100, 200, 300, 400, 500, 600)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_dir: str
    output_dir: str
    dataset_path: str
    train_episodes: int
    val_episodes: int
    train_windows: int
    val_windows: int
    updates_per_epoch: int


SPECS = (
    DatasetSpec(
        name="classify_blocks",
        config_dir="bsp_unet_v4_classify_blocks_600e",
        output_dir="BSP_UNET_V4_CLASSIFY_BLOCKS_600E",
        dataset_path="/scratch/wangpc/B-Spline-Inpainting/Data/classify_blocks_30hz",
        train_episodes=45,
        val_episodes=5,
        train_windows=88315,
        val_windows=9673,
        updates_per_epoch=1379,
    ),
    DatasetSpec(
        name="hanging_mug",
        config_dir="bsp_unet_v4_hanging_mug_600e",
        output_dir="BSP_UNET_V4_HANGING_MUG_600E",
        dataset_path="/scratch/wangpc/B-Spline-Inpainting/Data/hanging_mug_30hz",
        train_episodes=55,
        val_episodes=6,
        train_windows=26725,
        val_windows=2979,
        updates_per_epoch=417,
    ),
)
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(output: Path, gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(ROOT / "src")
    cache = output / "cache" / "model_weights"
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


def _publish_epoch_aliases(checkpoint: Path, updates_per_epoch: int) -> None:
    for epoch in SAVE_EPOCHS:
        update = epoch * updates_per_epoch
        source = checkpoint.with_name(f"{checkpoint.stem}.step_{update:06d}{checkpoint.suffix}")
        if not source.is_file():
            raise FileNotFoundError(f"missing epoch-{epoch} snapshot: {source}")
        destination = checkpoint.with_name(f"{checkpoint.stem}.epoch_{epoch:03d}{checkpoint.suffix}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        os.replace(temporary, destination)


def _worker(
    spec: DatasetSpec,
    variant: str,
    gpu: int,
    status: dict,
    lock: threading.Lock,
) -> None:
    output = ROOT / "outputs" / spec.output_dir
    config = ROOT / "configs" / spec.config_dir / f"{variant}.yaml"
    checkpoint = output / "checkpoints" / variant / "base.pt"
    log = output / "logs" / f"{variant}.log"
    job = f"{spec.name}/{variant}"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        if checkpoint.is_file():
            state = "already_complete"
        else:
            with lock:
                status["jobs"][job] = {
                    "gpu": gpu, "state": "running", "started_unix": time.time()
                }
                _atomic_json(ROOT / "outputs" / "BSP_UNET_V4_TWO_TASKS_600E_STATUS.json", status)
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
                    env=_environment(output, gpu),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"training failed ({result.returncode}); see {log}")
            state = "complete"
        _publish_epoch_aliases(checkpoint, spec.updates_per_epoch)
        with lock:
            status["jobs"][job] = {
                "gpu": gpu, "state": state, "completed_unix": time.time()
            }
            _atomic_json(ROOT / "outputs" / "BSP_UNET_V4_TWO_TASKS_600E_STATUS.json", status)
    except BaseException as exc:
        with lock:
            status["jobs"][job] = {
                "gpu": gpu, "state": "failed", "error": str(exc),
                "failed_unix": time.time(),
            }
            _atomic_json(ROOT / "outputs" / "BSP_UNET_V4_TWO_TASKS_600E_STATUS.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,4,5")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error("--gpus must contain exactly four distinct GPU indices")

    jobs = [(spec, variant) for spec in SPECS for variant in VARIANTS]
    for spec, variant in jobs:
        output = ROOT / "outputs" / spec.output_dir
        representation = "raw" if variant == "fm_raw_h2" else "bspline"
        prepared = output / "cache" / "prepared" / representation
        if not (prepared / "action_manifest.json").is_file():
            raise FileNotFoundError(f"missing prepared targets: {prepared}")
        rgb = output / "cache" / "rgb84" / "manifest.json"
        if not rgb.is_file():
            raise FileNotFoundError(f"missing complete RGB cache manifest: {rgb}")

    status = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "protocol": {
            "epochs": 600,
            "batch_size": 64,
            "checkpoint_epochs": list(SAVE_EPOCHS),
            "full_validation_every_epochs": 1,
            "observation_horizon": 2,
            "cameras_per_timestep": 2,
            "images_per_example": 4,
            "vision_encoder": "ResNet18 trained from scratch",
            "test_episodes": 0,
        },
        "datasets": {
            spec.name: {
                "path": spec.dataset_path,
                "train_episodes": spec.train_episodes,
                "val_episodes": spec.val_episodes,
                "train_windows": spec.train_windows,
                "val_windows": spec.val_windows,
                "updates_per_epoch": spec.updates_per_epoch,
                "total_updates": spec.updates_per_epoch * 600,
            }
            for spec in SPECS
        },
        "gpus": gpus,
        "jobs": {},
    }
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker, args=(*job, gpu, status, lock))
        for job, gpu in zip(jobs, gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [name for name, value in status["jobs"].items() if value["state"] == "failed"]
    if failures:
        raise SystemExit(f"failed jobs: {failures}")
    _atomic_json(
        ROOT / "outputs" / "BSP_UNET_V4_TWO_TASKS_600E_COMPLETE.json",
        {"completed_unix": time.time(), "jobs": [f"{s.name}/{v}" for s, v in jobs]},
    )


if __name__ == "__main__":
    main()
