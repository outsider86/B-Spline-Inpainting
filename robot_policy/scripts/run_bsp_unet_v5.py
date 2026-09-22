#!/usr/bin/env python3
"""Prepare and train the six V5 state+previous-command FM checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "outputs" / "BSP_UNET_V5_STATUS.json"
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")
BEST_EPOCHS = tuple(range(10, 101, 10))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_name: str
    output_name: str
    dataset_path: str
    train_episodes: int
    val_episodes: int
    train_windows: int
    val_windows: int
    updates_per_epoch: int

    @property
    def config_root(self) -> Path:
        return ROOT / "configs" / "bsp_unet_v5" / self.config_name

    @property
    def output_root(self) -> Path:
        return ROOT / "outputs" / self.output_name


SPECS = (
    DatasetSpec(
        "classify_blocks",
        "classify_blocks",
        "BSP_UNET_V5_CLASSIFY_BLOCKS_100E",
        "/scratch/wangpc/B-Spline-Inpainting/Data/Processed/classify_blocks_30hz_cleanup",
        45,
        5,
        86561,
        9493,
        1352,
    ),
    DatasetSpec(
        "hanging_mug",
        "hanging_mug",
        "BSP_UNET_V5_HANGING_MUG_100E",
        "/scratch/wangpc/B-Spline-Inpainting/Data/Processed/hanging_mug_30hz_cleanup",
        55,
        6,
        24150,
        2738,
        377,
    ),
    DatasetSpec(
        "stacking_cup",
        "stacking_cup",
        "BSP_UNET_V5_STACKING_CUP_100E",
        "/scratch/wangpc/B-Spline-Inpainting/Data/Processed/stacking_cup_30hz_cleanup",
        55,
        6,
        37780,
        4245,
        590,
    ),
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(spec: DatasetSpec, gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cache = spec.output_root / "cache" / "runtime"
    # All caches remain under scratch; never spill model/data caches into HOME.
    environment["HF_HOME"] = str(cache / "huggingface")
    environment["TORCH_HOME"] = str(cache / "torch")
    environment["XDG_CACHE_HOME"] = str(cache / "xdg")
    environment["WANDB_DATA_DIR"] = str(cache / "wandb_data")
    environment["WANDB_CACHE_DIR"] = str(cache / "wandb_cache")
    return environment


def _run_logged(command: list[str], log: Path, spec: DatasetSpec, gpu: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(spec, gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _prepare_actions(spec: DatasetSpec, variant: str) -> None:
    config = spec.config_root / f"{variant}.yaml"
    prepared = spec.output_root / "cache" / "prepared" / (
        "raw" if variant == "fm_raw_h2" else "bspline"
    )
    manifest = prepared / "action_manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text())
        if value.get("config", {}).get("data", {}).get("state_dim") == 14:
            return
    _run_logged(
        [sys.executable, "-m", "robot_policy.cli", "prepare_actions", "--config", str(config)],
        spec.output_root / "logs" / f"prepare_{variant}.log",
        spec,
    )


def _prepare_rgb(spec: DatasetSpec) -> None:
    output = spec.output_root / "cache" / "rgb84"
    manifest = output / "manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text())
        if (
            value.get("complete") is True
            and value.get("frames") == spec.train_windows + spec.val_windows
            and Path(value.get("source", "")).resolve() == Path(spec.dataset_path).resolve()
        ):
            return
    config = spec.config_root / "fm_raw_h2.yaml"
    _run_logged(
        [
            sys.executable,
            "-m",
            "robot_policy.cli",
            "prepare_rgb_cache",
            "--config",
            str(config),
        ],
        spec.output_root / "logs" / "prepare_rgb.log",
        spec,
    )


def prepare_all() -> None:
    failures: list[str] = []
    lock = threading.Lock()

    def action_worker(spec: DatasetSpec, variant: str) -> None:
        try:
            _prepare_actions(spec, variant)
        except BaseException as exc:
            with lock:
                failures.append(f"{spec.name}/{variant}: {exc}")

    threads = [
        threading.Thread(target=action_worker, args=(spec, variant))
        for spec in SPECS
        for variant in VARIANTS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("action preparation failed: " + "; ".join(failures))

    def rgb_worker(spec: DatasetSpec) -> None:
        try:
            _prepare_rgb(spec)
        except BaseException as exc:
            with lock:
                failures.append(f"{spec.name}/rgb: {exc}")

    threads = [threading.Thread(target=rgb_worker, args=(spec,)) for spec in SPECS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError("RGB preparation failed: " + "; ".join(failures))


def _resume_arguments(checkpoint: Path) -> list[str]:
    recovery = checkpoint.with_suffix(checkpoint.suffix + ".resume")
    if not recovery.is_file():
        return []
    result = ["--resume", str(recovery)]
    wandb = checkpoint.with_suffix(checkpoint.suffix + ".wandb.json")
    if wandb.is_file():
        result.extend(["--wandb-resume", str(wandb)])
    return result


def _verify_models(spec: DatasetSpec, variant: str, checkpoint: Path) -> None:
    expected = [
        checkpoint.with_name(f"{checkpoint.stem}.best_epoch_{epoch:03d}{checkpoint.suffix}")
        for epoch in BEST_EPOCHS
    ]
    missing = [str(path) for path in (checkpoint, *expected) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V5 checkpoints: " + ", ".join(missing))
    for epoch, path in zip(BEST_EPOCHS, expected):
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        try:
            if payload.get("checkpoint_epoch") != epoch:
                raise ValueError(f"{path} has wrong checkpoint_epoch")
            if payload.get("config", {}).get("data", {}).get("state_dim") != 14:
                raise ValueError(f"{path} is not a 14D V5 checkpoint")
            if payload.get("architecture") != "bsp_unet_fm":
                raise ValueError(f"{path} has wrong architecture")
        finally:
            del payload
    final = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    try:
        if final.get("config", {}).get("data", {}).get("state_dim") != 14:
            raise ValueError(f"{checkpoint} is not a 14D V5 checkpoint")
        if int(final.get("update", -1)) != spec.updates_per_epoch * 100:
            raise ValueError(f"{checkpoint} did not complete 100 epochs")
    finally:
        del final


def _train_worker(
    spec: DatasetSpec,
    variant: str,
    gpu: int,
    status: dict,
    lock: threading.Lock,
) -> None:
    job = f"{spec.name}/{variant}"
    checkpoint = spec.output_root / "checkpoints" / variant / "base.pt"
    log = spec.output_root / "logs" / f"{variant}.log"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock:
            status["jobs"][job] = {
                "gpu": gpu,
                "state": "running",
                "started_unix": time.time(),
            }
            _atomic_json(STATUS_PATH, status)
        if not checkpoint.is_file():
            command = [
                sys.executable,
                "-m",
                "robot_policy.cli",
                "train_base",
                "--config",
                str(spec.config_root / f"{variant}.yaml"),
                "--architecture",
                "bsp_unet_fm",
                "--output",
                str(checkpoint),
                *_resume_arguments(checkpoint),
            ]
            _run_logged(command, log, spec, gpu)
            state = "complete"
        else:
            state = "already_complete"
        _verify_models(spec, variant, checkpoint)
        # The rolling recovery state is useful only while a job is active.  V5
        # publishes model checkpoints, not optimizer archives.
        checkpoint.with_suffix(checkpoint.suffix + ".resume").unlink(missing_ok=True)
        with lock:
            status["jobs"][job] = {
                "gpu": gpu,
                "state": state,
                "completed_unix": time.time(),
            }
            _atomic_json(STATUS_PATH, status)
    except BaseException as exc:
        with lock:
            status["jobs"][job] = {
                "gpu": gpu,
                "state": "failed",
                "error": str(exc),
                "failed_unix": time.time(),
            }
            _atomic_json(STATUS_PATH, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.skip_prepare:
        parser.error("--prepare-only and --skip-prepare are mutually exclusive")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not 1 <= len(gpus) <= 6 or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain one to six distinct GPU indices")

    if not args.skip_prepare:
        prepare_all()
    if args.prepare_only:
        return

    status = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "protocol": {
            "version": "v5",
            "state_dim": 14,
            "state_layout": "measured_state[0:7] + previous_command[7:14]",
            "epochs": 100,
            "batch_size": 64,
            "full_validation_every_epochs": 1,
            "best_checkpoint_every_epochs": 10,
            "best_checkpoint_semantics": "exact best validation action_mse observed through boundary",
            "observation_horizon": 2,
            "images_per_example": 4,
            "vision_encoder": "independent ResNet18 trained from scratch",
            "wandb_artifacts": False,
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
                "total_updates": spec.updates_per_epoch * 100,
            }
            for spec in SPECS
        },
        "gpus": gpus,
        "jobs": {},
    }
    _atomic_json(STATUS_PATH, status)
    jobs = [(spec, variant) for spec in SPECS for variant in VARIANTS]
    lock = threading.Lock()
    pending: queue.Queue[tuple[DatasetSpec, str]] = queue.Queue()
    for job in jobs:
        pending.put(job)

    def gpu_worker(gpu: int) -> None:
        while True:
            try:
                spec, variant = pending.get_nowait()
            except queue.Empty:
                return
            try:
                _train_worker(spec, variant, gpu, status, lock)
            finally:
                pending.task_done()

    threads = [
        threading.Thread(target=gpu_worker, args=(gpu,)) for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [name for name, value in status["jobs"].items() if value["state"] == "failed"]
    if failures:
        raise SystemExit(f"failed V5 jobs: {failures}")
    _atomic_json(
        ROOT / "outputs" / "BSP_UNET_V5_COMPLETE.json",
        {
            "completed_unix": time.time(),
            "jobs": [f"{spec.name}/{variant}" for spec, variant in jobs],
            "best_checkpoint_epochs": list(BEST_EPOCHS),
        },
    )


if __name__ == "__main__":
    main()
