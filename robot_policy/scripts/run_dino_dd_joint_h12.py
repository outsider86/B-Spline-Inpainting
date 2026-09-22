#!/usr/bin/env python3
"""Build the shared DINOv2 cache, then train four base + four ttRTC models."""

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
OUTPUT = ROOT / "outputs" / "DINO_DD_JOINT_H12"
CONFIG_ROOT = ROOT / "configs" / "dino_dd_joint_h12"
VARIANTS = ("raw_h1", "raw_h2", "bspline_h1", "bspline_h2")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(gpus: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["PYTHONPATH"] = str(ROOT / "src")
    # Model weights and experiment caches must stay in scratch, never $HOME.
    cache = OUTPUT / "cache" / "model_weights"
    env["HF_HOME"] = str(cache / "huggingface")
    env["TORCH_HOME"] = str(cache / "torch")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")
    return env


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n$ {' '.join(command)}\n")
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def _train_command(gpus: list[int], arguments: list[str]) -> list[str]:
    if len(gpus) == 1:
        return [sys.executable, *arguments]
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(gpus)}",
        *arguments,
    ]


def _resume_arguments(output: Path, *, fresh_wandb: bool = False) -> list[str]:
    """Resume an interrupted stage without forking its W&B run."""
    recovery = output.with_suffix(output.suffix + ".resume")
    if not recovery.is_file():
        return []
    arguments = ["--resume", str(recovery)]
    if fresh_wandb:
        arguments.append("--fresh-wandb")
        return arguments
    wandb = output.with_suffix(output.suffix + ".wandb.json")
    if wandb.is_file():
        arguments.extend(["--wandb-resume", str(wandb)])
    return arguments


def _prepare_vision(gpus: list[int], batch_size: int) -> None:
    config = CONFIG_ROOT / "raw_h1.yaml"
    log_root = OUTPUT / "logs" / "vision_cache"
    # Prime the scratch-resident timm/HuggingFace cache once to avoid seven
    # simultaneous downloads.  The very large world size assigns only episode
    # zero to this process; regular workers complete the remaining episodes.
    _run(
        [
            sys.executable,
            "-m",
            "robot_policy.cli",
            "prepare_observations",
            "--config",
            str(config),
            "--rank",
            "0",
            "--world-size",
            "1000000",
            "--batch-size",
            str(batch_size),
        ],
        log_root / "prime.log",
        _environment([gpus[0]]),
    )

    failures: list[BaseException] = []

    def worker(rank: int, gpu: int) -> None:
        try:
            _run(
                [
                    sys.executable,
                    "-m",
                    "robot_policy.cli",
                    "prepare_observations",
                    "--config",
                    str(config),
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(len(gpus)),
                    "--batch-size",
                    str(batch_size),
                ],
                log_root / f"gpu{gpu}.log",
                _environment([gpu]),
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(rank, gpu), daemon=False)
        for rank, gpu in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"vision cache failed: {failures[0]}")


def _worker(
    variant: str,
    gpus: list[int],
    status: dict,
    lock: threading.Lock,
    fresh_wandb_on_resume: bool,
) -> None:
    config = CONFIG_ROOT / f"{variant}.yaml"
    checkpoint_root = OUTPUT / "checkpoints" / variant
    base = checkpoint_root / "base.pt"
    rtc = checkpoint_root / "rtc.pt"
    log = OUTPUT / "logs" / f"{variant}.log"
    env = _environment(gpus)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    stages = [
        (
            "base",
            _train_command(
                gpus,
                [
                    "-m",
                    "robot_policy.cli",
                    "train_base",
                    "--config",
                    str(config),
                    "--architecture",
                    "discrete_joint",
                    "--output",
                    str(base),
                    *_resume_arguments(base, fresh_wandb=fresh_wandb_on_resume),
                ],
            ),
            base,
        ),
        (
            "parent_cache",
            [
                sys.executable,
                "-m",
                "robot_policy.cli",
                "cache_parent_predictions",
                "--config",
                str(config),
                "--architecture",
                "discrete_joint",
                "--checkpoint",
                str(base),
                "--batch-size",
                "32",
            ],
            None,
        ),
        (
            "rtc",
            _train_command(
                gpus,
                [
                    "-m",
                    "robot_policy.cli",
                    "finetune_rtc",
                    "--config",
                    str(config),
                    "--architecture",
                    "discrete_joint",
                    "--parent",
                    str(base),
                    "--output",
                    str(rtc),
                    *_resume_arguments(rtc, fresh_wandb=fresh_wandb_on_resume),
                ],
            ),
            rtc,
        ),
    ]
    try:
        for stage, command, expected in stages:
            if expected is not None and expected.is_file():
                state = "already_complete"
            else:
                state = "running"
                with lock:
                    status[variant] = {
                        "gpus": gpus,
                        "stage": stage,
                        "state": state,
                        "started_unix": time.time(),
                    }
                    _atomic_json(OUTPUT / "status.json", status)
                _run(command, log, env)
                state = "complete"
            with lock:
                status[variant] = {
                    "gpus": gpus,
                    "stage": stage,
                    "state": state,
                    "completed_unix": time.time(),
                }
                _atomic_json(OUTPUT / "status.json", status)
    except BaseException as exc:
        with lock:
            status[variant] = {
                "gpus": gpus,
                "stage": status.get(variant, {}).get("stage"),
                "state": "failed",
                "error": str(exc),
                "failed_unix": time.time(),
            }
            _atomic_json(OUTPUT / "status.json", status)


def _parse_groups(value: str) -> list[list[int]]:
    groups = [
        [int(gpu) for gpu in group.split(",") if gpu.strip()]
        for group in value.split(";")
        if group.strip()
    ]
    flat = [gpu for group in groups for gpu in group]
    if not groups or any(not group for group in groups) or len(flat) != len(set(flat)):
        raise ValueError("GPU groups must be non-empty and disjoint")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-groups", default="0;1;2;3")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--vision-batch-size", type=int, default=32)
    parser.add_argument("--skip-vision-cache", action="store_true")
    parser.add_argument(
        "--fresh-wandb-on-resume",
        action="store_true",
        help="resume model/optimizer state but fork a new W&B run",
    )
    args = parser.parse_args()
    groups = _parse_groups(args.gpu_groups)
    reserved_gpus = {gpu for group in groups for gpu in group}
    if len(reserved_gpus) > 4:
        raise ValueError("this experiment may reserve at most four GPUs")
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if len(groups) < len(variants):
        raise ValueError("each concurrent variant requires a dedicated GPU group")

    all_gpus = [gpu for group in groups for gpu in group]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_vision_cache:
        _prepare_vision(all_gpus, args.vision_batch_size)

    status: dict = {
        "launcher_pid": os.getpid(),
        "started_unix": time.time(),
        "architecture": "discrete_joint",
        "vision": {
            "tokenizer": "dinov2",
            "raw_patch_tokens_per_camera": 256,
            "resampled_tokens_per_camera": 32,
            "visual_tokens_h1": 64,
            "visual_tokens_h2": 128,
        },
    }
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                variant,
                groups[index],
                status,
                lock,
                args.fresh_wandb_on_resume,
            ),
            daemon=False,
        )
        for index, variant in enumerate(variants)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [
        variant
        for variant in variants
        if status.get(variant, {}).get("state") == "failed"
    ]
    if failures:
        raise SystemExit(f"failed variants: {failures}")
    _atomic_json(
        OUTPUT / "COMPLETE.json",
        {"completed_unix": time.time(), "variants": variants},
    )


if __name__ == "__main__":
    main()
