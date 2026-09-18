#!/usr/bin/env python3
"""Six-GPU dynamic continuation for the DiT capacity sweep."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import queue
import sys
import threading

from robot_policy.config import load_config

from run_model_size_sweep import (
    ARCHITECTURES,
    REPRESENTATIONS,
    SIZES,
    StatusLog,
    checkpoint_complete,
    ensure_joint_parent_cache,
    train_stage,
)


def task_priority(task: tuple[str, str, str], project_root: Path) -> tuple[int, int, int]:
    size, representation, architecture = task
    config = project_root / "configs" / "model_size_sweep" / f"{size}_{representation}.yaml"
    cfg = load_config(config, [f"policy.architecture={architecture}"])
    root = project_root / "outputs" / "model_size_sweep_50k_bs32" / size / representation / "checkpoints"
    base = root / f"{architecture}_base.pt"
    child = root / f"{architecture}_ttrtc.pt"
    base_done = checkpoint_complete(base, architecture, cfg.policy.model_size, cfg.train.updates)
    child_done = checkpoint_complete(child, architecture, cfg.policy.model_size, cfg.train.rtc_updates)
    if base_done and not child_done:
        progress = 0
    elif base.with_suffix(base.suffix + ".resume").exists():
        progress = 1
    else:
        progress = 2
    size_order = {"dit_l": 0, "dit_b": 1, "dit_s": 2}[size]
    architecture_order = {"fm": 0, "discrete_joint": 1}[architecture]
    return progress, size_order, architecture_order


def run_chain(task: tuple[str, str, str], *, gpu: str, project_root: Path, status: StatusLog,
              dry_run: bool) -> None:
    size, representation, architecture = task
    config = project_root / "configs" / "model_size_sweep" / f"{size}_{representation}.yaml"
    root = project_root / "outputs" / "model_size_sweep_50k_bs32" / size / representation
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str((project_root / "src").resolve())
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base = checkpoints / f"{architecture}_base.pt"
    child = checkpoints / f"{architecture}_ttrtc.pt"
    train_stage(config_path=config, architecture=architecture, stage="base", parent=None, output=base,
                gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run)
    if architecture == "discrete_joint":
        ensure_joint_parent_cache(config_path=config, parent=base, gpu=gpu, env=env, logs=logs,
                                  status=status, dry_run=dry_run)
    train_stage(config_path=config, architecture=architecture, stage="ttrtc", parent=base, output=child,
                gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("at least one GPU is required")
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "outputs" / "model_size_sweep_50k_bs32"
    lock_file = (output_root / "launcher.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another model-size sweep launcher is already active") from exc
    status = StatusLog(output_root / "sweep_status.json")
    status.record(state="parallel_launcher_started", gpus=gpus, dry_run=args.dry_run, python=sys.executable)
    pending: queue.Queue[tuple[str, str, str]] = queue.Queue()
    tasks = [(size, representation, architecture) for size in SIZES
             for representation in REPRESENTATIONS for architecture in ARCHITECTURES]
    tasks.sort(key=lambda item: task_priority(item, project_root))
    for task in tasks:
        size, representation, architecture = task
        cfg = load_config(project_root / "configs" / "model_size_sweep" / f"{size}_{representation}.yaml",
                          [f"policy.architecture={architecture}"])
        root = output_root / size / representation / "checkpoints"
        if not checkpoint_complete(root / f"{architecture}_ttrtc.pt", architecture,
                                   cfg.policy.model_size, cfg.train.rtc_updates):
            pending.put(task)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    stop_event = threading.Event()

    def worker(gpu: str) -> None:
        while not stop_event.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_chain(task, gpu=gpu, project_root=project_root, status=status, dry_run=args.dry_run)
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)
                status.record(state="parallel_task_failed", gpu=gpu, task=list(task),
                              error=f"{type(exc).__name__}: {exc}")
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"parallel-sweep-{gpu}", daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        status.record(state="parallel_launcher_interrupted")
        raise
    if errors:
        raise SystemExit("; ".join(str(error) for error in errors))
    status.record(state="parallel_launcher_complete")


if __name__ == "__main__":
    main()
