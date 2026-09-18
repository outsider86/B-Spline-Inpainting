#!/usr/bin/env python3
"""Run active-policy 512-vision-token training chains on free GPUs.

Each chain is base training -> exact parent-generation cache -> ttRTC training.
The launcher is restart-safe and can run one capacity first, then the remaining
capacities, without changing checkpoint identity or W&B run provenance.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Lock, Thread

from robot_policy.config import ACTIVE_ARCHITECTURES, load_config

from run_model_size_sweep import (
    StatusLog,
    checkpoint_complete,
    ensure_parent_cache,
    train_stage,
)


VALID_SIZES = ("dit_s", "dit_b")
REPRESENTATIONS = ("raw", "bspline")


def run_chain(task: tuple[str, str, str], *, gpu: str, project: Path,
              status: StatusLog, dry_run: bool) -> None:
    size, representation, architecture = task
    config = project / "configs" / "full_vision_512" / f"{size}_{representation}.yaml"
    cfg = load_config(config, [f"policy.architecture={architecture}"])
    root = project / "outputs" / "FULL_VISION_512" / size / representation
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    base = checkpoints / f"{architecture}_base.pt"
    child = checkpoints / f"{architecture}_ttrtc.pt"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str((project / "src").resolve())
    env["HF_HOME"] = str((project.parent / ".cache" / "huggingface").resolve())
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    train_stage(
        config_path=config, architecture=architecture, stage="base", parent=None,
        output=base, gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run,
    )
    ensure_parent_cache(
        config_path=config, architecture=architecture, parent=base, gpu=gpu,
        env=env, logs=logs, status=status, dry_run=dry_run,
    )
    train_stage(
        config_path=config, architecture=architecture, stage="ttrtc", parent=base,
        output=child, gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run,
    )
    status.record(
        state="chain_complete", gpu=gpu, model_size=cfg.policy.model_size,
        representation=representation, architecture=architecture,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="dit_s", help="comma-separated active sizes: dit_s,dit_b")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--architectures", default=",".join(ACTIVE_ARCHITECTURES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sizes = tuple(item.strip() for item in args.sizes.split(",") if item.strip())
    unknown = set(sizes).difference(VALID_SIZES)
    if unknown or not sizes:
        parser.error(f"invalid sizes {sorted(unknown)}; choose from {VALID_SIZES}")
    representations = tuple(item.strip() for item in args.representations.split(",") if item.strip())
    unknown_representations = set(representations).difference(REPRESENTATIONS)
    if unknown_representations or not representations:
        parser.error(f"invalid representations {sorted(unknown_representations)}")
    architectures = tuple(item.strip() for item in args.architectures.split(",") if item.strip())
    unknown_architectures = set(architectures).difference(ACTIVE_ARCHITECTURES)
    if unknown_architectures or not architectures:
        parser.error(f"invalid architectures {sorted(unknown_architectures)}")
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus:
        parser.error("at least one GPU is required")

    project = Path(__file__).resolve().parents[1]
    output = project / "outputs" / "FULL_VISION_512"
    output.mkdir(parents=True, exist_ok=True)
    # Capacity-scoped locks allow disjoint S/B sweeps to occupy otherwise
    # idle GPUs while still preventing duplicate work for the same selection.
    lock_scope = "_".join((*sorted(sizes), *sorted(representations), *sorted(architectures)))
    lock_path = output / f"launcher.{lock_scope}.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another full-vision launcher is active") from exc

    status = StatusLog(output / "sweep_status.json")
    tasks: Queue[tuple[str, str, str]] = Queue()
    selected = [
        (size, representation, architecture)
        for size in sizes
        for representation in representations
        for architecture in architectures
    ]
    for task in selected:
        size, representation, architecture = task
        config = project / "configs" / "full_vision_512" / f"{size}_{representation}.yaml"
        cfg = load_config(config, [f"policy.architecture={architecture}"])
        checkpoint = output / size / representation / "checkpoints" / f"{architecture}_ttrtc.pt"
        if not checkpoint_complete(checkpoint, architecture, cfg.policy.model_size, cfg.train.rtc_updates):
            tasks.put(task)
    status.record(
        state="launcher_started", sizes=list(sizes), gpus=list(gpus),
        chains=tasks.qsize(), dry_run=args.dry_run, python=sys.executable,
    )

    errors: list[tuple[tuple[str, str, str], BaseException]] = []
    errors_lock = Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except Empty:
                return
            try:
                run_chain(task, gpu=gpu, project=project, status=status, dry_run=args.dry_run)
            except BaseException as exc:
                with errors_lock:
                    errors.append((task, exc))
                status.record(
                    state="chain_failed", gpu=gpu, task=list(task),
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                tasks.task_done()

    threads = [Thread(target=worker, args=(gpu,), name=f"full-vision-{gpu}") for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit("; ".join(f"{task}: {error}" for task, error in errors))
    status.record(state="launcher_complete", sizes=list(sizes), chains=len(selected))


if __name__ == "__main__":
    main()
