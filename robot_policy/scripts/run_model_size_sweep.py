#!/usr/bin/env python3
"""Resumable active DiT-S/B × raw/B-spline × base/ttRTC training sweep.

One representation is assigned to each GPU so concurrent jobs never race on a
shared checkpoint manifest. Final checkpoints and periodic ``.resume`` files
are both recognized, making the launcher safe to restart.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import torch

from robot_policy.config import ACTIVE_ARCHITECTURES, load_config


SIZES = ("dit_s", "dit_b")
ARCHITECTURES = ACTIVE_ARCHITECTURES
REPRESENTATIONS = ("raw", "bspline")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_complete(path: Path, architecture: str, model_size: str, updates: int) -> bool:
    payload = checkpoint_payload(path)
    if payload is None:
        return False
    return (
        payload.get("architecture") == architecture
        and payload.get("update") == updates
        and payload.get("config", {}).get("policy", {}).get("model_size") == model_size
    )


class StatusLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.process_lock_path = path.with_suffix(path.suffix + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event: Any) -> None:
        event = {"time": utc_now(), **event}
        with self.lock:
            with self.process_lock_path.open("a+") as process_lock:
                fcntl.flock(process_lock, fcntl.LOCK_EX)
                state = json.loads(self.path.read_text()) if self.path.exists() else {"events": []}
                state["events"].append(event)
                state["last_event"] = event
                temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(state, indent=2) + "\n")
                os.replace(temporary, self.path)


def run_logged(command: list[str], *, env: dict[str, str], log_path: Path, status: StatusLog,
               gpu: str, task: str, cwd: Path, dry_run: bool) -> None:
    status.record(state="starting", gpu=gpu, task=task, command=command, log=str(log_path.resolve()))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return_code = process.wait()
    status.record(state="finished" if return_code == 0 else "failed", gpu=gpu, task=task,
                  return_code=return_code)
    if return_code:
        raise RuntimeError(f"{task} failed with exit code {return_code}; see {log_path}")


def train_stage(*, config_path: Path, architecture: str, stage: str, parent: Path | None,
                output: Path, gpu: str, env: dict[str, str], logs: Path, status: StatusLog,
                dry_run: bool) -> None:
    cfg = load_config(config_path, [f"policy.architecture={architecture}"])
    updates = cfg.train.updates if stage == "base" else cfg.train.rtc_updates
    if checkpoint_complete(output, architecture, cfg.policy.model_size, updates):
        status.record(state="skipped_complete", gpu=gpu, task=f"{cfg.policy.model_size}/{cfg.data.action_representation}/{architecture}/{stage}",
                      checkpoint=str(output.resolve()))
        return
    command = [sys.executable, "-m", "robot_policy.cli", "train_base" if stage == "base" else "finetune_rtc",
               "--config", str(config_path), "--architecture", architecture, "--output", str(output)]
    if parent is not None:
        command += ["--parent", str(parent)]
    resume = output.with_suffix(output.suffix + ".resume")
    wandb_resume = output.with_suffix(output.suffix + ".wandb.json")
    if resume.exists():
        payload = checkpoint_payload(resume)
        if (
            payload is None
            or payload.get("architecture") != architecture
            or payload.get("config", {}).get("policy", {}).get("model_size") != cfg.policy.model_size
        ):
            raise RuntimeError(f"incompatible resume checkpoint: {resume}")
        command += ["--resume", str(resume)]
    elif wandb_resume.exists():
        command += ["--wandb-resume", str(wandb_resume)]
    task = f"{cfg.policy.model_size}/{cfg.data.action_representation}/{architecture}/{stage}"
    project_root = Path(env["PYTHONPATH"]).resolve().parent
    run_logged(command, env=env, log_path=logs / f"{architecture}_{stage}.log", status=status,
               gpu=gpu, task=task, cwd=project_root, dry_run=dry_run)
    if not dry_run and not checkpoint_complete(output, architecture, cfg.policy.model_size, updates):
        raise RuntimeError(f"training returned success but final checkpoint is invalid: {output}")


def ensure_joint_parent_cache(*, config_path: Path, parent: Path, gpu: str, env: dict[str, str],
                              logs: Path, status: StatusLog, dry_run: bool) -> None:
    ensure_parent_cache(
        config_path=config_path,
        architecture="discrete_joint",
        parent=parent,
        gpu=gpu,
        env=env,
        logs=logs,
        status=status,
        dry_run=dry_run,
    )


def ensure_parent_cache(*, config_path: Path, architecture: str, parent: Path, gpu: str,
                        env: dict[str, str], logs: Path, status: StatusLog,
                        dry_run: bool) -> None:
    """Cache exact parent generations for deterministic, efficient RTC training."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"parent cache architecture must be active, got {architecture!r}")
    cfg = load_config(config_path, [f"policy.architecture={architecture}"])
    checkpoint_hash = sha256(parent.read_bytes()).hexdigest() if parent.exists() else "pending"
    project_root = Path(env["PYTHONPATH"]).resolve().parent
    output = project_root / cfg.data.prepared_path / "parent_predictions" / architecture / checkpoint_hash
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("parent_checkpoint_sha256") == checkpoint_hash:
            status.record(state="skipped_complete", gpu=gpu,
                          task=f"{cfg.policy.model_size}/{cfg.data.action_representation}/{architecture}/parent_cache",
                          cache=str(output.resolve()))
            return
    batch_size = {"DiT-S": 64, "DiT-B": 32}[cfg.policy.model_size]
    command = [sys.executable, "-m", "robot_policy.cli", "cache_parent_predictions", "--config", str(config_path),
               "--architecture", architecture, "--checkpoint", str(parent), "--output", str(output),
               "--batch-size", str(batch_size)]
    task = f"{cfg.policy.model_size}/{cfg.data.action_representation}/{architecture}/parent_cache"
    run_logged(command, env=env, log_path=logs / f"{architecture}_parent_cache.log", status=status,
               gpu=gpu, task=task, cwd=project_root, dry_run=dry_run)


def representation_worker(*, representation: str, gpu: str, project_root: Path, status: StatusLog,
                          dry_run: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str((project_root / "src").resolve())
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for size in SIZES:
        config_path = project_root / "configs" / "model_size_sweep" / f"{size}_{representation}.yaml"
        cfg = load_config(config_path)
        output_root = project_root / "outputs" / "model_size_sweep_50k_bs32" / size / representation
        checkpoints = output_root / "checkpoints"
        logs = output_root / "logs"
        for architecture in ARCHITECTURES:
            base = checkpoints / f"{architecture}_base.pt"
            ttrtc = checkpoints / f"{architecture}_ttrtc.pt"
            train_stage(config_path=config_path, architecture=architecture, stage="base", parent=None,
                        output=base, gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run)
            if architecture == "discrete_joint":
                ensure_joint_parent_cache(config_path=config_path, parent=base, gpu=gpu, env=env,
                                          logs=logs, status=status, dry_run=dry_run)
            train_stage(config_path=config_path, architecture=architecture, stage="ttrtc", parent=base,
                        output=ttrtc, gpu=gpu, env=env, logs=logs, status=status, dry_run=dry_run)
        status.record(state="size_complete", gpu=gpu, model_size=cfg.policy.model_size,
                      representation=representation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="4,5", help="physical GPU ids assigned to raw,bspline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 2:
        raise SystemExit("--gpus must provide exactly two ids: one for raw and one for B-spline")
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "outputs" / "model_size_sweep_50k_bs32"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_file = (output_root / "launcher.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another model-size sweep launcher is already active") from exc
    status = StatusLog(output_root / "sweep_status.json")
    status.record(state="launcher_started", gpus=gpus, dry_run=args.dry_run, python=sys.executable)
    errors: list[BaseException] = []

    def target(representation: str, gpu: str) -> None:
        try:
            representation_worker(representation=representation, gpu=gpu, project_root=project_root,
                                  status=status, dry_run=args.dry_run)
        except BaseException as exc:
            errors.append(exc)
            status.record(state="worker_failed", representation=representation, gpu=gpu,
                          error=f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=target, args=(rep, gpu), name=f"sweep-{rep}")
               for rep, gpu in zip(REPRESENTATIONS, gpus)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit("; ".join(str(error) for error in errors))
    status.record(state="launcher_complete")


if __name__ == "__main__":
    main()
