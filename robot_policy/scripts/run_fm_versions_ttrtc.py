#!/usr/bin/env python3
"""Fine-tune V4Full, V5, and V5.1 BSP-UNet FM policies with ttRTC.

The canonical output topology is::

    outputs/{V4,V5,V51}/{task}/{raw,bspline}/{base.pt,ttrtc.pt}

``base.pt`` is a link to the authoritative parent.  ``ttrtc.pt`` is a local
five-epoch validation-best model.  Every job starts a fresh optimizer from the
parent weights; no base optimizer state is loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
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
sys.path.insert(0, str(ROOT / "src"))

from robot_policy.config import load_config
from robot_policy.data.dataset import create_policy_dataset
from robot_policy.deployment.server_config import write_server_config


TASK_INSTRUCTIONS = {
    "classify_blocks": "Classify the blocks.",
    "hanging_mug": "Hang the mug.",
    "stacking_cup": "Stack the cups.",
}


@dataclass(frozen=True)
class Job:
    version: str
    task: str
    representation: str
    config: Path
    updates_per_epoch: int
    prepared_path: Path | None = None
    rgb_cache_path: Path | None = None

    @property
    def leaf(self) -> Path:
        return ROOT / "outputs" / self.version / self.task / self.representation

    @property
    def parent(self) -> Path:
        return self.leaf / "base.pt"

    @property
    def output(self) -> Path:
        return self.leaf / "ttrtc.pt"

    @property
    def total_updates(self) -> int:
        return 5 * self.updates_per_epoch

    @property
    def name(self) -> str:
        return f"{self.version}/{self.task}/{self.representation}"

    @property
    def horizon(self) -> int:
        return 1 if self.version == "V51" else 2

    @property
    def state_dim(self) -> int:
        return 7 if self.version == "V4" else 14


def _job_matrix() -> tuple[Job, ...]:
    definitions = {
        "V4": {
            "classify_blocks": (
                "configs/bsp_unet_v4_classify_blocks_600e",
                1379,
                "outputs/BSP_UNET_V4_FULL/BSP_UNET_V4_CLASSIFY_BLOCKS_600E/cache",
            ),
            "hanging_mug": (
                "configs/bsp_unet_v4_hanging_mug_600e",
                417,
                "outputs/BSP_UNET_V4_FULL/BSP_UNET_V4_HANGING_MUG_600E/cache",
            ),
            "stacking_cup": (
                "configs/bsp_unet_v4_stacking_cup_600e",
                636,
                "outputs/V4/stacking_cup/cache",
            ),
        },
        "V5": {
            "classify_blocks": ("configs/bsp_unet_v5/classify_blocks", 1352, None),
            "hanging_mug": ("configs/bsp_unet_v5/hanging_mug", 377, None),
            "stacking_cup": ("configs/bsp_unet_v5/stacking_cup", 590, None),
        },
        "V51": {
            "classify_blocks": ("configs/bsp_unet_v5_1/classify_blocks", 1352, None),
            "hanging_mug": ("configs/bsp_unet_v5_1/hanging_mug", 377, None),
            "stacking_cup": ("configs/bsp_unet_v5_1/stacking_cup", 590, None),
        },
    }
    jobs: list[Job] = []
    for version, tasks in definitions.items():
        for task, (config_root, updates_per_epoch, cache_root) in tasks.items():
            horizon = 1 if version == "V51" else 2
            for representation in ("raw", "bspline"):
                prepared = None
                rgb = None
                if cache_root is not None:
                    prepared = ROOT / cache_root / "prepared" / representation
                    rgb = ROOT / cache_root / "rgb84"
                jobs.append(
                    Job(
                        version=version,
                        task=task,
                        representation=representation,
                        config=ROOT
                        / config_root
                        / f"fm_{representation}_h{horizon}.yaml",
                        updates_per_epoch=updates_per_epoch,
                        prepared_path=prepared,
                        rgb_cache_path=rgb,
                    )
                )
    # Longest jobs start first so six independent GPU workers stay balanced.
    return tuple(sorted(jobs, key=lambda item: (-item.total_updates, item.name)))


def _overrides(job: Job) -> list[str]:
    values = {
        "train.rtc_updates": job.total_updates,
        "train.eval_every": job.updates_per_epoch,
        "train.save_every": job.total_updates,
        "train.keep_periodic_checkpoints": "false",
        "train.updates_per_epoch": job.updates_per_epoch,
        "train.best_checkpoint_every_epochs": 5,
        "wandb.run_suffix": (
            f"{job.version.lower()}-{job.task}-{job.representation}"
            f"-h{job.horizon}-ttrtc-5epochs"
        ),
        "wandb.output_dir": str(job.leaf / "wandb"),
        "data.parent_prediction_cache_path": str(job.leaf / "cache" / "parent_predictions"),
    }
    if job.prepared_path is not None:
        values["data.prepared_path"] = str(job.prepared_path)
    if job.rgb_cache_path is not None:
        values["data.rgb_cache_path"] = str(job.rgb_cache_path)
    result: list[str] = []
    for key, value in values.items():
        result.extend(("--set", f"{key}={value}"))
    return result


def _resolved_config(job: Job):
    pairs = _overrides(job)
    overrides = [pairs[index + 1] for index in range(0, len(pairs), 2)]
    return load_config(job.config, overrides)


def _audit_job(job: Job) -> dict[str, object]:
    if not job.parent.is_file():
        raise FileNotFoundError(f"missing parent checkpoint: {job.parent}")
    if not job.config.is_file():
        raise FileNotFoundError(f"missing config: {job.config}")
    cfg = _resolved_config(job)
    parent = torch.load(job.parent, map_location="cpu", weights_only=False, mmap=True)
    data = parent.get("config", {}).get("data", {})
    parent_state_dim = int(data.get("state_dim", 7))
    if parent.get("architecture") != "bsp_unet_fm":
        raise ValueError(f"{job.parent} is not bsp_unet_fm")
    if data.get("action_representation") != job.representation:
        raise ValueError(f"{job.parent} representation mismatch")
    if int(data.get("observation_horizon", -1)) != job.horizon:
        raise ValueError(f"{job.parent} horizon mismatch")
    if parent_state_dim != job.state_dim:
        raise ValueError(f"{job.parent} state width mismatch")
    del parent
    for required in (
        Path(cfg.data.prepared_path) / "encoder.json",
        Path(cfg.data.prepared_path) / "splits.json",
        Path(cfg.data.rgb_cache_path) / "manifest.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing cache input: {required}")
    train_count = len(create_policy_dataset(cfg, "train"))
    val_count = len(create_policy_dataset(cfg, "val"))
    actual_updates = train_count // cfg.train.batch_size
    if actual_updates != job.updates_per_epoch:
        raise ValueError(
            f"{job.name}: expected {job.updates_per_epoch} updates/epoch, got {actual_updates}"
        )
    validation_batches = (val_count + cfg.train.validation_batch_size - 1) // cfg.train.validation_batch_size
    if cfg.train.validation_max_batches < validation_batches:
        raise ValueError(f"{job.name}: validation is not full-split")
    return {
        "parent": str(job.parent.resolve()),
        "parent_sha256": sha256(job.parent.read_bytes()).hexdigest(),
        "config": str(job.config),
        "train_windows": train_count,
        "validation_windows": val_count,
        "updates_per_epoch": job.updates_per_epoch,
        "total_updates": job.total_updates,
        "full_validation_batches": validation_batches,
        "observation_horizon": job.horizon,
        "state_dim": job.state_dim,
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _environment(job: Job, gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["ROBOT_POLICY_EPISODE_CACHE_SIZE"] = "64"
    runtime = job.leaf / "cache" / "runtime"
    environment["HF_HOME"] = str(runtime / "huggingface")
    environment["TORCH_HOME"] = str(runtime / "torch")
    environment["XDG_CACHE_HOME"] = str(runtime / "xdg")
    environment["WANDB_DATA_DIR"] = str(runtime / "wandb_data")
    environment["WANDB_CACHE_DIR"] = str(runtime / "wandb_cache")
    return environment


def _command(job: Job) -> list[str]:
    return [
        sys.executable,
        "-m",
        "robot_policy.cli",
        "finetune_rtc",
        "--config",
        str(job.config),
        "--architecture",
        "bsp_unet_fm",
        "--parent",
        str(job.parent),
        "--output",
        str(job.output),
        "--updates",
        str(job.total_updates),
        *_overrides(job),
    ]


def _publish_best(job: Job) -> None:
    best = job.output.with_suffix(job.output.suffix + ".best.weights.pt")
    final = torch.load(job.output, map_location="cpu", weights_only=False, mmap=True)
    if int(final.get("update", -1)) != job.total_updates:
        raise ValueError(f"{job.output} did not finish five epochs")
    if Path(final.get("parent_checkpoint", "")).resolve() != job.parent.resolve():
        raise ValueError(f"{job.output} has the wrong parent")
    del final
    if not best.is_file():
        raise FileNotFoundError(f"missing validation-best model: {best}")
    selected = torch.load(best, map_location="cpu", weights_only=False, mmap=True)
    if selected.get("checkpoint_kind") != "best_validation_model":
        raise ValueError(f"{best} is not a validation-best model")
    if int(selected.get("checkpoint_epoch", -1)) != 5:
        raise ValueError(f"{best} is not the five-epoch boundary model")
    del selected
    os.replace(best, job.output)
    job.output.with_suffix(job.output.suffix + ".resume").unlink(missing_ok=True)
    for snapshot in job.output.parent.glob(f"{job.output.stem}.best_epoch_*.pt"):
        snapshot.unlink()

    manifest_path = job.output.parent / "checkpoint_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        digest = sha256(job.output.read_bytes()).hexdigest()
        for entry in manifest.get("checkpoints", []):
            if Path(entry.get("file_path", "")).resolve() == job.output.resolve():
                entry["sha256"] = digest
                entry["checkpoint_kind"] = "best_validation_model"
        _atomic_json(manifest_path, manifest)
    manifest_path.with_suffix(manifest_path.suffix + ".lock").unlink(missing_ok=True)
    write_server_config(
        job.output,
        job.output.with_suffix(".server.json"),
        task_instruction=TASK_INSTRUCTIONS[job.task],
    )


def _completed(job: Job) -> bool:
    if not job.output.is_file():
        return False
    try:
        payload = torch.load(job.output, map_location="cpu", weights_only=False, mmap=True)
        return (
            payload.get("checkpoint_kind") == "best_validation_model"
            and int(payload.get("checkpoint_epoch", -1)) == 5
            and Path(payload.get("parent_checkpoint", "")).resolve() == job.parent.resolve()
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not 1 <= len(gpus) <= 7 or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain one to seven distinct GPU indices")

    jobs = _job_matrix()
    status_path = ROOT / "outputs" / "TTRTC_5E_STATUS.json"
    audits = {job.name: _audit_job(job) for job in jobs}
    status: dict[str, object] = {
        "state": "audited" if args.audit_only else "running",
        "started_unix": time.time(),
        "epochs": 5,
        "fresh_optimizer": True,
        "training_semantics": {
            "delay": "random simulated raw-action delay",
            "prefix_time": 1.0,
            "suffix_time": "random Beta-derived flow time",
            "loss": "mutable suffix only",
        },
        "gpus": gpus,
        "jobs": {name: {"state": "audited", **audit} for name, audit in audits.items()},
    }
    _atomic_json(status_path, status)
    if args.audit_only:
        print(json.dumps(status, indent=2))
        return
    if args.dry_run:
        for job in jobs:
            print(" ".join(_command(job)))
        return

    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                if _completed(job):
                    state = "already_complete"
                else:
                    if job.output.exists():
                        raise FileExistsError(
                            f"incomplete output exists; inspect before resuming: {job.output}"
                        )
                    with lock:
                        status["jobs"][job.name].update(
                            state="running", gpu=gpu, started_unix=time.time()
                        )
                        _atomic_json(status_path, status)
                    job.leaf.mkdir(parents=True, exist_ok=True)
                    log = job.leaf / "ttrtc.log"
                    with log.open("a", buffering=1) as stream:
                        stream.write("\n$ " + " ".join(_command(job)) + "\n")
                        result = subprocess.run(
                            _command(job),
                            cwd=ROOT,
                            env=_environment(job, gpu),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                    if result.returncode:
                        raise RuntimeError(f"training exited {result.returncode}; see {log}")
                    _publish_best(job)
                    state = "complete"
                with lock:
                    status["jobs"][job.name].update(
                        state=state, gpu=gpu, completed_unix=time.time()
                    )
                    _atomic_json(status_path, status)
            except BaseException as exc:
                with lock:
                    status["jobs"][job.name].update(
                        state="failed", gpu=gpu, error=str(exc), failed_unix=time.time()
                    )
                    _atomic_json(status_path, status)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [
        name
        for name, value in status["jobs"].items()
        if value["state"] == "failed"
    ]
    status["state"] = "failed" if failures else "complete"
    status["completed_unix"] = time.time()
    status["failures"] = failures
    _atomic_json(status_path, status)
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
