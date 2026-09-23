#!/usr/bin/env python3
"""Run the six-lane ActionJoint V6 BSP-UNet FM training pipeline.

For each task, the launcher prepares three observation-state variants and two
action representations, trains all six base models concurrently on six GPUs,
then runs the corresponding five-epoch ttRTC fine-tunes. Tasks are processed
strictly in this order: hanging_mug, stacking_cup, classify_blocks.

Canonical local topology::

    output/NEW/<task>/<state_variant>/<raw|bspline>/{base.pt,ttrtc.pt}

The per-task RGB cache is shared only after a byte-level source-equivalence
audit. Action caches remain isolated because they contain variant-specific
state normalization statistics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "Data" / "ActionJoint"
OUTPUT_ROOT = ROOT / "output" / "NEW"
CONFIG = ROOT / "configs" / "action_joint_v6" / "common.yaml"
STATUS_PATH = OUTPUT_ROOT / "V6_STATUS.json"
AUDIT_PATH = OUTPUT_ROOT / "V6_AUDIT.json"
HF_REPO = "DiscreteRTC/dRTC"
HF_ROOT = "NewModel/V6"
BATCH_SIZE = 64
BASE_EPOCHS = 100
BASE_VALIDATION_EPOCHS = 10
RTC_EPOCHS = 5

sys.path.insert(0, str(ROOT / "src"))

from robot_policy.config import load_config
from robot_policy.deployment.server_config import write_server_config
from robot_policy.policies import load_policy_checkpoint


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str
    train_episodes: int
    val_episodes: int
    instruction: str


@dataclass(frozen=True)
class StateSpec:
    name: str
    state_dim: int
    layout: str


TASKS = (
    TaskSpec("hanging_mug", "hanging_mug_30hz_cleanup", 55, 6, "Hang the mug."),
    TaskSpec("stacking_cup", "stacking_cup_30hz_cleanup", 55, 6, "Stack the cups."),
    TaskSpec("classify_blocks", "classify_blocks_30hz_cleanup", 45, 5, "Classify the blocks."),
)
STATES = (
    StateSpec("LastCommand_Joint", 7, "last_command[0:7]"),
    StateSpec("State_Joint", 7, "current_joint_angle[0:7]"),
    StateSpec(
        "State_LastCommand_Joint",
        14,
        "current_joint_angle[0:7] + last_command[7:14]",
    ),
)
REPRESENTATIONS = ("raw", "bspline")


@dataclass(frozen=True)
class TaskStats:
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    train_windows: int
    val_windows: int
    updates_per_epoch: int
    validation_batches: int


@dataclass(frozen=True)
class Job:
    task: TaskSpec
    state: StateSpec
    representation: str
    stats: TaskStats

    @property
    def dataset_path(self) -> Path:
        return DATA_ROOT / self.state.name / self.task.dataset_name

    @property
    def task_root(self) -> Path:
        return OUTPUT_ROOT / self.task.name

    @property
    def leaf(self) -> Path:
        return self.task_root / self.state.name / self.representation

    @property
    def prepared_path(self) -> Path:
        return (
            self.task_root
            / "cache"
            / "prepared"
            / self.state.name
            / self.representation
        )

    @property
    def rgb_cache_path(self) -> Path:
        return self.task_root / "cache" / "rgb84"

    @property
    def parent_prediction_path(self) -> Path:
        return (
            self.task_root
            / "cache"
            / "parent_predictions"
            / self.state.name
            / self.representation
        )

    @property
    def name(self) -> str:
        return f"{self.task.name}/{self.state.name}/{self.representation}"

    def checkpoint(self, stage: str) -> Path:
        return self.leaf / f"{stage}.pt"

    def log(self, stage: str) -> Path:
        return self.leaf / f"{stage}.log"


_status_lock = threading.Lock()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_status() -> dict[str, Any]:
    if STATUS_PATH.is_file():
        return json.loads(STATUS_PATH.read_text())
    return {}


def _update_status(mutator) -> None:
    with _status_lock:
        status = _read_status()
        mutator(status)
        status["updated_unix"] = time.time()
        _atomic_json(STATUS_PATH, status)


def _environment(job: Job | None = None, gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["ROBOT_POLICY_EPISODE_CACHE_SIZE"] = "64"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    runtime = OUTPUT_ROOT / "_runtime"
    environment["HF_HOME"] = str(runtime / "huggingface")
    environment["TORCH_HOME"] = str(runtime / "torch")
    environment["XDG_CACHE_HOME"] = str(runtime / "xdg")
    environment["WANDB_CONFIG_DIR"] = str(runtime / "wandb_config")
    environment["WANDB_DATA_DIR"] = str(runtime / "wandb_data")
    environment["WANDB_CACHE_DIR"] = str(runtime / "wandb_cache")
    if job is not None:
        environment["WANDB_DIR"] = str(job.task_root / "wandb")
    for value in environment.values():
        if isinstance(value, str) and value.startswith(str(runtime)):
            Path(value).mkdir(parents=True, exist_ok=True)
    return environment


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _task_stats(task: TaskSpec) -> TaskStats:
    canonical = DATA_ROOT / STATES[0].name / task.dataset_name / "data" / "chunk-000"
    rows = {
        int(path.stem.rsplit("_", 1)[1]): pq.ParquetFile(path).metadata.num_rows
        for path in canonical.glob("episode_*.parquet")
    }
    ids = np.asarray(sorted(rows), dtype=np.int64)
    expected = task.train_episodes + task.val_episodes
    if len(ids) != expected:
        raise ValueError(f"{task.name}: expected {expected} episodes, found {len(ids)}")
    np.random.default_rng(20260915).shuffle(ids)
    train_ids = tuple(sorted(int(value) for value in ids[: task.train_episodes]))
    val_ids = tuple(sorted(int(value) for value in ids[task.train_episodes :]))
    train_windows = sum(rows[value] for value in train_ids)
    val_windows = sum(rows[value] for value in val_ids)
    return TaskStats(
        train_ids=train_ids,
        val_ids=val_ids,
        train_windows=train_windows,
        val_windows=val_windows,
        updates_per_epoch=train_windows // BATCH_SIZE,
        validation_batches=math.ceil(val_windows / BATCH_SIZE),
    )


def _jobs(task: TaskSpec) -> tuple[Job, ...]:
    stats = _task_stats(task)
    return tuple(
        Job(task, state, representation, stats)
        for state in STATES
        for representation in REPRESENTATIONS
    )


def _overrides(job: Job, stage: str) -> list[str]:
    updates_per_epoch = job.stats.updates_per_epoch
    values: dict[str, Any] = {
        "data.dataset_path": str(job.dataset_path),
        "data.prepared_path": str(job.prepared_path),
        "data.rgb_cache_path": str(job.rgb_cache_path),
        "data.parent_prediction_cache_path": str(job.parent_prediction_path),
        "data.action_representation": job.representation,
        "data.state_dim": job.state.state_dim,
        "data.observation_horizon": 1,
        "data.train_episodes": job.task.train_episodes,
        "data.val_episodes": job.task.val_episodes,
        "data.test_episodes": 0,
        "train.updates": BASE_EPOCHS * updates_per_epoch,
        "train.rtc_updates": RTC_EPOCHS * updates_per_epoch,
        "train.batch_size": BATCH_SIZE,
        "train.effective_batch_size": BATCH_SIZE,
        "train.validation_batch_size": BATCH_SIZE,
        "train.validation_max_batches": job.stats.validation_batches,
        "train.updates_per_epoch": updates_per_epoch,
        "train.keep_periodic_checkpoints": False,
        "wandb.output_dir": str(job.task_root / "wandb"),
        # Exactly one W&B project per task. Base and ttRTC runs remain in the
        # same project and are distinguished by tracker stage/group metadata.
        "wandb.base_project": f"robot-policy-bsp-unet-v6-{job.task.name.replace('_', '-')}",
        "wandb.rtc_project": f"robot-policy-bsp-unet-v6-{job.task.name.replace('_', '-')}",
        "wandb.run_suffix": (
            f"v6-{job.task.name}-{job.state.name.lower()}-"
            f"{job.representation}-h1-{stage}"
        ),
    }
    if stage == "base":
        values.update(
            {
                "train.eval_every": BASE_VALIDATION_EPOCHS * updates_per_epoch,
                "train.save_every": BASE_VALIDATION_EPOCHS * updates_per_epoch,
                "train.best_checkpoint_every_epochs": BASE_VALIDATION_EPOCHS,
            }
        )
    elif stage == "ttrtc":
        values.update(
            {
                "train.eval_every": updates_per_epoch,
                "train.save_every": RTC_EPOCHS * updates_per_epoch,
                "train.best_checkpoint_every_epochs": RTC_EPOCHS,
            }
        )
    else:
        raise ValueError(stage)
    result: list[str] = []
    for key, value in values.items():
        if isinstance(value, bool):
            encoded = str(value).lower()
        else:
            encoded = str(value)
        result.extend(("--set", f"{key}={encoded}"))
    return result


def _resolved_config(job: Job, stage: str):
    args = _overrides(job, stage)
    return load_config(CONFIG, [args[index + 1] for index in range(0, len(args), 2)])


def _run_logged(
    command: list[str],
    log: Path,
    *,
    job: Job | None = None,
    gpu: int | None = None,
    status_key: str | None = None,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_environment(job, gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        if status_key is not None:
            _update_status(
                lambda status: status.setdefault("jobs", {}).setdefault(
                    status_key, {}
                ).update({"pid": process.pid})
            )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"command failed ({return_code}); see {log}")


def _dataset_equivalence(task: TaskSpec) -> dict[str, Any]:
    """Prove that task videos/actions align before sharing an RGB cache."""

    records: dict[str, dict[str, Any]] = {}
    for state in STATES:
        dataset = DATA_ROOT / state.name / task.dataset_name
        video_digest = sha256()
        videos = sorted((dataset / "videos").rglob("*.mp4"))
        for path in videos:
            relative = path.relative_to(dataset).as_posix()
            file_digest = _sha256(path)
            video_digest.update(relative.encode())
            video_digest.update(str(path.stat().st_size).encode())
            video_digest.update(file_digest.encode())
        row_digest = sha256()
        parquet_files = sorted((dataset / "data").rglob("*.parquet"))
        dimensions: set[tuple[int, int]] = set()
        total_rows = 0
        for path in parquet_files:
            table = pq.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                ],
            ).to_pydict()
            observation_state = np.asarray(
                table["observation.state"], dtype=np.float32
            )
            action = np.asarray(table["action"], dtype=np.float32)
            timestamp = np.asarray(table["timestamp"], dtype=np.float32)
            frame_index = np.asarray(table["frame_index"], dtype=np.int64)
            episode_index = np.asarray(table["episode_index"], dtype=np.int64)
            dimensions.add((action.shape[1], observation_state.shape[1]))
            if action.shape[1] != 7 or observation_state.shape[1] != state.state_dim:
                raise ValueError(
                    f"{path}: expected action/state dimensions "
                    f"7/{state.state_dim}, got {action.shape[1]}/{observation_state.shape[1]}"
                )
            total_rows += len(action)
            row_digest.update(path.name.encode())
            for value in (action, timestamp, frame_index, episode_index):
                row_digest.update(value.tobytes(order="C"))
        records[state.name] = {
            "video_files": len(videos),
            "video_sha256_aggregate": video_digest.hexdigest(),
            "episodes": len(parquet_files),
            "rows": total_rows,
            "aligned_action_time_sha256": row_digest.hexdigest(),
            "action_state_dimensions": [list(value) for value in sorted(dimensions)],
        }
    video_values = {value["video_sha256_aggregate"] for value in records.values()}
    row_values = {value["aligned_action_time_sha256"] for value in records.values()}
    if len(video_values) != 1 or len(row_values) != 1:
        raise ValueError(f"{task.name}: state variants are not RGB/action aligned")
    result = {
        "task": task.name,
        "verified_unix": time.time(),
        "method": "SHA-256 over every MP4 plus exact action/timestamp/frame/episode arrays",
        "safe_shared_artifact": "cache/rgb84",
        "records": records,
    }
    _atomic_json(OUTPUT_ROOT / task.name / "cache" / "shared_rgb_equivalence.json", result)
    return result


def _valid_action_cache(job: Job) -> bool:
    manifest_path = job.prepared_path / "action_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    data = manifest.get("config", {}).get("data", {})
    return (
        Path(data.get("dataset_path", "")).resolve() == job.dataset_path.resolve()
        and data.get("action_representation") == job.representation
        and int(data.get("state_dim", -1)) == job.state.state_dim
        and manifest.get("splits", {}).get("train") == list(job.stats.train_ids)
        and manifest.get("splits", {}).get("val") == list(job.stats.val_ids)
    )


def _prepare_action(job: Job) -> None:
    if _valid_action_cache(job):
        return
    command = [
        sys.executable,
        "-m",
        "robot_policy.cli",
        "prepare_actions",
        "--config",
        str(CONFIG),
        *_overrides(job, "base"),
    ]
    _run_logged(command, job.log("prepare_actions"), job=job)
    if not _valid_action_cache(job):
        raise RuntimeError(f"invalid prepared action cache: {job.prepared_path}")


def _prepare_task(task: TaskSpec, jobs: tuple[Job, ...]) -> None:
    _update_status(
        lambda status: status.setdefault("tasks", {}).setdefault(task.name, {}).update(
            {"phase": "source_equivalence_audit", "phase_started_unix": time.time()}
        )
    )
    equivalence = _dataset_equivalence(task)
    _update_status(
        lambda status: status.setdefault("tasks", {}).setdefault(task.name, {}).update(
            {"source_equivalence": equivalence}
        )
    )
    _update_status(
        lambda status: status["tasks"][task.name].update(
            {"phase": "prepare_actions", "phase_started_unix": time.time()}
        )
    )
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_prepare_action, job): job for job in jobs}
        for future in as_completed(futures):
            future.result()

    canonical = next(
        job
        for job in jobs
        if job.state.name == "LastCommand_Joint" and job.representation == "raw"
    )
    rgb_manifest = canonical.rgb_cache_path / "manifest.json"
    valid_rgb = False
    if rgb_manifest.is_file():
        value = json.loads(rgb_manifest.read_text())
        valid_rgb = (
            value.get("complete") is True
            and int(value.get("episodes", -1)) == task.train_episodes + task.val_episodes
            and int(value.get("frames", -1))
            == canonical.stats.train_windows + canonical.stats.val_windows
        )
    if not valid_rgb:
        _update_status(
            lambda status: status["tasks"][task.name].update(
                {"phase": "prepare_shared_rgb", "phase_started_unix": time.time()}
            )
        )
        command = [
            sys.executable,
            "-m",
            "robot_policy.cli",
            "prepare_rgb_cache",
            "--config",
            str(CONFIG),
            *_overrides(canonical, "base"),
        ]
        _run_logged(
            command,
            canonical.task_root / "cache" / "prepare_rgb.log",
            job=canonical,
        )
    value = json.loads(rgb_manifest.read_text())
    if not value.get("complete") or int(value.get("frames", -1)) != (
        canonical.stats.train_windows + canonical.stats.val_windows
    ):
        raise RuntimeError(f"incomplete RGB cache: {rgb_manifest}")


def _resume_arguments(checkpoint: Path) -> list[str]:
    resume = checkpoint.with_suffix(checkpoint.suffix + ".resume")
    if not resume.is_file():
        return []
    result = ["--resume", str(resume)]
    wandb = checkpoint.with_suffix(checkpoint.suffix + ".wandb.json")
    if wandb.is_file():
        result.extend(("--wandb-resume", str(wandb)))
    return result


def _checkpoint_matches(job: Job, stage: str) -> bool:
    checkpoint = job.checkpoint(stage)
    if not checkpoint.is_file():
        return False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    try:
        config = payload.get("config", {})
        data = config.get("data", {})
        expected_type = (
            "base"
            if stage == "base"
            else "ttrtc"
            if job.representation == "raw"
            else "rtc"
        )
        return (
            payload.get("checkpoint_kind") == "best_validation_model"
            and payload.get("architecture") == "bsp_unet_fm"
            and payload.get("training_type") == expected_type
            and data.get("action_representation") == job.representation
            and int(data.get("state_dim", -1)) == job.state.state_dim
            and int(data.get("observation_horizon", -1)) == 1
            and Path(data.get("dataset_path", "")).resolve() == job.dataset_path.resolve()
        )
    finally:
        del payload


def _training_summary(payload: dict[str, Any], job: Job, stage: str) -> dict[str, Any]:
    traces = payload.get("loss_trace", [])
    history = payload.get("history", [])
    finite_gradients = [
        float(item["grad_norm"])
        for item in traces
        if "grad_norm" in item and math.isfinite(float(item["grad_norm"]))
    ]
    return {
        "job": job.name,
        "stage": stage,
        "dataset_path": str(job.dataset_path),
        "state_layout": job.state.layout,
        "state_dim": job.state.state_dim,
        "action_representation": job.representation,
        "observation_horizon": 1,
        "images_per_example": 2,
        "updates_per_epoch": job.stats.updates_per_epoch,
        "completed_updates": int(payload["update"]),
        "selected_update": int(payload.get("selected_update", payload["update"])),
        "selected_epoch": float(
            payload.get("selected_update", payload["update"])
            / job.stats.updates_per_epoch
        ),
        "best_validation_action_mse": float(payload["best_validation"]),
        "validation_history": history,
        "validation_points": len(history),
        "optimizer_steps_skipped": int(
            sum(float(item.get("optimizer_step_skipped", 0.0)) > 0 for item in traces)
        ),
        "maximum_finite_gradient_norm": max(finite_gradients, default=None),
        "wall_seconds": float(payload.get("wall_seconds", 0.0)),
        "wandb": payload.get("wandb"),
        "source_checkpoint_sha256_before_promotion": _sha256(job.checkpoint(stage)),
    }


def _refresh_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint.parent / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    digest = _sha256(checkpoint)
    found = False
    for entry in manifest.get("checkpoints", []):
        if Path(entry.get("file_path", "")).resolve() == checkpoint.resolve():
            entry["sha256"] = digest
            entry["checkpoint_kind"] = "best_validation_model"
            found = True
    if not found:
        raise RuntimeError(f"manifest does not describe {checkpoint}")
    _atomic_json(manifest_path, manifest)
    manifest_path.with_suffix(manifest_path.suffix + ".lock").unlink(missing_ok=True)


def _finalize_checkpoint(job: Job, stage: str) -> dict[str, Any]:
    checkpoint = job.checkpoint(stage)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected_updates = job.stats.updates_per_epoch * (
        BASE_EPOCHS if stage == "base" else RTC_EPOCHS
    )
    expected_validations = (
        BASE_EPOCHS // BASE_VALIDATION_EPOCHS if stage == "base" else RTC_EPOCHS
    )
    if int(payload.get("update", -1)) != expected_updates:
        raise ValueError(f"{checkpoint}: incomplete update count")
    if len(payload.get("history", [])) != expected_validations:
        raise ValueError(
            f"{checkpoint}: expected {expected_validations} validation points, "
            f"found {len(payload.get('history', []))}"
        )
    summary = _training_summary(payload, job, stage)
    del payload
    _atomic_json(checkpoint.with_suffix(".training.json"), summary)

    best = checkpoint.with_suffix(checkpoint.suffix + ".best.weights.pt")
    if not best.is_file():
        raise FileNotFoundError(best)
    selected = torch.load(best, map_location="cpu", weights_only=False, mmap=True)
    try:
        if selected.get("checkpoint_kind") != "best_validation_model":
            raise ValueError(f"{best}: not a validation-best model")
        expected_boundary = BASE_EPOCHS if stage == "base" else RTC_EPOCHS
        if int(selected.get("checkpoint_epoch", -1)) != expected_boundary:
            raise ValueError(f"{best}: wrong checkpoint boundary")
    finally:
        del selected
    os.replace(best, checkpoint)
    checkpoint.with_suffix(checkpoint.suffix + ".resume").unlink(missing_ok=True)
    for snapshot in checkpoint.parent.glob(
        f"{checkpoint.stem}.best_epoch_*{checkpoint.suffix}"
    ):
        snapshot.unlink()
    _refresh_manifest(checkpoint)
    write_server_config(
        checkpoint,
        checkpoint.with_suffix(".server.json"),
        task_instruction=job.task.instruction,
    )
    cfg = _resolved_config(job, stage)
    loaded, metadata = load_policy_checkpoint(checkpoint, cfg, "cpu")
    del loaded
    summary["published_sha256"] = _sha256(checkpoint)
    summary["parameter_counts"] = metadata.get("parameter_counts")
    _atomic_json(checkpoint.with_suffix(".training.json"), summary)
    return summary


def _run_training(job: Job, stage: str, gpu: int) -> dict[str, Any]:
    key = f"{job.name}/{stage}"
    if _checkpoint_matches(job, stage):
        summary_path = job.checkpoint(stage).with_suffix(".training.json")
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"final checkpoint exists without its training summary: {summary_path}"
            )
        summary = json.loads(summary_path.read_text())
        server_path = job.checkpoint(stage).with_suffix(".server.json")
        if not server_path.is_file():
            write_server_config(
                job.checkpoint(stage),
                server_path,
                task_instruction=job.task.instruction,
            )
        _refresh_manifest(job.checkpoint(stage))
        _update_status(
            lambda status: status.setdefault("jobs", {}).setdefault(key, {}).update(
                {"state": "already_complete", "gpu": gpu, "completed_unix": time.time()}
            )
        )
        return summary

    checkpoint = job.checkpoint(stage)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        expected_updates = job.stats.updates_per_epoch * (
            BASE_EPOCHS if stage == "base" else RTC_EPOCHS
        )
        training_finished = (
            payload.get("checkpoint_kind") != "best_validation_model"
            and int(payload.get("update", -1)) == expected_updates
        )
        del payload
        if training_finished:
            summary = _finalize_checkpoint(job, stage)
            _update_status(
                lambda status: status.setdefault("jobs", {}).setdefault(key, {}).update(
                    {
                        "state": "complete",
                        "gpu": gpu,
                        "completed_unix": time.time(),
                        "checkpoint": str(checkpoint),
                        "sha256": summary["published_sha256"],
                        "best_validation_action_mse": summary[
                            "best_validation_action_mse"
                        ],
                        "selected_epoch": summary["selected_epoch"],
                    }
                )
            )
            return summary
    _update_status(
        lambda status: status.setdefault("jobs", {}).setdefault(key, {}).update(
            {"state": "running", "gpu": gpu, "started_unix": time.time()}
        )
    )
    command = [
        sys.executable,
        "-m",
        "robot_policy.cli",
        "train_base" if stage == "base" else "finetune_rtc",
        "--config",
        str(CONFIG),
        "--architecture",
        "bsp_unet_fm",
        "--output",
        str(checkpoint),
    ]
    if stage == "ttrtc":
        parent = job.checkpoint("base")
        if not _checkpoint_matches(job, "base"):
            raise RuntimeError(f"missing finalized base parent: {parent}")
        command.extend(("--parent", str(parent)))
    command.extend(_resume_arguments(checkpoint))
    command.extend(_overrides(job, stage))
    try:
        _run_logged(
            command,
            job.log(stage),
            job=job,
            gpu=gpu,
            status_key=key,
        )
        summary = _finalize_checkpoint(job, stage)
        _update_status(
            lambda status: status["jobs"][key].update(
                {
                    "state": "complete",
                    "completed_unix": time.time(),
                    "checkpoint": str(checkpoint),
                    "sha256": summary["published_sha256"],
                    "best_validation_action_mse": summary[
                        "best_validation_action_mse"
                    ],
                    "selected_epoch": summary["selected_epoch"],
                }
            )
        )
        return summary
    except BaseException as exc:
        _update_status(
            lambda status: status.setdefault("jobs", {}).setdefault(key, {}).update(
                {"state": "failed", "failed_unix": time.time(), "error": repr(exc)}
            )
        )
        raise


def _run_stage(task: TaskSpec, jobs: tuple[Job, ...], stage: str, gpus: list[int]) -> None:
    _update_status(
        lambda status: status.setdefault("tasks", {}).setdefault(task.name, {}).update(
            {"phase": stage, "phase_started_unix": time.time()}
        )
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_run_training, job, stage, gpu): job
            for job, gpu in zip(jobs, gpus, strict=True)
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append(f"{job.name}: {exc}")
    if failures:
        raise RuntimeError(f"{task.name}/{stage} failures: " + "; ".join(failures))


def audit_all(
    require_complete: bool = True,
    *,
    tasks: tuple[TaskSpec, ...] = TASKS,
    report_path: Path = AUDIT_PATH,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    unwanted: list[str] = []
    expected_models = len(tasks) * len(STATES) * len(REPRESENTATIONS) * 2
    for task in tasks:
        for job in _jobs(task):
            for stage in ("base", "ttrtc"):
                checkpoint = job.checkpoint(stage)
                if not checkpoint.is_file():
                    missing.append(str(checkpoint))
                    continue
                if not _checkpoint_matches(job, stage):
                    raise ValueError(f"checkpoint contract mismatch: {checkpoint}")
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False, mmap=True
                )
                try:
                    entries.append(
                        {
                            "task": task.name,
                            "state_variant": job.state.name,
                            "representation": job.representation,
                            "stage": stage,
                            "path": str(checkpoint),
                            "sha256": _sha256(checkpoint),
                            "selected_update": int(payload["selected_update"]),
                            "best_validation_action_mse": float(
                                payload["validation_action_mse"]
                            ),
                            "state_dim": job.state.state_dim,
                            "observation_horizon": 1,
                        }
                    )
                finally:
                    del payload
                for required in (
                    checkpoint.with_suffix(".training.json"),
                    checkpoint.with_suffix(".server.json"),
                ):
                    if not required.is_file():
                        missing.append(str(required))
            unwanted.extend(
                str(path)
                for pattern in ("*.resume", "*.best.weights.pt", "*.best_epoch_*.pt")
                for path in job.leaf.glob(pattern)
            )
    report = {
        "audited_unix": time.time(),
        "tasks": [task.name for task in tasks],
        "expected_models": expected_models,
        "verified_models": len(entries),
        "missing": missing,
        "unwanted_training_checkpoints": sorted(set(unwanted)),
        "entries": entries,
    }
    _atomic_json(report_path, report)
    if require_complete and (len(entries) != expected_models or missing or unwanted):
        raise RuntimeError(
            f"V6 audit incomplete: models={len(entries)}/{expected_models}, "
            f"missing={len(missing)}, unwanted={len(unwanted)}"
        )
    return report


def _upload(
    report: dict[str, Any],
    *,
    result_path: Path = OUTPUT_ROOT / "V6_HF_UPLOAD.json",
    update_runtime_status: bool = True,
) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    uploads: list[dict[str, Any]] = []
    for entry in report["entries"]:
        checkpoint = Path(entry["path"])
        relative = checkpoint.relative_to(OUTPUT_ROOT).as_posix()
        files = (
            checkpoint,
            checkpoint.with_suffix(".server.json"),
            checkpoint.with_suffix(".training.json"),
        )
        operations = []
        pending: list[tuple[Path, str]] = []
        for local in files:
            remote = f"{HF_ROOT}/{local.relative_to(OUTPUT_ROOT).as_posix()}"
            operations.append(
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            )
            pending.append((local, remote))
        api.create_commit(
            repo_id=HF_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload V6 {relative}",
        )
        for local, remote in pending:
            uploads.append(
                {
                    "local": str(local),
                    "remote": remote,
                    "sha256": _sha256(local),
                    "size": local.stat().st_size,
                }
            )
            if update_runtime_status:
                _update_status(
                    lambda status: status.setdefault("upload", {}).update(
                        {"state": "running", "files_uploaded": len(uploads)}
                    )
                )
    repository = api.dataset_info(HF_REPO, files_metadata=True)
    remote_files = {item.rfilename: item for item in repository.siblings}
    missing_remote = [item["remote"] for item in uploads if item["remote"] not in remote_files]
    hash_mismatches: list[dict[str, str | None]] = []
    for item in uploads:
        if not item["remote"].endswith(".pt") or item["remote"] not in remote_files:
            continue
        lfs = getattr(remote_files[item["remote"]], "lfs", None)
        remote_sha = getattr(lfs, "sha256", None)
        if remote_sha is None and isinstance(lfs, dict):
            remote_sha = lfs.get("sha256")
        if remote_sha != item["sha256"]:
            hash_mismatches.append(
                {
                    "remote": item["remote"],
                    "local_sha256": item["sha256"],
                    "remote_sha256": remote_sha,
                }
            )
    if missing_remote or hash_mismatches:
        raise RuntimeError(
            f"remote V6 verification failed: missing={len(missing_remote)}, "
            f"hash_mismatches={len(hash_mismatches)}"
        )
    result = {
        "repo_id": HF_REPO,
        "repo_type": "dataset",
        "remote_root": HF_ROOT,
        "revision": repository.sha,
        "completed_unix": time.time(),
        "files": uploads,
        "missing_remote": missing_remote,
        "model_hash_mismatches": hash_mismatches,
    }
    _atomic_json(result_path, result)
    return result


def _initialize_status(gpus: list[int]) -> None:
    stats = {task.name: _task_stats(task) for task in TASKS}
    existing = _read_status()
    value = {
        **existing,
        "state": "running",
        "launcher_pid": os.getpid(),
        "started_unix": existing.get("started_unix", time.time()),
        "current_invocation_started_unix": time.time(),
        "output_root": str(OUTPUT_ROOT),
        "task_order": [task.name for task in TASKS],
        "gpus": gpus,
        "protocol": {
            "architecture": "BSP_UNet_FlowMatching",
            "action_space": "7D measured joint angle",
            "state_variants": {
                state.name: {"state_dim": state.state_dim, "layout": state.layout}
                for state in STATES
            },
            "representations": list(REPRESENTATIONS),
            "observation_horizon": 1,
            "images_per_example": 2,
            "base_epochs": BASE_EPOCHS,
            "base_full_validation_every_epochs": BASE_VALIDATION_EPOCHS,
            "ttrtc_epochs": RTC_EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_episodes": 0,
            "wandb_artifacts": False,
            "wandb_projects": {
                task.name: f"robot-policy-bsp-unet-v6-{task.name.replace('_', '-')}"
                for task in TASKS
            },
            "shared_cache": "one byte-audited RGB84 cache per task",
        },
        "datasets": {
            task.name: {
                "dataset_name": task.dataset_name,
                "train_episodes": task.train_episodes,
                "val_episodes": task.val_episodes,
                "train_windows": stats[task.name].train_windows,
                "val_windows": stats[task.name].val_windows,
                "updates_per_epoch": stats[task.name].updates_per_epoch,
                "base_updates": BASE_EPOCHS
                * stats[task.name].updates_per_epoch,
                "ttrtc_updates": RTC_EPOCHS
                * stats[task.name].updates_per_epoch,
            }
            for task in TASKS
        },
        "jobs": existing.get("jobs", {}),
        "tasks": existing.get("tasks", {}),
    }
    _atomic_json(STATUS_PATH, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--upload-task", choices=[task.name for task in TASKS])
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 6 or len(set(gpus)) != 6:
        parser.error("--gpus must contain exactly six distinct GPU indices")
    if sum(
        (
            args.prepare_only,
            args.audit_only,
            args.upload_only,
            args.upload_task is not None,
        )
    ) > 1:
        parser.error("prepare/audit/upload modes are mutually exclusive")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        print(json.dumps(audit_all(require_complete=True), indent=2))
        return
    if args.upload_only:
        report = audit_all(require_complete=True)
        print(json.dumps(_upload(report), indent=2))
        return
    if args.upload_task is not None:
        task = next(task for task in TASKS if task.name == args.upload_task)
        task_audit_path = OUTPUT_ROOT / f"V6_AUDIT_{task.name}.json"
        task_upload_path = OUTPUT_ROOT / f"V6_HF_UPLOAD_{task.name}.json"
        report = audit_all(
            require_complete=True,
            tasks=(task,),
            report_path=task_audit_path,
        )
        print(
            json.dumps(
                _upload(
                    report,
                    result_path=task_upload_path,
                    update_runtime_status=False,
                ),
                indent=2,
            )
        )
        return

    _initialize_status(gpus)
    try:
        for task in TASKS:
            jobs = _jobs(task)
            _prepare_task(task, jobs)
            if args.prepare_only:
                continue
            _run_stage(task, jobs, "base", gpus)
            _run_stage(task, jobs, "ttrtc", gpus)
            _update_status(
                lambda status, name=task.name: status["tasks"][name].update(
                    {"phase": "complete", "completed_unix": time.time()}
                )
            )
        if args.prepare_only:
            _update_status(lambda status: status.update({"state": "prepared"}))
            return
        report = audit_all(require_complete=True)
        if not args.skip_upload:
            _update_status(
                lambda status: status.update(
                    {"state": "uploading", "upload": {"state": "running"}}
                )
            )
            upload = _upload(report)
            _update_status(
                lambda status: status.update(
                    {"upload": {"state": "complete", **upload}}
                )
            )
        _update_status(
            lambda status: status.update(
                {"state": "complete", "completed_unix": time.time()}
            )
        )
    except BaseException as exc:
        _update_status(
            lambda status: status.update(
                {"state": "failed", "failed_unix": time.time(), "error": repr(exc)}
            )
        )
        raise


if __name__ == "__main__":
    main()
