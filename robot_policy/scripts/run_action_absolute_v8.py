#!/usr/bin/env python3
"""Train and publish V8 absolute-action classify-blocks BSP-UNet FM models.

The four base lanes are StateAbsJoint/StateAbsEE crossed with raw/B-spline.
Each model observes one timestep containing the global and hand RGB frames.
After all validation-selected bases complete, four five-epoch ttRTC children
are finetuned from those exact parents.  Only final best checkpoints and their
deployment/training metadata are retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import run_action_joint_v6 as pipeline


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "Data"
OUTPUT_ROOT = ROOT / "output" / "NEW" / "V8Full"
CONFIG = ROOT / "configs" / "action_absolute_v8" / "common.yaml"
STATUS_PATH = OUTPUT_ROOT / "V8_STATUS.json"
AUDIT_PATH = OUTPUT_ROOT / "V8_AUDIT.json"
UPLOAD_PATH = OUTPUT_ROOT / "V8_HF_UPLOAD.json"
DATASET_AUDIT_PATH = OUTPUT_ROOT / "V8_DATASET_AUDIT.json"
HF_ROOT = "NewModel/V8Full"
WANDB_PROJECT = "robot-policy-bsp-unet-v8-classify-blocks"

TASKS = (
    pipeline.TaskSpec(
        "classify_blocks",
        "classify_blocks_30hz_cleanup",
        76,
        9,
        "Classify the blocks.",
    ),
)
STATES = (
    pipeline.StateSpec(
        "StateAbsJoint",
        7,
        "measured_joint_position[0:6] + measured_gripper[6:7]",
        7,
        "absolute_measured_joint_position[t+1,0:6] + measured_gripper[t+1,6:7]",
    ),
    pipeline.StateSpec(
        "StateAbsEE",
        7,
        "measured_joint_position[0:6] + measured_gripper[6:7]",
        8,
        "absolute_tcp_xyz[t+1,0:3] + quaternion_xyzw[t+1,3:7] + measured_gripper[t+1,7:8]",
    ),
)
REPRESENTATIONS = ("raw", "bspline")


def _configure_pipeline() -> None:
    pipeline.DATA_ROOT = DATA_ROOT
    pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    pipeline.CONFIG = CONFIG
    pipeline.STATUS_PATH = STATUS_PATH
    pipeline.AUDIT_PATH = AUDIT_PATH
    pipeline.HF_ROOT = HF_ROOT
    pipeline.EXPERIMENT_LABEL = "V8"
    pipeline.WANDB_PROJECT = WANDB_PROJECT
    pipeline.TASKS = TASKS
    pipeline.STATES = STATES
    pipeline.REPRESENTATIONS = REPRESENTATIONS


_configure_pipeline()


def _all_jobs() -> tuple[pipeline.Job, ...]:
    return tuple(job for task in TASKS for job in pipeline._jobs(task))


def _aggregate_file_digest(paths: list[Path], root: Path) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(pipeline._sha256(path).encode())
    return digest.hexdigest()


def _audit_dataset_contracts() -> dict[str, Any]:
    """Audit dimensions, temporal rows and exact two-camera source identity."""

    task = TASKS[0]
    result: dict[str, Any] = {
        "audited_unix": time.time(),
        "task": task.name,
        "split": {"train_episodes": 76, "val_episodes": 9, "test_episodes": 0},
        "observation_horizon": 1,
        "images_per_example": 2,
        "states": {},
    }
    video_digests: set[str] = set()
    temporal_digests: set[str] = set()
    for state in STATES:
        dataset = DATA_ROOT / state.name / task.dataset_name
        info = json.loads((dataset / "meta" / "info.json").read_text())
        modality = json.loads((dataset / "meta" / "modality.json").read_text())
        embodiment = json.loads((dataset / "meta" / "embodiment.json").read_text())
        features = info.get("features", {})
        if features.get("observation.state", {}).get("shape") != [state.state_dim]:
            raise ValueError(f"{dataset}: wrong observation.state metadata shape")
        if features.get("action", {}).get("shape") != [state.action_dim]:
            raise ValueError(f"{dataset}: wrong action metadata shape")
        if set(modality.get("video", {})) != {"global", "hand"}:
            raise ValueError(f"{dataset}: expected global and hand cameras")

        files = sorted((dataset / "data").rglob("episode_*.parquet"))
        if len(files) != 85:
            raise ValueError(f"{dataset}: expected 85 episodes, found {len(files)}")
        temporal = sha256()
        action = sha256()
        lengths: list[int] = []
        minimum_dt = math.inf
        maximum_dt = -math.inf
        for expected_episode, path in enumerate(files):
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
            observation = np.asarray(table["observation.state"], dtype=np.float32)
            target = np.asarray(table["action"], dtype=np.float32)
            timestamp = np.asarray(table["timestamp"], dtype=np.float64)
            frame = np.asarray(table["frame_index"], dtype=np.int64)
            episode = np.asarray(table["episode_index"], dtype=np.int64)
            if observation.shape != (len(frame), state.state_dim):
                raise ValueError(f"{path}: bad state shape {observation.shape}")
            if target.shape != (len(frame), state.action_dim):
                raise ValueError(f"{path}: bad action shape {target.shape}")
            if not np.isfinite(observation).all() or not np.isfinite(target).all():
                raise ValueError(f"{path}: non-finite state/action")
            if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
                raise ValueError(f"{path}: non-consecutive frame_index")
            if not np.all(episode == expected_episode):
                raise ValueError(f"{path}: episode_index mismatch")
            delta = np.diff(timestamp)
            if len(delta):
                if not np.all(delta > 0):
                    raise ValueError(f"{path}: non-increasing timestamps")
                minimum_dt = min(minimum_dt, float(delta.min()))
                maximum_dt = max(maximum_dt, float(delta.max()))
            lengths.append(len(frame))
            temporal.update(path.name.encode())
            for value in (timestamp, frame, episode):
                temporal.update(value.tobytes(order="C"))
            action.update(path.name.encode())
            action.update(target.tobytes(order="C"))

        videos = sorted((dataset / "videos").rglob("*.mp4"))
        if len(videos) != 170:
            raise ValueError(f"{dataset}: expected 170 videos, found {len(videos)}")
        video_digest = _aggregate_file_digest(videos, dataset)
        temporal_digest = temporal.hexdigest()
        video_digests.add(video_digest)
        temporal_digests.add(temporal_digest)
        result["states"][state.name] = {
            "dataset": str(dataset),
            "episodes": len(files),
            "frames": sum(lengths),
            "video_files": len(videos),
            "episode_length_min": min(lengths),
            "episode_length_max": max(lengths),
            "timestamp_step_min": minimum_dt,
            "timestamp_step_max": maximum_dt,
            "state_dim": state.state_dim,
            "action_dim": state.action_dim,
            "state_layout": state.layout,
            "action_layout": state.action_layout,
            "embodiment": embodiment,
            "video_sha256_aggregate": video_digest,
            "temporal_sha256": temporal_digest,
            "action_sha256": action.hexdigest(),
        }
    if len(video_digests) != 1 or len(temporal_digests) != 1:
        raise ValueError("StateAbsEE and StateAbsJoint are not image/time aligned")
    result["shared_rgb_cache_safe"] = True
    result["shared_rgb_cache_proof"] = (
        "identical relative MP4 names, sizes, SHA-256 bytes, timestamps, frame indices, and episode indices"
    )
    pipeline._atomic_json(DATASET_AUDIT_PATH, result)
    return result


def _shared_rgb_equivalence(task: pipeline.TaskSpec) -> dict[str, Any]:
    audit = (
        json.loads(DATASET_AUDIT_PATH.read_text())
        if DATASET_AUDIT_PATH.is_file()
        else _audit_dataset_contracts()
    )
    return {
        "task": task.name,
        "verified_unix": audit["audited_unix"],
        "method": audit["shared_rgb_cache_proof"],
        "safe_shared_artifact": "cache/rgb84",
        "records": audit["states"],
    }


def _initialize_status(gpus: list[int]) -> None:
    task = TASKS[0]
    stats = pipeline._task_stats(task)
    existing = pipeline._read_status()
    pipeline._atomic_json(
        STATUS_PATH,
        {
            **existing,
            "state": "running",
            "launcher_pid": os.getpid(),
            "started_unix": existing.get("started_unix", time.time()),
            "current_invocation_started_unix": time.time(),
            "output_root": str(OUTPUT_ROOT),
            "gpus": gpus,
            "parallel_schedule": "four base lanes, then four matching ttRTC lanes",
            "protocol": {
                "architecture": "BSP_UNet_FlowMatching",
                "task": task.name,
                "state_action_variants": [state.name for state in STATES],
                "representations": list(REPRESENTATIONS),
                "observation_horizon": 1,
                "images_per_example": 2,
                "camera_keys": ["observation.images.global", "observation.images.hand"],
                "base_epochs": pipeline.BASE_EPOCHS,
                "base_full_validation_every_epochs": pipeline.BASE_VALIDATION_EPOCHS,
                "ttrtc_epochs": pipeline.RTC_EPOCHS,
                "batch_size": pipeline.BATCH_SIZE,
                "test_episodes": 0,
                "wandb_artifacts": False,
                "wandb_project": WANDB_PROJECT,
            },
            "dataset": {
                "train_episodes": task.train_episodes,
                "val_episodes": task.val_episodes,
                "train_windows": stats.train_windows,
                "val_windows": stats.val_windows,
                "updates_per_epoch": stats.updates_per_epoch,
                "base_updates": pipeline.BASE_EPOCHS * stats.updates_per_epoch,
                "ttrtc_updates": pipeline.RTC_EPOCHS * stats.updates_per_epoch,
            },
            "jobs": existing.get("jobs", {}),
            "tasks": existing.get("tasks", {}),
        },
    )


def _prepare_all() -> None:
    # V8 variants intentionally have different actions, while their RGB and
    # temporal sources are identical. Replace V6's action-equality precondition
    # with the stronger image-byte/time alignment audit above.
    original = pipeline._dataset_equivalence
    pipeline._dataset_equivalence = _shared_rgb_equivalence
    try:
        pipeline._prepare_task(TASKS[0], pipeline._jobs(TASKS[0]))
    finally:
        pipeline._dataset_equivalence = original


def _run_stage(stage: str, gpus: list[int]) -> None:
    jobs = _all_jobs()
    if len(jobs) != 4:
        raise RuntimeError(f"expected four V8 jobs, found {len(jobs)}")
    pipeline._update_status(
        lambda status: status.setdefault("tasks", {})
        .setdefault(TASKS[0].name, {})
        .update({"phase": stage, "phase_started_unix": time.time()})
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(pipeline._run_training, job, stage, gpu): job
            for job, gpu in zip(jobs, gpus, strict=True)
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append(f"{job.name}: {exc}")
    if failures:
        raise RuntimeError(f"V8 {stage} failures: " + "; ".join(failures))


def _normalize_parent_provenance() -> None:
    for job in _all_jobs():
        base = job.checkpoint("base")
        child = job.checkpoint("ttrtc")
        if not base.is_file() or not child.is_file():
            continue
        parent_sha = pipeline._sha256(base)
        for stage, parent, digest in (
            ("base", None, None),
            ("ttrtc", str(base.resolve()), parent_sha),
        ):
            summary_path = job.checkpoint(stage).with_suffix(".training.json")
            summary = json.loads(summary_path.read_text())
            summary["parent_checkpoint"] = parent
            summary["parent_checkpoint_sha256"] = digest
            pipeline._atomic_json(summary_path, summary)


def _audit_complete() -> dict[str, Any]:
    _normalize_parent_provenance()
    return pipeline.audit_all(
        require_complete=True,
        tasks=TASKS,
        report_path=AUDIT_PATH,
    )


def _upload(report: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    result = pipeline._upload(report, result_path=UPLOAD_PATH)
    api = HfApi()
    repository = api.dataset_info(pipeline.HF_REPO, files_metadata=True)
    remote_files = {item.rfilename: item for item in repository.siblings}
    sidecars: list[dict[str, Any]] = []
    operations = []
    for job in _all_jobs():
        for name in ("encoder.json", "normalization.json"):
            local = job.prepared_path / name
            data = local.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            remote = (
                f"{HF_ROOT}/{job.task.name}/{job.state.name}/sidecars/"
                f"{job.representation}/{name}"
            )
            entry = {
                "local": str(local),
                "remote": remote,
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_sha1": blob,
                "size": len(data),
                "role": "portable_load_sidecar",
            }
            sidecars.append(entry)
            sibling = remote_files.get(remote)
            if sibling is None or sibling.blob_id != blob:
                operations.append(
                    CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                )
    if operations:
        api.create_commit(
            repo_id=pipeline.HF_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message="Upload V8 absolute-action portable load sidecars",
        )
    repository = api.dataset_info(pipeline.HF_REPO, files_metadata=True)
    remote_files = {item.rfilename: item for item in repository.siblings}
    missing = [item["remote"] for item in sidecars if item["remote"] not in remote_files]
    mismatches = [
        item["remote"]
        for item in sidecars
        if item["remote"] in remote_files
        and remote_files[item["remote"]].blob_id != item["git_blob_sha1"]
    ]
    if missing or mismatches:
        raise RuntimeError(f"V8 sidecar verification failed: {missing=}, {mismatches=}")
    result["files"].extend(sidecars)
    result["revision"] = repository.sha
    result["portable_sidecars"] = len(sidecars)
    pipeline._atomic_json(UPLOAD_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error("--gpus must contain exactly four distinct GPU indices")
    if sum((args.prepare_only, args.audit_only, args.upload_only)) > 1:
        parser.error("prepare/audit/upload modes are mutually exclusive")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        print(json.dumps(_audit_complete(), indent=2))
        return
    if args.upload_only:
        print(json.dumps(_upload(_audit_complete()), indent=2))
        return

    _initialize_status(gpus)
    try:
        audit = _audit_dataset_contracts()
        pipeline._update_status(lambda status: status.update({"dataset_audit": audit}))
        _prepare_all()
        if args.prepare_only:
            pipeline._update_status(lambda status: status.update({"state": "prepared"}))
            return
        _run_stage("base", gpus)
        _run_stage("ttrtc", gpus)
        pipeline._update_status(
            lambda status: status["tasks"][TASKS[0].name].update(
                {"phase": "complete", "completed_unix": time.time()}
            )
        )
        report = _audit_complete()
        if not args.skip_upload:
            pipeline._update_status(
                lambda status: status.update(
                    {"state": "uploading", "upload": {"state": "running"}}
                )
            )
            upload = _upload(report)
            pipeline._update_status(
                lambda status: status.update(
                    {"upload": {"state": "complete", **upload}}
                )
            )
        pipeline._update_status(
            lambda status: status.update(
                {"state": "complete", "completed_unix": time.time()}
            )
        )
    except BaseException as exc:
        pipeline._update_status(
            lambda status: status.update(
                {"state": "failed", "failed_unix": time.time(), "error": repr(exc)}
            )
        )
        raise


if __name__ == "__main__":
    main()
