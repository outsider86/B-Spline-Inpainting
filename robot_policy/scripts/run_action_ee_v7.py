#!/usr/bin/env python3
"""Run the six-lane ActionEE V7 BSP-UNet flow-matching pipeline.

All three tasks and both action representations train concurrently. The six
base jobs finish before the corresponding six five-epoch ttRTC jobs start.
The implementation deliberately reuses the audited V6 training, finalization,
load-validation, cleanup, and Hugging Face publication primitives.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import run_action_joint_v6 as pipeline


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "Data" / "ActionEE" / "State_EE"
OUTPUT_ROOT = ROOT / "output" / "NEW" / "V7Full"
CONFIG = ROOT / "configs" / "action_ee_v7" / "common.yaml"
STATUS_PATH = OUTPUT_ROOT / "V7_STATUS.json"
AUDIT_PATH = OUTPUT_ROOT / "V7_AUDIT.json"
UPLOAD_PATH = OUTPUT_ROOT / "V7_HF_UPLOAD.json"
DATASET_AUDIT_PATH = OUTPUT_ROOT / "V7_DATASET_AUDIT.json"
HF_ROOT = "NewModel/V7Full"
WANDB_PROJECT = "robot-policy-bsp-unet-v7-ee"

TASKS = (
    pipeline.TaskSpec(
        "hanging_mug", "hanging_mug_30hz_cleanup", 55, 6, "Hang the mug."
    ),
    pipeline.TaskSpec(
        "stacking_cup", "stacking_cup_30hz_cleanup", 55, 6, "Stack the cups."
    ),
    pipeline.TaskSpec(
        "classify_blocks",
        "classify_blocks_30hz_cleanup",
        45,
        5,
        "Classify the blocks.",
    ),
)
STATES = (
    pipeline.StateSpec(
        "State_EE",
        8,
        "measured_tcp_position[0:3] + measured_tcp_quaternion_xyzw[3:7] + "
        "measured_gripper[7:8]",
    ),
)
REPRESENTATIONS = ("raw", "bspline")
EXPECTED_ACTION_LAYOUT = (
    "local_delta_tcp_translation[t->t+1] + "
    "local_delta_tcp_rotvec[t->t+1] + measured_gripper[t+1]"
)
EXPECTED_STATE_LAYOUT = (
    "measured_tcp_position[t] + measured_tcp_quaternion_xyzw[t] + "
    "measured_gripper[t]"
)


def _configure_pipeline() -> None:
    pipeline.DATA_ROOT = DATA_ROOT.parent
    pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    pipeline.CONFIG = CONFIG
    pipeline.STATUS_PATH = STATUS_PATH
    pipeline.AUDIT_PATH = AUDIT_PATH
    pipeline.HF_ROOT = HF_ROOT
    pipeline.EXPERIMENT_LABEL = "V7"
    pipeline.WANDB_PROJECT = WANDB_PROJECT
    pipeline.TASKS = TASKS
    pipeline.STATES = STATES
    pipeline.REPRESENTATIONS = REPRESENTATIONS


_configure_pipeline()


def _all_jobs() -> tuple[pipeline.Job, ...]:
    return tuple(job for task in TASKS for job in pipeline._jobs(task))


def _audit_dataset_contracts() -> dict[str, Any]:
    """Verify every source episode and the EE/delta-EE semantic contract."""

    result: dict[str, Any] = {
        "audited_unix": time.time(),
        "data_root": str(DATA_ROOT),
        "expected_state_dim": 8,
        "expected_action_dim": 7,
        "state_layout": EXPECTED_STATE_LAYOUT,
        "action_layout": EXPECTED_ACTION_LAYOUT,
        "tasks": {},
    }
    for task in TASKS:
        dataset = DATA_ROOT / task.dataset_name
        info = json.loads((dataset / "meta" / "info.json").read_text())
        modality = json.loads((dataset / "meta" / "modality.json").read_text())
        embodiment = json.loads((dataset / "meta" / "embodiment.json").read_text())
        features = info.get("features", {})
        if features.get("observation.state", {}).get("shape") != [8]:
            raise ValueError(f"{dataset}: metadata state shape is not [8]")
        if features.get("action", {}).get("shape") != [7]:
            raise ValueError(f"{dataset}: metadata action shape is not [7]")
        if embodiment.get("state_source") != "observation_ee":
            raise ValueError(f"{dataset}: unexpected state source")
        if embodiment.get("action_source") != "observation_delta_ee":
            raise ValueError(f"{dataset}: unexpected action source")
        if embodiment.get("state_layout") != EXPECTED_STATE_LAYOUT:
            raise ValueError(f"{dataset}: unexpected state layout")
        if embodiment.get("action_layout") != EXPECTED_ACTION_LAYOUT:
            raise ValueError(f"{dataset}: unexpected action layout")
        if set(modality.get("video", {})) != {"global", "hand"}:
            raise ValueError(f"{dataset}: expected global and hand cameras")

        files = sorted((dataset / "data" / "chunk-000").glob("episode_*.parquet"))
        expected_episodes = task.train_episodes + task.val_episodes
        if len(files) != expected_episodes:
            raise ValueError(
                f"{dataset}: expected {expected_episodes} episodes, found {len(files)}"
            )
        total_rows = 0
        lengths: list[int] = []
        timestamp_min_step = math.inf
        timestamp_max_step = -math.inf
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
            state = np.asarray(table["observation.state"], dtype=np.float32)
            action = np.asarray(table["action"], dtype=np.float32)
            timestamp = np.asarray(table["timestamp"], dtype=np.float64)
            frame = np.asarray(table["frame_index"], dtype=np.int64)
            episode = np.asarray(table["episode_index"], dtype=np.int64)
            if state.ndim != 2 or state.shape[1] != 8:
                raise ValueError(f"{path}: expected state shape [N,8], got {state.shape}")
            if action.ndim != 2 or action.shape[1] != 7:
                raise ValueError(f"{path}: expected action shape [N,7], got {action.shape}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{path}: non-finite state/action")
            if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
                raise ValueError(f"{path}: non-consecutive frame_index")
            if not np.all(episode == expected_episode):
                raise ValueError(f"{path}: episode_index mismatch")
            delta = np.diff(timestamp)
            if len(delta):
                if not np.all(delta > 0):
                    raise ValueError(f"{path}: timestamps are not strictly increasing")
                timestamp_min_step = min(timestamp_min_step, float(delta.min()))
                timestamp_max_step = max(timestamp_max_step, float(delta.max()))
            total_rows += len(state)
            lengths.append(len(state))

        videos = sorted((dataset / "videos").rglob("*.mp4"))
        if len(videos) != 2 * expected_episodes:
            raise ValueError(
                f"{dataset}: expected {2 * expected_episodes} videos, found {len(videos)}"
            )
        result["tasks"][task.name] = {
            "dataset": str(dataset),
            "episodes": len(files),
            "frames": total_rows,
            "video_files": len(videos),
            "episode_length_min": min(lengths),
            "episode_length_max": max(lengths),
            "timestamp_step_min": timestamp_min_step,
            "timestamp_step_max": timestamp_max_step,
            "state_dim": 8,
            "action_dim": 7,
        }
    pipeline._atomic_json(DATASET_AUDIT_PATH, result)
    return result


def _initialize_status(gpus: list[int]) -> None:
    stats = {task.name: pipeline._task_stats(task) for task in TASKS}
    existing = pipeline._read_status()
    value = {
        **existing,
        "state": "running",
        "launcher_pid": os.getpid(),
        "started_unix": existing.get("started_unix", time.time()),
        "current_invocation_started_unix": time.time(),
        "output_root": str(OUTPUT_ROOT),
        "parallel_schedule": "all three tasks x raw/bspline in one six-GPU wave",
        "gpus": gpus,
        "protocol": {
            "architecture": "BSP_UNet_FlowMatching",
            "action_space": "7D local delta EE (translation + rotvec) + next gripper",
            "state": {"name": "State_EE", "state_dim": 8, "layout": STATES[0].layout},
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
            "cache_root": str(OUTPUT_ROOT),
        },
        "datasets": {
            task.name: {
                "dataset_path": str(DATA_ROOT / task.dataset_name),
                "train_episodes": task.train_episodes,
                "val_episodes": task.val_episodes,
                "train_windows": stats[task.name].train_windows,
                "val_windows": stats[task.name].val_windows,
                "updates_per_epoch": stats[task.name].updates_per_epoch,
                "base_updates": pipeline.BASE_EPOCHS
                * stats[task.name].updates_per_epoch,
                "ttrtc_updates": pipeline.RTC_EPOCHS
                * stats[task.name].updates_per_epoch,
            }
            for task in TASKS
        },
        "jobs": existing.get("jobs", {}),
        "tasks": existing.get("tasks", {}),
    }
    pipeline._atomic_json(STATUS_PATH, value)


def _prepare_all() -> None:
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(pipeline._prepare_task, task, pipeline._jobs(task)): task
            for task in TASKS
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append(f"{task.name}: {exc}")
    if failures:
        raise RuntimeError("V7 preparation failures: " + "; ".join(failures))


def _run_global_stage(stage: str, gpus: list[int]) -> None:
    jobs = _all_jobs()
    if len(jobs) != 6:
        raise RuntimeError(f"expected six V7 jobs, found {len(jobs)}")
    started = time.time()
    for task in TASKS:
        pipeline._update_status(
            lambda status, name=task.name: status.setdefault("tasks", {})
            .setdefault(name, {})
            .update({"phase": stage, "phase_started_unix": started})
        )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
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
        raise RuntimeError(f"V7 {stage} failures: " + "; ".join(failures))


def _normalize_parent_provenance() -> None:
    """Make the base-to-ttRTC lineage explicit in every load sidecar.

    Deployment checkpoints already carry ``parent_checkpoint``.  The V7
    training summary additionally records the immutable parent digest so that
    a copied model tree can prove which validation-selected base initialized
    each ttRTC child without relying on a path string alone.
    """

    for job in _all_jobs():
        base = job.checkpoint("base")
        ttrtc = job.checkpoint("ttrtc")
        if not base.is_file() or not ttrtc.is_file():
            continue
        base_sha256 = pipeline._sha256(base)
        payload = pipeline.torch.load(
            ttrtc, map_location="cpu", weights_only=False, mmap=True
        )
        try:
            recorded_parent = payload.get("parent_checkpoint")
            if recorded_parent is None or Path(recorded_parent).resolve() != base.resolve():
                raise ValueError(
                    f"{ttrtc}: parent mismatch: {recorded_parent!r} != {base.resolve()}"
                )
        finally:
            del payload

        for stage, parent, parent_sha in (
            ("base", None, None),
            ("ttrtc", str(base.resolve()), base_sha256),
        ):
            summary_path = job.checkpoint(stage).with_suffix(".training.json")
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text())
            summary["parent_checkpoint"] = parent
            summary["parent_checkpoint_sha256"] = parent_sha
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

    # Publish the canonical model/server/training triples first.  The generic
    # uploader validates every model through its remote LFS SHA-256.
    result = pipeline._upload(report, result_path=UPLOAD_PATH)

    # A downloaded release must not depend on this workstation's absolute
    # prepared_path.  ``resolve_prepared_path`` already searches this portable
    # task-local layout after the configured path, so publish the six exact
    # encoder/normalization pairs alongside the models.
    api = HfApi()
    repository = api.dataset_info(pipeline.HF_REPO, files_metadata=True)
    remote_files = {item.rfilename: item for item in repository.siblings}
    sidecars: list[dict[str, Any]] = []
    operations = []
    for job in _all_jobs():
        for name in ("encoder.json", "normalization.json"):
            local = job.prepared_path / name
            if not local.is_file():
                raise FileNotFoundError(local)
            data = local.read_bytes()
            git_blob_sha1 = hashlib.sha1(
                f"blob {len(data)}\0".encode() + data
            ).hexdigest()
            remote = (
                f"{HF_ROOT}/{job.task.name}/{job.state.name}/sidecars/"
                f"{job.representation}/{name}"
            )
            entry = {
                "local": str(local),
                "remote": remote,
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_sha1": git_blob_sha1,
                "size": len(data),
                "role": "portable_load_sidecar",
            }
            sidecars.append(entry)
            sibling = remote_files.get(remote)
            if sibling is None or sibling.blob_id != git_blob_sha1:
                operations.append(
                    CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                )
    if operations:
        api.create_commit(
            repo_id=pipeline.HF_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message="Upload V7 portable ActionEE load sidecars",
        )

    repository = api.dataset_info(pipeline.HF_REPO, files_metadata=True)
    remote_files = {item.rfilename: item for item in repository.siblings}
    sidecar_missing = [
        item["remote"] for item in sidecars if item["remote"] not in remote_files
    ]
    sidecar_hash_mismatches = [
        {
            "remote": item["remote"],
            "local_git_blob_sha1": item["git_blob_sha1"],
            "remote_git_blob_sha1": remote_files[item["remote"]].blob_id,
        }
        for item in sidecars
        if item["remote"] in remote_files
        and remote_files[item["remote"]].blob_id != item["git_blob_sha1"]
    ]
    if sidecar_missing or sidecar_hash_mismatches:
        raise RuntimeError(
            "remote V7 sidecar verification failed: "
            f"missing={len(sidecar_missing)}, "
            f"hash_mismatches={len(sidecar_hash_mismatches)}"
        )
    result["files"].extend(sidecars)
    result["revision"] = repository.sha
    result["portable_sidecars"] = len(sidecars)
    result["sidecar_missing_remote"] = sidecar_missing
    result["sidecar_hash_mismatches"] = sidecar_hash_mismatches
    pipeline._atomic_json(UPLOAD_PATH, result)
    pipeline._update_status(
        lambda status: status.update({"upload": {"state": "complete", **result}})
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 6 or len(set(gpus)) != 6:
        parser.error("--gpus must contain exactly six distinct GPU indices")
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
        pipeline._update_status(
            lambda status: status.update({"dataset_audit": audit})
        )
        _prepare_all()
        if args.prepare_only:
            pipeline._update_status(lambda status: status.update({"state": "prepared"}))
            return

        _run_global_stage("base", gpus)
        _run_global_stage("ttrtc", gpus)
        completed = time.time()
        for task in TASKS:
            pipeline._update_status(
                lambda status, name=task.name: status["tasks"][name].update(
                    {"phase": "complete", "completed_unix": completed}
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
