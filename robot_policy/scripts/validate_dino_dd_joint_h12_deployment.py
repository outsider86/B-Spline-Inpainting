#!/usr/bin/env python3
"""Exercise all eight DINO joint-DD checkpoints through the WebSocket API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import av
import numpy as np

from robot_policy.config import load_config
from robot_policy.deployment.client import PolicyClient


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("raw_h1", "raw_h2", "bspline_h1", "bspline_h2")
STAGES = ("base", "rtc")


def _video_frame(path: Path, index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"video {path} has no frame {index}")


def _example(output: Path, variant: str) -> dict[str, Any]:
    config = ROOT / "configs" / "dino_dd_joint_h12" / f"{variant}.yaml"
    cfg = load_config(config, ["policy.architecture=discrete_joint"])
    prepared = Path(cfg.data.prepared_path)
    episode = int(json.loads((prepared / "splits.json").read_text())["test"][0])
    with np.load(prepared / "actions" / f"episode_{episode:06d}.npz") as actions:
        states = np.asarray(actions["state"][:2], dtype=np.float32)
    cameras: list[list[np.ndarray]] = []
    for frame in (0, 1):
        cameras.append([
            _video_frame(
                Path(cfg.data.dataset_path) / "videos" / "chunk-000" / camera
                / f"episode_{episode:06d}.mp4",
                frame,
            )
            for camera in cfg.data.camera_keys
        ])
    result: dict[str, Any] = {
        "image": cameras[1],
        "state": states[1],
        "lang": "Stack the cups.",
    }
    if cfg.data.observation_horizon == 2:
        result["image_history"] = cameras
        result["state_history"] = states
    return result


def _wait(host: str, port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server {port} exited with {process.returncode}")
        try:
            with PolicyClient(host, port, timeout=1.0):
                return
        except (OSError, TimeoutError):
            time.sleep(0.2)
    raise TimeoutError(f"server {port} did not become ready")


def _exercise(host: str, port: int, variant: str, stage: str, example: dict[str, Any]) -> dict[str, Any]:
    representation = "bspline" if variant.startswith("bspline") else "raw"
    horizon = 2 if variant.endswith("h2") else 1
    with PolicyClient(host, port, timeout=240.0) as client:
        metadata = client.metadata
        expected_metadata_method = (
            "discrete_direct_hard_mask"
            if stage == "base"
            else "training_time_direct_hard_mask"
        )
        if metadata["architecture"] != "discrete_joint":
            raise RuntimeError(f"wrong architecture: {metadata}")
        if metadata["action_representation"] != representation:
            raise RuntimeError(f"wrong representation: {metadata}")
        if metadata["observation_horizon"] != horizon:
            raise RuntimeError(f"wrong horizon: {metadata}")
        if metadata["observation_source"] != "features":
            raise RuntimeError(f"wrong observation source: {metadata}")
        if metadata["vision_raw_tokens_per_camera"] != 256:
            raise RuntimeError(f"wrong raw vision-token count: {metadata}")
        if metadata["vision_policy_tokens_per_camera"] != 32:
            raise RuntimeError(f"wrong policy vision-token count: {metadata}")
        if metadata["rtc_mask_type"] != "hard":
            raise RuntimeError(f"RTC is not hard masked: {metadata}")
        if metadata["rtc_inference_method"] != expected_metadata_method:
            raise RuntimeError(f"wrong RTC metadata method: {metadata}")

        initial = client.predict_action(
            [example], state_coordinates="physical", seed=20260915
        )
        actions = np.asarray(initial["actions"])
        if actions.shape != (1, 30, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid standard inference: {actions.shape}")
        previous = (
            actions
            if representation == "raw"
            else np.asarray(initial["normalized_control_rows"])
        )
        realtime = client.predict_action_realtime(
            [example], inference_delay=3, previous=previous,
            state_coordinates="physical", seed=20260915,
        )
        realtime_actions = np.asarray(realtime["actions"])
        inference = realtime["inference_metadata"]
        expected_result_method = (
            "discrete_hard_mask" if stage == "base" else "training_time_hard_mask"
        )
        if realtime_actions.shape != (1, 30, 7) or not np.isfinite(realtime_actions).all():
            raise RuntimeError(f"invalid realtime inference: {realtime_actions.shape}")
        if inference["rtc_inference_method"] != expected_result_method:
            raise RuntimeError(f"wrong runtime method: {inference}")
        expected_rows = 3 if representation == "raw" else 5
        if int(inference["fixed_control_rows"]) != expected_rows:
            raise RuntimeError(f"wrong hard-mask rows: {inference}")
        if representation == "bspline" and int(inference["affected_spans"]) != 2:
            raise RuntimeError(f"wrong B-spline affected spans: {inference}")
        return {
            "variant": variant,
            "stage": stage,
            "representation": representation,
            "observation_horizon": horizon,
            "checkpoint_training_type": metadata["checkpoint_training_type"],
            "standard_actions_shape": list(actions.shape),
            "standard_sampling_ms": float(initial["sampling_ms"]),
            "realtime_actions_shape": list(realtime_actions.shape),
            "realtime_sampling_ms": float(realtime["sampling_ms"]),
            "rtc_inference_method": inference["rtc_inference_method"],
            "fixed_control_rows": int(inference["fixed_control_rows"]),
            "affected_spans": inference.get("affected_spans"),
            "hard_mask_verified": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/DINO_DD_JOINT_H12"))
    parser.add_argument("--gpus", default="0,1,2,3,0,1,2,3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=12220)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root = output_root.resolve()
    output = args.output or output_root / "summary" / "deployment" / "websocket_validation.json"
    if not output.is_absolute():
        output = ROOT / output
    cases = [(variant, stage) for variant in VARIANTS for stage in STAGES]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != len(cases) or len(set(gpus)) > 4:
        parser.error("--gpus must provide eight assignments using at most four distinct GPUs")

    log_dir = output_root / "summary" / "deployment" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_cache = output_root / "cache" / "model_weights"
    processes: list[subprocess.Popen] = []
    logs = []
    endpoints = []
    try:
        for index, ((variant, stage), gpu) in enumerate(zip(cases, gpus)):
            checkpoint = output_root / "checkpoints" / variant / f"{stage}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            port = args.base_port + index
            log_path = log_dir / f"{variant}_{stage}.log"
            log = log_path.open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(ROOT / "src")
            env["HF_HOME"] = str(model_cache / "huggingface")
            env["TORCH_HOME"] = str(model_cache / "torch")
            env["XDG_CACHE_HOME"] = str(model_cache / "xdg")
            command = [
                sys.executable, "-m", "robot_policy.deployment.server",
                "--checkpoint", str(checkpoint), "--host", args.host,
                "--port", str(port), "--device", "cuda", "--precision", "bf16",
                "--idle-timeout", "600",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            processes.append(process)
            logs.append(log)
            endpoints.append((variant, stage, gpu, port, process, log_path))

        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = [
                pool.submit(_wait, args.host, port, process, args.startup_timeout)
                for _, _, _, port, process, _ in endpoints
            ]
            for future in as_completed(futures):
                future.result()
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = {
                pool.submit(_exercise, args.host, port, variant, stage, _example(output_root, variant)):
                (variant, stage, gpu, port, log_path)
                for variant, stage, gpu, port, _, log_path in endpoints
            }
            records = []
            for future in as_completed(futures):
                variant, stage, gpu, port, log_path = futures[future]
                record = future.result()
                record.update({
                    "gpu": gpu, "port": port, "server_log": str(log_path.resolve())
                })
                records.append(record)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()

    records.sort(key=lambda row: (VARIANTS.index(row["variant"]), STAGES.index(row["stage"])))
    report = {
        "scope": "real concurrent Piper-compatible WebSocket deployment audit",
        "server_count": len(records),
        "distinct_gpu_count": len(set(gpus)),
        "records": records,
        "passed": len(records) == 8 and all(row["hard_mask_verified"] for row in records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"server_count": len(records), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
