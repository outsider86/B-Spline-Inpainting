#!/usr/bin/env python3
"""Validate the two h2 BSP-UNet v4 FM checkpoints through the real server API."""

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

import numpy as np

from robot_policy.config import load_config
from robot_policy.deployment.client import PolicyClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4"
CONFIG_ROOT = ROOT / "configs" / "bsp_unet_v4"
VARIANTS = ("fm_raw_h2", "fm_bspline_h2")


def _example(variant: str) -> tuple[dict[str, Any], str, int]:
    cfg = load_config(CONFIG_ROOT / f"{variant}.yaml")
    prepared = Path(cfg.data.prepared_path)
    splits = json.loads((prepared / "splits.json").read_text())
    # V4's fixed protocol deliberately has no test episodes. Prefer test for
    # compatibility with older layouts, then validation, then training.
    selected = next(
        ((name, ids) for name in ("test", "val", "train") if (ids := splits.get(name))),
        None,
    )
    if selected is None:
        raise RuntimeError(f"no non-empty split in {prepared / 'splits.json'}")
    split, episode_ids = selected
    episode = int(episode_ids[0])
    with np.load(prepared / "actions" / f"episode_{episode:06d}.npz") as data:
        states = np.asarray(data["state"][:2], dtype=np.float32)
    rgb = np.load(
        Path(cfg.data.rgb_cache_path) / f"episode_{episode:06d}.npy",
        mmap_mode="r",
    )

    def cameras(frame: int) -> list[np.ndarray]:
        return [
            np.ascontiguousarray(rgb[frame, camera].transpose(1, 2, 0))
            for camera in range(2)
        ]

    return {
        "image": cameras(1),
        "state": states[1],
        "image_history": [cameras(0), cameras(1)],
        "state_history": states,
        "lang": "Stack the cups.",
    }, split, episode


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


def _exercise(host: str, port: int, variant: str) -> dict[str, Any]:
    representation = "bspline" if "bspline" in variant else "raw"
    with PolicyClient(host, port, timeout=240.0) as client:
        metadata = client.metadata
        required = {
            "architecture": "bsp_unet_fm",
            "action_representation": representation,
            "observation_horizon": 2,
            "observation_source": "rgb",
            "rtc_mask_type": "hard",
            "rtc_inference_method": "pigdm_binary_hard_mask",
            "rtc_output_contract": "exact_committed_prefix_plus_generated_suffix",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"wrong {key} for {variant}: {metadata}")
        if metadata.get("observation_history_fields") != ["image_history", "state_history"]:
            raise RuntimeError(f"h2 history contract is missing: {metadata}")

        example, example_split, example_episode = _example(variant)
        standard = client.predict_action([example], state_coordinates="physical", seed=20260915)
        actions = np.asarray(standard["actions"])
        if actions.shape != (1, 30, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid standard output for {variant}: {actions.shape}")
        previous = (
            actions
            if representation == "raw"
            else np.asarray(standard["normalized_control_rows"])
        )
        realtime = client.predict_action_realtime(
            [example], inference_delay=3, previous=previous,
            state_coordinates="physical", seed=20260915,
        )
        rtc_actions = np.asarray(realtime["actions"])
        info = realtime["inference_metadata"]
        if rtc_actions.shape != (1, 30, 7) or not np.isfinite(rtc_actions).all():
            raise RuntimeError(f"invalid RTC output for {variant}: {rtc_actions.shape}")
        if info.get("rtc_inference_method") != "pigdm_hard_mask":
            raise RuntimeError(f"base FM did not use PiGDM: {info}")
        expected_rows = 3 if representation == "raw" else 5
        if int(info.get("fixed_control_rows", -1)) != expected_rows:
            raise RuntimeError(f"wrong hard-mask support for {variant}: {info}")
        if representation == "bspline" and int(info.get("affected_spans", -1)) != 2:
            raise RuntimeError(f"wrong B-spline affected spans: {info}")
        expected_prefix = actions[:, 3:6]
        if not np.array_equal(rtc_actions[:, :3], expected_prefix):
            difference = float(np.max(np.abs(rtc_actions[:, :3] - expected_prefix)))
            raise RuntimeError(
                f"committed RTC prefix is not exact for {variant}; max_abs={difference}"
            )
        if (
            info.get("rtc_output_contract")
            != "exact_committed_prefix_plus_generated_suffix"
            or info.get("committed_prefix_exact") is not True
        ):
            raise RuntimeError(f"RTC output contract is missing for {variant}: {info}")
        return {
            "variant": variant,
            "representation": representation,
            "observation_horizon": 2,
            "images_per_example": 4,
            "example_split": example_split,
            "example_episode": example_episode,
            "standard_actions_shape": list(actions.shape),
            "standard_sampling_ms": float(standard["sampling_ms"]),
            "rtc_actions_shape": list(rtc_actions.shape),
            "rtc_sampling_ms": float(realtime["sampling_ms"]),
            "rtc_inference_method": info["rtc_inference_method"],
            "fixed_control_rows": int(info["fixed_control_rows"]),
            "affected_spans": info.get("affected_spans"),
            "committed_prefix_steps": int(info["committed_prefix_steps"]),
            "committed_prefix_exact": True,
            "committed_prefix_max_abs_error": 0.0,
            "passed": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=12320)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUT / "summary" / "deployment" / "websocket_validation.json",
    )
    args = parser.parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or len(set(gpus)) > 2:
        parser.error("--gpus must provide two assignments using at most two GPUs")

    log_root = OUTPUT / "summary" / "deployment" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    streams = []
    endpoints = []
    try:
        for index, (variant, gpu) in enumerate(zip(VARIANTS, gpus)):
            checkpoint = OUTPUT / "checkpoints" / variant / "base.pt"
            cfg = load_config(CONFIG_ROOT / f"{variant}.yaml")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            port = args.base_port + index
            log = log_root / f"{variant}.log"
            stream = log.open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(ROOT / "src")
            command = [
                sys.executable, "-m", "robot_policy.deployment.server",
                "--checkpoint", str(checkpoint),
                "--prepared-path", str(Path(cfg.data.prepared_path)),
                "--host", args.host, "--port", str(port),
                "--device", "cuda", "--precision", "bf16",
                "--idle-timeout", "300",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            processes.append(process)
            streams.append(stream)
            endpoints.append((variant, gpu, port, process, log))
        with ThreadPoolExecutor(max_workers=2) as pool:
            waits = [
                pool.submit(_wait, args.host, port, process, args.startup_timeout)
                for _, _, port, process, _ in endpoints
            ]
            for future in as_completed(waits):
                future.result()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_exercise, args.host, port, variant): (variant, gpu, port, log)
                for variant, gpu, port, _, log in endpoints
            }
            records = []
            for future in as_completed(futures):
                variant, gpu, port, log = futures[future]
                record = future.result()
                record.update({"gpu": gpu, "port": port, "server_log": str(log.resolve())})
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
        for stream in streams:
            stream.close()

    records.sort(key=lambda row: VARIANTS.index(row["variant"]))
    report = {
        "scope": "BSP_UNet_V4 two-checkpoint h2 Flow-Matching deployment",
        "server_count": len(records),
        "records": records,
        "passed": len(records) == 2 and all(row["passed"] for row in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"server_count": len(records), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
