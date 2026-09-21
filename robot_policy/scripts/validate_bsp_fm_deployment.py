#!/usr/bin/env python3
"""Launch and validate all eight BSP-UNet FM WebSocket deployments."""

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

from robot_policy.deployment.client import PolicyClient


VARIANTS = ("fm_raw_h1", "fm_raw_h2", "fm_bspline_h1", "fm_bspline_h2")
STAGES = ("base", "rtc")


def _prepared_path(output: Path, variant: str) -> Path:
    representation = "bspline" if "bspline" in variant else "raw"
    return output / "cache" / "prepared" / representation


def _example(output: Path, horizon: int) -> dict[str, Any]:
    prepared = output / "cache" / "prepared" / "raw"
    splits = json.loads((prepared / "splits.json").read_text())
    episode = int(splits["test"][0])
    with np.load(prepared / "actions" / f"episode_{episode:06d}.npz") as action_data:
        states = np.asarray(action_data["state"][:2], dtype=np.float32)
    rgb = np.load(output / "cache" / "rgb84" / f"episode_{episode:06d}.npy", mmap_mode="r")
    if len(states) < 2 or len(rgb) < 2:
        raise RuntimeError(f"test episode {episode} has fewer than two frames")

    def cameras(frame: int) -> list[np.ndarray]:
        return [np.ascontiguousarray(rgb[frame, camera].transpose(1, 2, 0)) for camera in range(2)]

    example: dict[str, Any] = {
        "image": cameras(1),
        "state": states[1],
        "lang": "Stack the cups.",
    }
    if horizon == 2:
        example["image_history"] = [cameras(0), cameras(1)]
        example["state_history"] = states
    return example


def _wait_for_server(host: str, port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with code {process.returncode}")
        try:
            with PolicyClient(host, port, timeout=1.0):
                return
        except (OSError, TimeoutError):
            time.sleep(0.2)
    raise TimeoutError(f"server on port {port} did not become ready in {timeout}s")


def _exercise(host: str, port: int, variant: str, stage: str, example: dict[str, Any]) -> dict[str, Any]:
    representation = "bspline" if "bspline" in variant else "raw"
    horizon = 2 if variant.endswith("h2") else 1
    with PolicyClient(host, port, timeout=120.0) as client:
        metadata = client.metadata
        if metadata["architecture"] != "bsp_unet_fm":
            raise RuntimeError(f"wrong architecture for {variant}/{stage}: {metadata}")
        if metadata["action_representation"] != representation:
            raise RuntimeError(f"wrong representation for {variant}/{stage}: {metadata}")
        if metadata["observation_horizon"] != horizon:
            raise RuntimeError(f"wrong observation horizon for {variant}/{stage}: {metadata}")
        if metadata["observation_source"] != "rgb" or "scratch ResNet18" not in metadata["image_preprocessing"]:
            raise RuntimeError(f"deployment is not using scratch-RGB policy input: {metadata}")

        initial = client.predict_action([example], state_coordinates="physical", seed=20260915)
        actions = np.asarray(initial["actions"])
        if actions.shape != (1, 30, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid initial actions for {variant}/{stage}: {actions.shape}")
        record: dict[str, Any] = {
            "variant": variant,
            "stage": stage,
            "architecture": metadata["architecture"],
            "representation": representation,
            "observation_horizon": horizon,
            "checkpoint_training_type": metadata["checkpoint_training_type"],
            "actions_shape": list(actions.shape),
            "actions_finite": True,
            "sampling_ms": float(initial["sampling_ms"]),
            "standard_infer": True,
            "realtime_infer": False,
            "rtc_hard_mask_verified": False,
        }
        if representation == "bspline":
            controls = np.asarray(initial["normalized_control_rows"])
            if controls.shape != (1, 18, 7) or not np.isfinite(controls).all():
                raise RuntimeError(f"invalid B-spline controls for {variant}/{stage}")
            record["control_rows_shape"] = list(controls.shape)

        previous = actions if representation == "raw" else np.asarray(initial["normalized_control_rows"])
        realtime = client.predict_action_realtime(
            [example], inference_delay=3, previous=previous,
            state_coordinates="physical", seed=20260915,
        )
        realtime_actions = np.asarray(realtime["actions"])
        inference_metadata = realtime["inference_metadata"]
        if realtime_actions.shape != (1, 30, 7) or not np.isfinite(realtime_actions).all():
            raise RuntimeError(f"invalid RTC actions for {variant}/{stage}")
        if metadata["rtc_mask_type"] != "hard":
            raise RuntimeError(f"RTC checkpoint does not advertise a hard mask: {metadata}")
        expected_method = (
            "pigdm_hard_mask" if stage == "base" else "training_time_hard_mask"
        )
        if inference_metadata["rtc_inference_method"] != expected_method:
            raise RuntimeError(f"wrong RTC method for {variant}/{stage}: {inference_metadata}")
        if representation == "raw":
            if stage != "base":
                np.testing.assert_allclose(realtime_actions[:, :3], actions[:, 3:6], atol=1e-5, rtol=1e-5)
            if inference_metadata["fixed_control_rows"] != 3:
                raise RuntimeError(f"wrong raw RTC prefix mask: {inference_metadata}")
        else:
            if metadata["rtc_bspline_mask_scope"] != "exact_union_of_control_rows_supporting_affected_spans":
                raise RuntimeError(f"wrong B-spline RTC mask contract: {metadata}")
            if inference_metadata["affected_spans"] != 2 or inference_metadata["fixed_control_rows"] != 5:
                raise RuntimeError(f"wrong B-spline RTC support mask: {inference_metadata}")
            rows = np.asarray(realtime["normalized_control_rows"])
            if rows.shape != (1, 18, 7) or not np.isfinite(rows).all():
                raise RuntimeError(f"invalid RTC B-spline controls for {variant}/{stage}")
        record.update(
            {
                "realtime_infer": True,
                "realtime_actions_shape": list(realtime_actions.shape),
                "realtime_sampling_ms": float(realtime["sampling_ms"]),
                "rtc_hard_mask_verified": True,
                "rtc_inference_method": inference_metadata["rtc_inference_method"],
                "fixed_control_rows": int(inference_metadata["fixed_control_rows"]),
                "affected_spans": inference_metadata.get("affected_spans"),
            }
        )
        return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/BSP_UNET_V3"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=12120)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root = output_root.resolve()
    report_path = args.output or output_root / "summary" / "deployment" / "fm_websocket_validation.json"
    if not report_path.is_absolute():
        report_path = root / report_path
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    cases = [(variant, stage) for variant in VARIANTS for stage in STAGES]
    if len(gpus) != len(cases):
        parser.error(f"--gpus must provide exactly {len(cases)} entries")

    log_dir = output_root / "summary" / "deployment" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    logs = []
    endpoints = []
    try:
        for index, ((variant, stage), gpu) in enumerate(zip(cases, gpus)):
            checkpoint = output_root / "checkpoints" / variant / f"{stage}.pt"
            prepared = _prepared_path(output_root, variant)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if not prepared.is_dir():
                raise FileNotFoundError(prepared)
            port = args.base_port + index
            log_path = log_dir / f"{variant}_{stage}.log"
            log = log_path.open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(root / "src")
            command = [
                sys.executable, "-m", "robot_policy.deployment.server",
                "--checkpoint", str(checkpoint),
                "--prepared-path", str(prepared),
                "--host", args.host,
                "--port", str(port),
                "--device", "cuda",
                "--precision", "bf16",
                "--idle-timeout", "300",
            ]
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            logs.append(log)
            endpoints.append((variant, stage, gpu, port, process, log_path))

        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            readiness = [pool.submit(_wait_for_server, args.host, port, process, args.startup_timeout)
                         for _, _, _, port, process, _ in endpoints]
            for future in as_completed(readiness):
                future.result()

        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = {}
            for variant, stage, gpu, port, _, log_path in endpoints:
                horizon = 2 if variant.endswith("h2") else 1
                future = pool.submit(_exercise, args.host, port, variant, stage, _example(output_root, horizon))
                futures[future] = (variant, stage, gpu, port, log_path)
            records = []
            for future in as_completed(futures):
                variant, stage, gpu, port, log_path = futures[future]
                record = future.result()
                record.update({"gpu": gpu, "port": port, "server_log": str(log_path.resolve())})
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
        "scope": "real concurrent Piper-compatible WebSocket deployment audit for all BSP-UNet FM checkpoints",
        "server_count": len(records),
        "standard_infer_passed": sum(bool(row["standard_infer"]) for row in records),
        "realtime_infer_passed": sum(bool(row["realtime_infer"]) for row in records),
        "rtc_hard_mask_passed": sum(bool(row["rtc_hard_mask_verified"]) for row in records),
        "records": records,
        "passed": (
            len(records) == 8
            and all(bool(row["standard_infer"]) for row in records)
            and sum(bool(row["realtime_infer"]) for row in records) == 8
            and sum(bool(row["rtc_hard_mask_verified"]) for row in records) == 8
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "server_count", "standard_infer_passed", "realtime_infer_passed",
        "rtc_hard_mask_passed", "passed",
    )}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
