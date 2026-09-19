#!/usr/bin/env python3
"""Keep six GPUs occupied while producing all 8 base + 8 RTC v3 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V3"
CONFIG = ROOT / "configs" / "bsp_unet_v3"
ARCHITECTURE = {
    "fm": "bsp_unet_fm",
    "dd": "bsp_unet_discrete",
}
VARIANTS = (
    "fm_raw_h1", "fm_raw_h2", "fm_bspline_h1", "fm_bspline_h2",
    "dd_raw_h1", "dd_raw_h2", "dd_bspline_h1", "dd_bspline_h2",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n$ {' '.join(command)}\n")
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def _worker(gpu: int, variants: list[str], status: dict, lock: threading.Lock) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src")
    python = sys.executable
    try:
        for variant in variants:
            family = variant.split("_", 1)[0]
            architecture = ARCHITECTURE[family]
            config = CONFIG / f"{variant}.yaml"
            run_root = OUTPUT / "checkpoints" / variant
            base = run_root / "base.pt"
            rtc = run_root / "rtc.pt"
            log = OUTPUT / "logs" / f"gpu{gpu}_{variant}.log"
            run_root.mkdir(parents=True, exist_ok=True)
            if base.is_file() and rtc.is_file():
                with lock:
                    status[variant] = {
                        "gpu": gpu, "stage": "rtc", "state": "already_complete"
                    }
                    _atomic_json(OUTPUT / "status.json", status)
                continue

            stages = [
                (
                    "base",
                    [
                        python, "-m", "robot_policy.cli", "train_base",
                        "--config", str(config), "--architecture", architecture,
                        "--output", str(base),
                    ],
                    base,
                ),
                (
                    "parent_cache",
                    [
                        python, "-m", "robot_policy.cli", "cache_parent_predictions",
                        "--config", str(config), "--architecture", architecture,
                        "--checkpoint", str(base), "--batch-size", "256",
                    ],
                    None,
                ),
                (
                    "rtc",
                    [
                        python, "-m", "robot_policy.cli", "finetune_rtc",
                        "--config", str(config), "--architecture", architecture,
                        "--parent", str(base), "--output", str(rtc),
                    ],
                    rtc,
                ),
            ]
            for stage, command, expected in stages:
                if expected is not None and expected.is_file():
                    with lock:
                        status[variant] = {"gpu": gpu, "stage": stage, "state": "already_complete"}
                        _atomic_json(OUTPUT / "status.json", status)
                    continue
                with lock:
                    status[variant] = {
                        "gpu": gpu,
                        "stage": stage,
                        "state": "running",
                        "started_unix": time.time(),
                    }
                    _atomic_json(OUTPUT / "status.json", status)
                _run(command, log, env)
                with lock:
                    status[variant] = {
                        "gpu": gpu,
                        "stage": stage,
                        "state": "complete",
                        "completed_unix": time.time(),
                    }
                    _atomic_json(OUTPUT / "status.json", status)
        with lock:
            status[f"gpu_{gpu}"] = {"state": "queue_complete", "completed_unix": time.time()}
            _atomic_json(OUTPUT / "status.json", status)
    except Exception as exc:
        with lock:
            status[f"gpu_{gpu}"] = {"state": "failed", "error": str(exc), "failed_unix": time.time()}
            _atomic_json(OUTPUT / "status.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="comma-separated subset of v3 variants",
    )
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown v3 variants: {unknown}")
    if not gpus or not variants:
        raise ValueError("at least one GPU and one variant are required")
    queues = {gpu: [] for gpu in gpus}
    for index, variant in enumerate(variants):
        queues[gpus[index % len(gpus)]].append(variant)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status: dict = {"launcher_pid": os.getpid(), "started_unix": time.time()}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker, args=(gpu, queues[gpu], status, lock), daemon=False)
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [key for key, value in status.items() if isinstance(value, dict) and value.get("state") == "failed"]
    if failures:
        raise SystemExit(f"failed GPU queues: {failures}")
    _atomic_json(OUTPUT / "COMPLETE.json", {"completed_unix": time.time(), "variants": sorted(variants)})


if __name__ == "__main__":
    main()
