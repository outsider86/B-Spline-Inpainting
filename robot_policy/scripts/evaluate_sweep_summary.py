#!/usr/bin/env python3
"""Run uniform open-loop or latency evaluation over all 36 sweep checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread


SIZES = ("dit_l", "dit_b", "dit_s")
REPRESENTATIONS = ("raw", "bspline")
ARCHITECTURES = ("fm", "discrete_layerwise", "discrete_joint")
STAGES = ("base", "ttrtc")


def tasks(root: Path, kind: str) -> list[dict[str, str | Path]]:
    result = []
    for size in SIZES:
        for representation in REPRESENTATIONS:
            config = root / "configs" / "model_size_sweep" / f"{size}_{representation}.yaml"
            prepared = root / "outputs" / "SWEEP" / "summary" / "cache" / representation
            for architecture in ARCHITECTURES:
                for stage in STAGES:
                    name = f"{architecture}_{stage}"
                    result.append({
                        "representation": representation,
                        "architecture": architecture,
                        "config": config,
                        "prepared": prepared,
                        "checkpoint": root / "outputs" / "SWEEP" / size / representation / "checkpoints" / f"{name}.pt",
                        "output": root / "outputs" / "SWEEP" / "summary" / kind / size / representation / f"{name}.json",
                        "name": f"{size}/{representation}/{name}",
                    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("open_loop", "latency", "rtc"), required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = root / "outputs" / "SWEEP" / "summary"
    log_dir = summary / "logs" / args.kind
    log_dir.mkdir(parents=True, exist_ok=True)
    pending: Queue[dict[str, str | Path]] = Queue()
    selected = []
    architectures = {item.strip() for item in args.architectures.split(",") if item.strip()}
    unknown = architectures.difference(ARCHITECTURES)
    if unknown:
        parser.error(f"unknown architectures: {sorted(unknown)}")
    for task in tasks(root, args.kind):
        if task["architecture"] not in architectures:
            continue
        if args.force or not Path(task["output"]).exists():
            pending.put(task)
            selected.append(task)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    failures: list[tuple[str, int]] = []
    print_lock = Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                task = pending.get_nowait()
            except Empty:
                return
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = log_dir / (str(task["name"]).replace("/", "__") + ".log")
            command = [
                sys.executable, "-m", "robot_policy.cli",
                {"open_loop":"evaluate_open_loop","latency":"benchmark_inference","rtc":"evaluate_rtc"}[args.kind],
                "--config", str(task["config"]),
                "--set", f"data.prepared_path={task['prepared']}",
                "--set", f"data.vision_cache_path={summary / 'cache' / 'vision'}",
                "--architecture", str(task["architecture"]),
                "--checkpoint", str(task["checkpoint"]),
                "--output", str(output),
            ]
            if args.kind in {"open_loop","rtc"}:
                command += ["--batch-size", str(args.batch_size)]
                if args.max_samples is not None:
                    command += ["--max-samples", str(args.max_samples)]
            else:
                command += ["--warmup", str(args.warmup), "--iterations", str(args.iterations)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(root / "src")
            with print_lock:
                print(f"[gpu {gpu}] start {task['name']}", flush=True)
            with log.open("w") as handle:
                completed = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
            with print_lock:
                print(f"[gpu {gpu}] {'done' if completed.returncode == 0 else 'FAILED'} {task['name']}", flush=True)
            if completed.returncode:
                failures.append((str(task["name"]), completed.returncode))
            pending.task_done()

    threads = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        for name, code in failures:
            print(f"failure: {name} exit={code}", file=sys.stderr)
        raise SystemExit(1)
    print(f"completed {len(selected)} {args.kind} tasks", flush=True)


if __name__ == "__main__":
    main()
