#!/usr/bin/env python3
"""Run uniform open-loop or latency evaluation for active sweep policies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread


SIZES = ("dit_b", "dit_s")
REPRESENTATIONS = ("raw", "bspline")
ARCHITECTURES = ("fm", "discrete_joint")
STAGES = ("base", "ttrtc")


def tasks(root: Path, kind: str, *, sweep_root: Path | None = None,
          config_dir: Path | None = None, summary: Path | None = None,
          sizes: tuple[str, ...] = SIZES,
          representations: tuple[str, ...] = REPRESENTATIONS,
          stages: tuple[str, ...] = STAGES) -> list[dict[str, str | Path]]:
    sweep_root = sweep_root or root / "outputs" / "SWEEP"
    config_dir = config_dir or root / "configs" / "model_size_sweep"
    summary = summary or sweep_root / "summary"
    result = []
    for size in sizes:
        for representation in representations:
            config = config_dir / f"{size}_{representation}.yaml"
            prepared = sweep_root / "cache" / representation
            for architecture in ARCHITECTURES:
                for stage in stages:
                    name = f"{architecture}_{stage}"
                    result.append({
                        "representation": representation,
                        "architecture": architecture,
                        "config": config,
                        "prepared": prepared,
                        "checkpoint": sweep_root / size / representation / "checkpoints" / f"{name}.pt",
                        "output": summary / kind / size / representation / f"{name}.json",
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
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sweep_root = args.sweep_root.resolve() if args.sweep_root else root / "outputs" / "SWEEP"
    config_dir = args.config_dir.resolve() if args.config_dir else root / "configs" / "model_size_sweep"
    summary = args.summary_root.resolve() if args.summary_root else sweep_root / "summary"
    log_dir = summary / "logs" / args.kind
    log_dir.mkdir(parents=True, exist_ok=True)
    pending: Queue[dict[str, str | Path]] = Queue()
    selected = []
    architectures = {item.strip() for item in args.architectures.split(",") if item.strip()}
    unknown = architectures.difference(ARCHITECTURES)
    if unknown:
        parser.error(f"unknown architectures: {sorted(unknown)}")
    sizes = tuple(item.strip() for item in args.sizes.split(",") if item.strip())
    unknown_sizes = set(sizes).difference(SIZES)
    if unknown_sizes or not sizes:
        parser.error(f"unknown sizes: {sorted(unknown_sizes)}")
    representations = tuple(
        item.strip() for item in args.representations.split(",") if item.strip()
    )
    unknown_representations = set(representations).difference(REPRESENTATIONS)
    if unknown_representations or not representations:
        parser.error(f"unknown representations: {sorted(unknown_representations)}")
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown_stages = set(stages).difference(STAGES)
    if unknown_stages or not stages:
        parser.error(f"unknown stages: {sorted(unknown_stages)}")
    for task in tasks(
        root,
        args.kind,
        sweep_root=sweep_root,
        config_dir=config_dir,
        summary=summary,
        sizes=sizes,
        representations=representations,
        stages=stages,
    ):
        if task["architecture"] not in architectures:
            continue
        # Incremental sweep evaluation is intentionally restart-safe: an
        # unfinished combination is not a failed evaluation task.  Its atomic
        # final checkpoint will be discovered by a later invocation.
        if not Path(task["checkpoint"]).is_file():
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
                "--set", f"data.vision_cache_path={sweep_root / 'cache' / 'vision'}",
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
