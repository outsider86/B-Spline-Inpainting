#!/usr/bin/env python3
"""Stop the two hanging-mug runs at epoch 100 and finalize validation-best bases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "BSP_UNET_V4_HANGING_MUG_600E"
GLOBAL_STATUS = ROOT / "outputs" / "BSP_UNET_V4_TWO_TASKS_600E_STATUS.json"
STATUS = OUTPUT / "EPOCH100_AUTOSTOP.json"
TARGET_UPDATE = 41_700
JOBS = {
    "fm_raw_h2": {
        "gpu": 4,
        "config": ROOT / "configs" / "bsp_unet_v4_hanging_mug_600e" / "fm_raw_h2.yaml",
    },
    "fm_bspline_h2": {
        "gpu": 5,
        "config": ROOT / "configs" / "bsp_unet_v4_hanging_mug_600e" / "fm_bspline_h2.yaml",
    },
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _checkpoint_paths(variant: str) -> dict[str, Path]:
    base = OUTPUT / "checkpoints" / variant / "base.pt"
    return {
        "base": base,
        "best": base.with_suffix(base.suffix + ".best.weights.pt"),
        "resume": base.with_suffix(base.suffix + ".resume"),
        "wandb": base.with_suffix(base.suffix + ".wandb.json"),
        "step": base.with_name(f"{base.stem}.step_{TARGET_UPDATE:06d}{base.suffix}"),
        "epoch": base.with_name(f"{base.stem}.epoch_100{base.suffix}"),
        "log": OUTPUT / "logs" / f"{variant}.log",
    }


def _load_update(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["update"])


def _ready(variant: str) -> bool:
    paths = _checkpoint_paths(variant)
    if not all(paths[key].is_file() for key in ("best", "resume", "step", "wandb")):
        return False
    return _load_update(paths["resume"]) == TARGET_UPDATE and _load_update(paths["step"]) == TARGET_UPDATE


def _process_table() -> dict[int, tuple[int, str]]:
    table: dict[int, tuple[int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            ppid = int((entry / "stat").read_text().split()[3])
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            table[pid] = (ppid, command)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return table


def _main_pid(variant: str) -> int:
    launcher_pid = int(json.loads(GLOBAL_STATUS.read_text())["launcher_pid"])
    config = str(JOBS[variant]["config"])
    matches = [
        pid for pid, (ppid, command) in _process_table().items()
        if ppid == launcher_pid and config in command and " train_base " in f" {command} "
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one live main process for {variant}, found {matches}")
    return matches[0]


def _descendants(root_pid: int, table: dict[int, tuple[int, str]]) -> list[int]:
    result: list[int] = []
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, (ppid, _) in table.items() if ppid == parent]
        result.extend(children)
        frontier.extend(children)
    return result


def _terminate_training(variant: str) -> None:
    main = _main_pid(variant)
    table = _process_table()
    targets = [main, *_descendants(main, table)]
    # Stop the owner first so it cannot request another batch, then its workers.
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline:
        alive = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
        if not alive:
            return
        time.sleep(0.1)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _publish_epoch_alias(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


def _finalize(variant: str) -> dict[str, object]:
    spec = JOBS[variant]
    paths = _checkpoint_paths(variant)
    _terminate_training(variant)
    _publish_epoch_alias(paths["step"], paths["epoch"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    environment["PYTHONPATH"] = str(ROOT / "src")
    cache = OUTPUT / "cache" / "model_weights"
    environment["HF_HOME"] = str(cache / "huggingface")
    environment["TORCH_HOME"] = str(cache / "torch")
    environment["XDG_CACHE_HOME"] = str(cache / "xdg")
    # W&B otherwise defaults artifact staging to ~/.local/share/wandb. Keep
    # all large temporary data inside the experiment's scratch output tree.
    environment["WANDB_DATA_DIR"] = str(cache / "wandb_data")
    environment["WANDB_CACHE_DIR"] = str(cache / "wandb_cache")
    command = [
        sys.executable,
        "-m",
        "robot_policy.cli",
        "train_base",
        "--config",
        str(spec["config"]),
        "--architecture",
        "bsp_unet_fm",
        "--output",
        str(paths["base"]),
        "--resume",
        str(paths["resume"]),
        "--wandb-resume",
        str(paths["wandb"]),
        "--updates",
        str(TARGET_UPDATE),
    ]
    with paths["log"].open("a", buffering=1) as stream:
        stream.write(f"\n# Epoch-100 autostop finalization\n$ {' '.join(command)}\n")
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"finalization failed for {variant}; see {paths['log']}")
    final = torch.load(paths["base"], map_location="cpu", weights_only=False)
    best = torch.load(paths["best"], map_location="cpu", weights_only=False) if paths["best"].exists() else None
    if int(final["update"]) != TARGET_UPDATE:
        raise RuntimeError(f"wrong final update for {variant}: {final['update']}")
    selected_update = int(final["selected_update"])
    if not 0 < selected_update <= TARGET_UPDATE:
        raise RuntimeError(f"invalid selected update for {variant}: {selected_update}")
    return {
        "state": "stopped_epoch100_finalized",
        "gpu": spec["gpu"],
        "training_update": int(final["update"]),
        "selected_update": selected_update,
        "best_validation_action_mse": float(final["best_validation"]),
        "epoch_checkpoint": str(paths["epoch"]),
        "base_checkpoint": str(paths["base"]),
        "best_sidecar_consumed": best is None,
        "completed_unix": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    state: dict[str, object] = {
        "watcher_pid": os.getpid(),
        "target_epoch": 100,
        "target_update": TARGET_UPDATE,
        "started_unix": time.time(),
        "jobs": {variant: {"state": "waiting_for_epoch100"} for variant in JOBS},
    }
    _atomic_json(STATUS, state)
    pending = set(JOBS)
    while pending:
        for variant in list(pending):
            if not _ready(variant):
                continue
            state["jobs"][variant] = {"state": "finalizing", "started_unix": time.time()}
            _atomic_json(STATUS, state)
            try:
                finalized = _finalize(variant)
                state["jobs"][variant] = finalized
                # The shared launcher records SIGTERM as failure. Replace that
                # expected state with the intentional early-stop result.
                global_status = json.loads(GLOBAL_STATUS.read_text())
                global_status["jobs"][f"hanging_mug/{variant}"] = finalized
                _atomic_json(GLOBAL_STATUS, global_status)
                pending.remove(variant)
            except BaseException as exc:
                state["jobs"][variant] = {
                    "state": "failed", "error": str(exc), "failed_unix": time.time()
                }
                _atomic_json(STATUS, state)
                raise
            _atomic_json(STATUS, state)
        if pending:
            time.sleep(args.poll_seconds)
    state["state"] = "complete"
    state["completed_unix"] = time.time()
    _atomic_json(STATUS, state)


if __name__ == "__main__":
    main()
