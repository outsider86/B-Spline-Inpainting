from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_policy.deployment.checkpoint import inspect_checkpoint


def build_piper_statistics(checkpoint: str | Path, prepared_path: str | Path | None = None) -> dict:
    """Build the dataset_statistics.json consumed by the existing Piper clients."""
    metadata = inspect_checkpoint(checkpoint, prepared_path)
    root = metadata.prepared_path
    stats = json.loads((root / "normalization.json").read_text())
    if "state_min" in stats and "state_max" in stats:
        state_min = np.asarray(stats["state_min"], dtype=np.float32)
        state_max = np.asarray(stats["state_max"], dtype=np.float32)
    else:
        splits = json.loads((root / "splits.json").read_text())
        states = []
        for episode_id in splits["train"]:
            with np.load(root / "actions" / f"episode_{int(episode_id):06d}.npz") as episode:
                states.append(np.asarray(episode["state"], dtype=np.float32))
        concatenated = np.concatenate(states)
        state_min = concatenated.min(axis=0)
        state_max = concatenated.max(axis=0)
    return {
        "new_embodiment": {
            "state": {
                "min": state_min.tolist(),
                "max": state_max.tolist(),
                "mean": list(stats["state_mean"]),
                "std": list(stats["state_std"]),
                "client_transform": "minmax_to_minus1_plus1",
                "server_transform": "invert_client_minmax_then_apply_training_zscore",
            },
            "action": {
                "min": list(stats["action_q01"]),
                "max": list(stats["action_q99"]),
                "server_output_coordinates": "physical_absolute_robot_coordinates",
                "training_bounds": "q01_q99",
            },
        }
    }


def build_start_statistics(
    checkpoint: str | Path, prepared_path: str | Path | None = None
) -> dict:
    """Summarize the first physical state of every recorded demonstration."""
    metadata = inspect_checkpoint(checkpoint, prepared_path)
    root = metadata.prepared_path
    action_files = sorted((root / "actions").glob("episode_*.npz"))
    if not action_files:
        raise FileNotFoundError(f"no prepared episode actions under {root / 'actions'}")
    first_states = []
    for path in action_files:
        with np.load(path) as episode:
            first_states.append(np.asarray(episode["state"][0], dtype=np.float32))
    values = np.stack(first_states)
    return {
        "source": f"{len(values)} first frames from {metadata.config.data.dataset_path}",
        "state_order": [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"
        ],
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export existing-Piper-client statistics for a robot_policy checkpoint."
    )
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--prepared-path")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--start-output",
        help="Optional stacking-cups demonstration-start min/max/median JSON.",
    )
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_piper_statistics(args.checkpoint, args.prepared_path), indent=2)
        + "\n"
    )
    print(output)
    if args.start_output:
        start_output = Path(args.start_output).expanduser().resolve()
        start_output.parent.mkdir(parents=True, exist_ok=True)
        start_output.write_text(
            json.dumps(build_start_statistics(args.checkpoint, args.prepared_path), indent=2)
            + "\n"
        )
        print(start_output)


if __name__ == "__main__":
    main()
