#!/usr/bin/env python3
"""Strict completion audit for the 18 V4Full/V5/V5.1 ttRTC models."""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fm_versions_ttrtc import _job_matrix
from robot_policy.deployment.checkpoint import inspect_checkpoint, load_deployment_policy
from robot_policy.deployment.server_config import load_server_config, validate_server_config


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "outputs" / "TTRTC_5E_AUDIT.json")
    )
    args = parser.parse_args()
    status = json.loads((ROOT / "outputs" / "TTRTC_5E_STATUS.json").read_text())
    if status.get("state") != "complete" or status.get("failures"):
        raise RuntimeError("training status is not cleanly complete")

    results: list[dict[str, object]] = []
    for job in _job_matrix():
        path = job.output
        server_path = path.with_suffix(".server.json")
        wandb_path = path.with_suffix(path.suffix + ".wandb.json")
        for required in (path, server_path, wandb_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        selected_update = int(payload.get("selected_update", -1))
        metric = float(payload.get("validation_action_mse", math.nan))
        expected_type = "ttrtc" if job.representation == "raw" else "rtc"
        checks = {
            "checkpoint_kind": payload.get("checkpoint_kind") == "best_validation_model",
            "checkpoint_epoch": int(payload.get("checkpoint_epoch", -1)) == 5,
            "container_update": int(payload.get("update", -1)) == job.total_updates,
            "selected_at_validation": selected_update
            in {
                epoch * job.updates_per_epoch for epoch in range(1, 6)
            },
            "finite_validation_metric": math.isfinite(metric),
            "training_type": payload.get("training_type") == expected_type,
            "parent": Path(payload.get("parent_checkpoint", "")).resolve()
            == job.parent.resolve(),
            "architecture": payload.get("architecture") == "bsp_unet_fm",
            "rtc_updates": int(payload["config"]["train"]["rtc_updates"])
            == job.total_updates,
            "eval_every": int(payload["config"]["train"]["eval_every"])
            == job.updates_per_epoch,
            "best_boundary": int(
                payload["config"]["train"]["best_checkpoint_every_epochs"]
            )
            == 5,
        }
        if not all(checks.values()):
            raise ValueError(f"{job.name} metadata checks failed: {checks}")
        del payload

        digest = sha256(path.read_bytes()).hexdigest()
        parent_digest = sha256(job.parent.read_bytes()).hexdigest()
        expected_parent_digest = status["jobs"][job.name]["parent_sha256"]
        if parent_digest != expected_parent_digest:
            raise ValueError(f"{job.name} parent changed during training")

        manifest_path = path.parent / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entries = [
            entry
            for entry in manifest.get("checkpoints", [])
            if Path(entry.get("file_path", "")).resolve() == path.resolve()
        ]
        if len(entries) != 1 or entries[0].get("sha256") != digest:
            raise ValueError(f"{job.name} manifest hash mismatch")

        metadata = inspect_checkpoint(path)
        server = load_server_config(server_path)
        validate_server_config(
            server,
            {
                "architecture": metadata.architecture,
                "action_representation": metadata.config.data.action_representation,
                "observation_horizon": metadata.config.data.observation_horizon,
                "state_shape": (metadata.config.data.state_dim,),
                "camera_keys": list(metadata.config.data.camera_keys),
            },
        )
        model = load_deployment_policy(metadata, torch.device("cpu"))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        del model
        gc.collect()

        log_lines = (job.leaf / "ttrtc.log").read_text(errors="replace").splitlines()
        validations = sum(line.startswith("validation ") for line in log_lines)
        skipped = 0
        command_lines = [line for line in log_lines if line.startswith("$ ")]
        for line in log_lines:
            if not line.startswith("{"):
                continue
            try:
                skipped += int(json.loads(line).get("optimizer_step_skipped", 0))
            except json.JSONDecodeError:
                pass
        if validations != 5:
            raise ValueError(f"{job.name} has {validations} validations, expected 5")
        if skipped:
            raise ValueError(f"{job.name} skipped {skipped} optimizer updates")
        if len(command_lines) != 1 or "--resume" in command_lines[0]:
            raise ValueError(f"{job.name} did not use one fresh-optimizer command")

        results.append(
            {
                "job": job.name,
                "checkpoint": str(path),
                "sha256": digest,
                "parent_sha256": parent_digest,
                "epochs": 5,
                "updates": job.total_updates,
                "selected_update": selected_update,
                "selected_epoch": selected_update // job.updates_per_epoch,
                "validation_action_mse": metric,
                "full_validation_runs": validations,
                "optimizer_steps_skipped": skipped,
                "fresh_optimizer": True,
                "parameter_count": parameter_count,
                "server_config": str(server_path),
                "wandb": json.loads(wandb_path.read_text()),
            }
        )

    artifacts = []
    for version in ("V4", "V5", "V51"):
        root = ROOT / "outputs" / version
        for pattern in ("*.resume", "*.best.weights.pt", "ttrtc.best_epoch_*.pt"):
            artifacts.extend(str(path) for path in root.glob(f"**/{pattern}"))
    if artifacts:
        raise ValueError(f"training artifacts remain: {artifacts}")

    report = {
        "state": "verified_complete",
        "scope": "V4Full, V5, V5.1 x three tasks x raw/B-spline",
        "checkpoint_count": len(results),
        "epochs_each": 5,
        "fresh_optimizer": True,
        "all_full_validation_runs": all(
            result["full_validation_runs"] == 5 for result in results
        ),
        "all_strict_cpu_loads": True,
        "training_artifacts": artifacts,
        "results": results,
    }
    if report["checkpoint_count"] != 18:
        raise ValueError("expected 18 audited ttRTC checkpoints")
    _atomic_json(Path(args.output).resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
