#!/usr/bin/env python3
"""Requirement-level audit for the 36-checkpoint DiT capacity sweep."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from robot_policy.config import load_config
from robot_policy.policies import create_policy
from robot_policy.policies.common import parameter_groups


SIZES = {
    "dit_s": ("DiT-S", 384, 6, 4),
    "dit_b": ("DiT-B", 768, 12, 12),
    "dit_l": ("DiT-L", 1024, 24, 16),
}
ARCHITECTURES = ("fm", "discrete_layerwise", "discrete_joint")
REPRESENTATIONS = ("raw", "bspline")
STAGES = ("base", "ttrtc")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_metric(run: Any, key: str) -> bool:
    for row in run.scan_history(keys=[key], page_size=1):
        if row.get(key) is not None:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-wandb", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sweep_root = project_root / "outputs" / "model_size_sweep_50k_bs32"
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    api = None
    if args.check_wandb:
        import wandb
        api = wandb.Api()

    final_files = sorted(sweep_root.glob("dit_*/*/checkpoints/*.pt"))
    expected_paths: set[Path] = set()
    for slug, shape in SIZES.items():
        model_size, width, depth, heads = shape
        for representation in REPRESENTATIONS:
            config_path = project_root / "configs" / "model_size_sweep" / f"{slug}_{representation}.yaml"
            for architecture in ARCHITECTURES:
                cfg = load_config(config_path, [f"policy.architecture={architecture}"])
                with torch.device("meta"):
                    expected_counts = parameter_groups(create_policy(cfg))
                checkpoint_dir = sweep_root / slug / representation / "checkpoints"
                manifest_path = checkpoint_dir / "checkpoint_manifest.json"
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"checkpoints": []}
                expected_training_type = {"base": "base", "ttrtc": "ttrtc" if representation == "raw" else "rtc"}
                for stage in STAGES:
                    path = checkpoint_dir / f"{architecture}_{stage}.pt"
                    expected_paths.add(path)
                    record: dict[str, Any] = {
                        "model_size": model_size,
                        "representation": representation,
                        "architecture": architecture,
                        "stage": stage,
                        "path": str(path.resolve()),
                        "checks": {},
                    }
                    if not path.exists():
                        errors.append(f"missing {path}")
                        records.append(record)
                        continue
                    digest = file_sha256(path)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    expected_update = cfg.train.updates if stage == "base" else cfg.train.rtc_updates
                    checks = record["checks"]
                    checks["update"] = payload.get("update") == expected_update
                    checks["shape"] = (
                        payload.get("config", {}).get("policy", {}).get("model_size") == model_size
                        and payload.get("config", {}).get("policy", {}).get("hidden_dim") == width
                        and payload.get("config", {}).get("policy", {}).get("depth") == depth
                        and payload.get("config", {}).get("policy", {}).get("heads") == heads
                    )
                    checks["batch_32"] = (
                        payload.get("config", {}).get("train", {}).get("batch_size") == 32
                        and payload.get("config", {}).get("train", {}).get("effective_batch_size") == 32
                    )
                    checks["parameters"] = payload.get("parameter_counts") == expected_counts
                    checks["action_mse_local"] = (
                        bool(payload.get("loss_trace"))
                        and all("action_mse" in item for item in payload["loss_trace"])
                        and bool(payload.get("history"))
                        and all("action_mse" in item.get("validation", {}) for item in payload["history"])
                    )
                    model = create_policy(cfg)
                    incompatible = model.load_state_dict(payload["model"])
                    checks["independent_reload"] = not incompatible.missing_keys and not incompatible.unexpected_keys
                    del model
                    entries = [item for item in manifest.get("checkpoints", []) if
                               item.get("architecture") == architecture and
                               item.get("training_type") == expected_training_type[stage]]
                    checks["one_manifest_entry"] = len(entries) == 1
                    if entries:
                        entry = entries[0]
                        checks["manifest_hash"] = entry.get("sha256") == digest
                        checks["manifest_update"] = entry.get("training_update_count") == expected_update
                        checks["manifest_parameters"] = entry.get("parameter_counts") == expected_counts
                    if stage == "base":
                        checks["no_parent"] = payload.get("parent_checkpoint") is None
                    else:
                        parent = checkpoint_dir / f"{architecture}_base.pt"
                        checks["parent_path"] = Path(payload.get("parent_checkpoint", "")).resolve() == parent.resolve()
                        manifest_parent = entries[0].get("parent_checkpoint") if entries else None
                        checks["manifest_parent_path"] = bool(manifest_parent) and Path(manifest_parent).resolve() == parent.resolve()
                        if parent.exists():
                            checks["parent_hash_stable"] = file_sha256(parent) == next(
                                (item.get("sha256") for item in manifest.get("checkpoints", [])
                                 if item.get("architecture") == architecture and item.get("training_type") == "base"), None
                            )
                    wandb_info = payload.get("wandb") or {}
                    expected_project = cfg.wandb.base_project if stage == "base" else cfg.wandb.rtc_project
                    checks["wandb_metadata"] = (
                        wandb_info.get("project") == expected_project and bool(wandb_info.get("run_id"))
                        and model_size.lower().replace("-", "-") in wandb_info.get("run_name", "").lower()
                    )
                    if api is not None and wandb_info.get("run_id"):
                        run = api.run(f"{cfg.wandb.entity}/{expected_project}/{wandb_info['run_id']}")
                        checks["wandb_finished"] = run.state == "finished"
                        checks["wandb_update_summary"] = run.summary.get("training_updates") == expected_update
                        checks["wandb_action_mse"] = has_metric(run, "train/action_mse") and has_metric(run, "validation/action_mse")
                        checks["wandb_model_artifact"] = any(artifact.type == "model" for artifact in run.logged_artifacts())
                    failed = [name for name, passed in checks.items() if not passed]
                    if failed:
                        errors.append(f"{path}: failed {failed}")
                    record.update({
                        "sha256": digest,
                        "update": payload.get("update"),
                        "parameters": payload.get("parameter_counts", {}).get("total_trainable"),
                        "final_train_action_mse": payload.get("loss_trace", [{}])[-1].get("action_mse"),
                        "final_validation_action_mse": payload.get("history", [{}])[-1].get("validation", {}).get("action_mse"),
                        "wandb": wandb_info,
                    })
                    records.append(record)

    extra = set(final_files) - expected_paths
    missing = expected_paths - set(final_files)
    if len(final_files) != 36:
        errors.append(f"expected exactly 36 final .pt files, found {len(final_files)}")
    if extra:
        errors.append(f"unexpected final files: {sorted(map(str, extra))}")
    if missing:
        errors.append(f"missing final files: {sorted(map(str, missing))}")
    run_ids = [item.get("wandb", {}).get("run_id") for item in records if item.get("wandb", {}).get("run_id")]
    project_counts = Counter(
        item.get("wandb", {}).get("project") for item in records if item.get("wandb", {}).get("project")
    )
    expected_project_counts = {
        "robot-policy-50k-bs32-basic": 18,
        "robot-policy-5k-bs32-ttRTC": 18,
    }
    if len(run_ids) != 36 or len(set(run_ids)) != 36:
        errors.append(f"expected 36 unique W&B run IDs, found {len(run_ids)} IDs / {len(set(run_ids))} unique")
    if dict(project_counts) != expected_project_counts:
        errors.append(f"W&B project counts mismatch: {dict(project_counts)} != {expected_project_counts}")
    report = {
        "passed": not errors,
        "expected_checkpoints": 36,
        "found_checkpoints": len(final_files),
        "records": records,
        "unique_wandb_run_ids": len(set(run_ids)),
        "wandb_project_counts": dict(project_counts),
        "errors": errors,
    }
    output = sweep_root / "completion_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "found_checkpoints": len(final_files), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
