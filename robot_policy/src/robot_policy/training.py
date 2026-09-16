from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from robot_policy.config import Config, config_dict
from robot_policy.data.dataset import PreparedPolicyDataset, collate_policy_batch
from robot_policy.policies import create_policy, load_policy_checkpoint
from robot_policy.policies.common import parameter_groups
from robot_policy.rtc.delay_mapping import sample_raw_delays, sample_spline_delays
from robot_policy.rtc.training import create_action_codec, make_rtc_condition
from robot_policy.tracking import WandbTracker


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _previous_batch(batch: dict[str, torch.Tensor], delays: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    row = torch.arange(len(delays), device=delays.device); idx = delays - 1
    previous = {"vision_features": batch["previous_vision_features"][row, idx], "state": batch["previous_state"][row, idx]}
    return previous, batch["has_previous"][row, idx]


def _cached_parent_prediction(batch: dict[str, torch.Tensor], delays: torch.Tensor) -> torch.Tensor | None:
    if "previous_parent_prediction" not in batch:
        return None
    row = torch.arange(len(delays), device=delays.device)
    return batch["previous_parent_prediction"][row, delays - 1]


def _matching_parent_cache(cfg: Config, parent_checkpoint: str | None) -> Path | None:
    """Return only a cache produced by the exact parent checkpoint.

    Capacity sweeps can share prepared actions, but never parent predictions.
    Hash-keyed directories prevent a cache from a different model size from
    silently changing the ttRTC parent policy.
    """
    if parent_checkpoint is None:
        return None
    checkpoint = Path(parent_checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    root = Path(cfg.data.prepared_path) / "parent_predictions" / cfg.policy.architecture
    candidates = (root / checkpoint_hash, root)
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("parent_checkpoint_sha256") == checkpoint_hash
            and manifest.get("architecture") == cfg.policy.architecture
            and manifest.get("action_representation") == cfg.data.action_representation
        ):
            return candidate
    return None


def _sample_training_delays(cfg: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    if cfg.data.action_representation == "raw":
        return sample_raw_delays(batch_size, device, cfg.rtc.raw_delay_min, cfg.rtc.raw_delay_max)
    return sample_spline_delays(batch_size, device, cfg.rtc.spline_delay_min, cfg.rtc.spline_delay_max)


def _training_type(cfg: Config, rtc: bool) -> str:
    if not rtc:
        return "base"
    return "ttrtc" if cfg.data.action_representation == "raw" else "rtc"


def _lr_multiplier(step: int, updates: int, warmup: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(updates - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


@torch.no_grad()
def validate(model, loader, device, cfg, rtc_parent=None, codec=None, max_batches: int = 16) -> dict[str, float]:
    model.eval(); totals: dict[str, float] = {}; count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches: break
        batch = _move(batch, device); condition = None
        if rtc_parent is not None:
            delays = _sample_training_delays(cfg, len(batch["state"]), device)
            previous, has_previous = _previous_batch(batch, delays)
            condition = make_rtc_condition(rtc_parent, previous, cfg.policy.architecture, codec, delays, has_previous,
                                           _cached_parent_prediction(batch, delays))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.precision == "bf16"):
            result = model.loss(batch, condition)
        for k, v in result.items(): totals[k] = totals.get(k, 0.0) + float(v)
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_manifest_entry(path: Path, payload: dict[str, Any], cfg: Config, selection: dict[str, float], parent: str | None) -> dict[str, Any]:
    file_hash = sha256(path.read_bytes()).hexdigest()
    prepared = Path(cfg.data.prepared_path).resolve()
    splits = json.loads((prepared / "splits.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    vision_root = Path(cfg.data.vision_cache_path).resolve() if cfg.data.vision_cache_path else prepared / "vision"
    vision_manifest = json.loads(next(vision_root.glob("worker_*_manifest.json")).read_text())
    return {
        "architecture": cfg.policy.architecture, "training_type": _training_type(cfg, parent is not None),
        "parent_checkpoint": parent, "file_path": str(path.resolve()), "sha256": file_hash,
        "configuration": config_dict(cfg), "code_commit": "root repository has no commit",
        "code_snapshot_sha256": payload["code_snapshot_sha256"], "data_split": splits,
        "encoder_version": normalization["encoder"], "vision_weight_versions": {"dino": vision_manifest["dino"], "siglip": vision_manifest["siglip"]},
        "training_update_count": payload["update"], "checkpoint_selection_metric": selection,
        "resume_command": payload["resume_command"], "parameter_counts": payload["parameter_counts"],
        "wandb": payload.get("wandb"), "determinism": payload.get("determinism"),
    }


def _training_payload(model, optimizer, cfg, *, architecture: str, rtc: bool, parent_checkpoint: str | None,
                      update: int, best: float, history: list[dict[str, Any]], counts: dict[str, int],
                      loss_trace: list[dict[str, Any]], started: float, world: int, peak: int, resume_command: str,
                      wandb_info: dict[str, Any] | None) -> dict[str, Any]:
    wall = time.time() - started
    return {
        "architecture": architecture, "training_type": _training_type(cfg, rtc), "parent_checkpoint": parent_checkpoint,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update, "best_validation": best,
        "history": history, "loss_trace": loss_trace, "config": config_dict(cfg), "parameter_counts": counts, "code_snapshot_sha256": _source_hash(),
        "wall_seconds": wall, "world_size": world, "samples_seen": update * cfg.train.effective_batch_size,
        "gpu_hours": wall * world / 3600, "peak_memory_bytes_per_rank": peak, "resume_command": resume_command,
        "wandb": wandb_info,
        "determinism": {"enabled": cfg.train.deterministic, "seed": cfg.train.seed, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Publish a checkpoint only after its complete byte stream is durable.

    Capacity-sweep checkpoints can be several GiB.  Writing directly to the
    recovery filename lets a preemption (or a concurrent health check) observe
    a truncated zip archive.  A sibling temporary file plus ``os.replace``
    keeps the last complete snapshot available until the new one is ready.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(cfg: Config, output_path: str | Path, *, parent_checkpoint: str | None = None,
          resume: str | None = None, updates: int | None = None,
          wandb_resume_info: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if cfg.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world, local = _distributed(); device = torch.device("cuda", local); torch.cuda.set_device(device)
    seed_all(cfg.train.seed + rank, cfg.train.deterministic)
    rtc = parent_checkpoint is not None
    total_updates = int(updates or (cfg.train.rtc_updates if rtc else cfg.train.updates))
    parent_cache = _matching_parent_cache(cfg, parent_checkpoint) if rtc else None
    dataset = PreparedPolicyDataset(cfg.data.prepared_path, "train", include_rtc_history=rtc, parent_prediction_path=parent_cache)
    valset = PreparedPolicyDataset(cfg.data.prepared_path, "val", include_rtc_history=rtc, parent_prediction_path=parent_cache)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg.train.seed) if world > 1 else None
    val_sampler = DistributedSampler(valset, world, rank, shuffle=False) if world > 1 else None
    loader_generator = torch.Generator().manual_seed(cfg.train.seed + rank)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg.train.num_workers, pin_memory=True, persistent_workers=cfg.train.num_workers > 0,
                        collate_fn=collate_policy_batch, drop_last=True, generator=loader_generator)
    val_loader = DataLoader(valset, batch_size=cfg.train.batch_size, sampler=val_sampler, num_workers=min(2,cfg.train.num_workers),
                            pin_memory=True, collate_fn=collate_policy_batch)
    model = create_policy(cfg).to(device)
    parent = codec = None
    if rtc:
        parent, _ = load_policy_checkpoint(parent_checkpoint, cfg, device)
        parent.requires_grad_(False).eval(); codec = create_action_codec(cfg, device)
        model.load_state_dict(parent.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9,0.95), weight_decay=cfg.train.weight_decay)
    start = 0; best = float("inf"); history = []; loss_trace = []; output_path = Path(output_path); resume_info = wandb_resume_info
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start = int(state["update"]); best = float(state.get("best_validation", best)); history = state.get("history", []); loss_trace = state.get("loss_trace", [])
        resume_info = state.get("wandb")
    counts = parameter_groups(model)
    train_model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    accumulation = cfg.train.effective_batch_size // (cfg.train.batch_size * world)
    if accumulation < 1 or accumulation * cfg.train.batch_size * world != cfg.train.effective_batch_size:
        raise ValueError("effective_batch_size must be an integer multiple of batch_size * world_size")
    iterator = iter(loader); epoch = 0; optimizer.zero_grad(set_to_none=True); started = time.time(); peak = 0
    tracker = WandbTracker(cfg, cfg.policy.architecture, rtc, output_path, resume_info=resume_info) if rank == 0 else None
    if tracker and tracker.info:
        wandb_sidecar = output_path.with_suffix(output_path.suffix + ".wandb.json")
        wandb_sidecar.parent.mkdir(parents=True, exist_ok=True)
        temporary = wandb_sidecar.with_name(f".{wandb_sidecar.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(tracker.info, indent=2) + "\n")
        os.replace(temporary, wandb_sidecar)
    resume_command = f"torchrun --standalone --nproc_per_node={world} -m robot_policy.cli {'finetune_rtc' if rtc else 'train_base'} --config configs/default.yaml --architecture {cfg.policy.architecture} --output {output_path}"
    if rtc: resume_command += f" --parent {parent_checkpoint}"
    for step in range(start, total_updates):
        aggregate: dict[str,float] = {}
        for micro in range(accumulation):
            try: batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None: sampler.set_epoch(epoch)
                iterator = iter(loader); batch = next(iterator)
            batch = _move(batch, device); condition = None
            if rtc:
                delays = _sample_training_delays(cfg, len(batch["state"]), device)
                previous, has_previous = _previous_batch(batch, delays)
                condition = make_rtc_condition(parent, previous, cfg.policy.architecture, codec, delays, has_previous,
                                               _cached_parent_prediction(batch, delays))
            sync = nullcontext() if world == 1 or micro == accumulation - 1 else train_model.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.precision == "bf16"):
                result = train_model(batch, condition)
                loss = result["loss"] / accumulation
            loss.backward()
            for k,v in result.items(): aggregate[k]=aggregate.get(k,0)+float(v)/accumulation
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        multiplier = _lr_multiplier(step,total_updates,cfg.train.warmup_updates,cfg.train.min_lr_ratio)
        for group in optimizer.param_groups: group["lr"] = cfg.train.learning_rate * multiplier
        peak = max(peak, torch.cuda.max_memory_allocated(device)); aggregate.update(update=step+1,lr=optimizer.param_groups[0]["lr"],grad_norm=grad_norm)
        if rank == 0:
            record = {"update": step + 1, **aggregate}
            loss_trace.append(record)
            tracker.log_train(step + 1, {k: v for k, v in aggregate.items() if k != "update"})
        if rank == 0 and ((step+1)%10==0 or step==start):
            print(json.dumps(aggregate),flush=True)
        if (step+1)%cfg.train.eval_every==0 or step+1==total_updates:
            metrics=validate(model,val_loader,device,cfg,parent,codec); score=metrics["loss"]
            if rank==0:
                history.append({"update":step+1,"train":aggregate,"validation":metrics}); print("validation",json.dumps(history[-1]),flush=True)
                tracker.log_validation(step + 1, metrics)
                if score < best: best=score
        if rank == 0 and (step + 1) % cfg.train.save_every == 0 and step + 1 < total_updates:
            resume_path = output_path.with_suffix(output_path.suffix + ".resume")
            _atomic_torch_save(_training_payload(model, optimizer, cfg, architecture=cfg.policy.architecture, rtc=rtc,
                       parent_checkpoint=parent_checkpoint, update=step+1, best=best, history=history, counts=counts,
                       loss_trace=loss_trace, started=started, world=world, peak=peak, resume_command=resume_command,
                       wandb_info=tracker.info if tracker else None), resume_path)
        if world>1: dist.barrier()
    if rank != 0:
        dist.destroy_process_group(); return None
    payload = _training_payload(model, optimizer, cfg, architecture=cfg.policy.architecture, rtc=rtc,
               parent_checkpoint=parent_checkpoint, update=total_updates, best=best, history=history, counts=counts,
               loss_trace=loss_trace, started=started, world=world, peak=peak, resume_command=resume_command,
               wandb_info=tracker.info if tracker else None)
    _atomic_torch_save(payload, output_path)
    # Prove clean-process loadability now, before publishing the manifest.
    reloaded,_=load_policy_checkpoint(output_path,cfg,"cpu"); del reloaded
    final_validation = history[-1]["validation"]["loss"] if history else None
    selection={"name":"final validation objective","value":final_validation,"split":"val","best_observed_value":best}
    entry=_checkpoint_manifest_entry(output_path,payload,cfg,selection,parent_checkpoint)
    manifest_path=output_path.parent/"checkpoint_manifest.json"
    lock_path=manifest_path.with_suffix(manifest_path.suffix+".lock")
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a+") as manifest_lock:
        fcntl.flock(manifest_lock,fcntl.LOCK_EX)
        existing=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"checkpoints":[]}
        existing["checkpoints"]=[x for x in existing["checkpoints"] if x["architecture"]!=cfg.policy.architecture or x["training_type"]!=entry["training_type"]]
        existing["checkpoints"].append(entry)
        temporary=manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(existing,indent=2)+"\n")
        os.replace(temporary,manifest_path)
    if tracker:
        tracker.finish(output_path, {"final_validation_loss": final_validation, "best_validation_loss": best, "training_updates": total_updates})
    if world>1: dist.destroy_process_group()
    return entry
