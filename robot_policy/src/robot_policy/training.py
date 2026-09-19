from __future__ import annotations

from contextlib import nullcontext
import copy
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

from robot_policy.config import (
    Config,
    config_dict,
    is_continuous_architecture,
    require_active_architecture,
    require_active_model_size,
)
from robot_policy.data.dataset import collate_policy_batch, create_policy_dataset
from robot_policy.policies import create_policy, load_policy_checkpoint
from robot_policy.policies.common import parameter_groups
from robot_policy.rtc.delay_mapping import sample_raw_delays, sample_spline_delays
from robot_policy.rtc.training import create_action_codec, make_rtc_condition
from robot_policy.tracking import WandbTracker


class _EMAModel:
    """Reference BSP EMA warmup, kept local to avoid another dependency."""

    def __init__(self, model: torch.nn.Module, cfg: Config):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.update_after_step = cfg.train.ema_update_after_step
        self.inv_gamma = cfg.train.ema_inv_gamma
        self.power = cfg.train.ema_power
        self.minimum = cfg.train.ema_min_value
        self.maximum = cfg.train.ema_max_value
        self.optimization_step = 0
        self.decay = 0.0

    def _decay(self) -> float:
        step = max(0, self.optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.minimum, min(value, self.maximum))

    @torch.no_grad()
    def step(self, online: torch.nn.Module) -> None:
        self.decay = self._decay()
        for averaged, current in zip(self.model.parameters(), online.parameters()):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1 - self.decay)
        for averaged, current in zip(self.model.buffers(), online.buffers()):
            averaged.copy_(current.detach())
        self.optimization_step += 1


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
    observation_key = "images" if "previous_images" in batch else "vision_features"
    previous = {
        observation_key: batch[f"previous_{observation_key}"][row, idx],
        "state": batch["previous_state"][row, idx],
    }
    return previous, batch["has_previous"][row, idx]


def _generation_input(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    observation_key = "images" if "images" in batch else "vision_features"
    return {observation_key: batch[observation_key], "state": batch["state"]}


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
    root = (
        Path(cfg.data.parent_prediction_cache_path)
        if cfg.data.parent_prediction_cache_path
        else Path(cfg.data.prepared_path) / "parent_predictions"
    ) / cfg.policy.architecture
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


def _skip_optimizer_step(architecture: str, grad_norm: float, fm_threshold: float,
                         update: int, fm_after_update: int) -> bool:
    """Reject non-finite gradients and finite FM spikes seen under BF16 attention.

    Base FM gets a warm-up allowance at the call site.  RTC starts from a
    trained parent, so the same finite guard applies from its first update.
    """
    return not math.isfinite(grad_norm) or (
        is_continuous_architecture(architecture)
        and update > fm_after_update
        and grad_norm > fm_threshold
    )


@torch.no_grad()
def validate(model, loader, device, cfg, rtc_parent=None, codec=None, max_batches: int = 16) -> dict[str, float]:
    """Measure unconditional generation, never a teacher-assisted denoising loss.

    Validation intentionally gives ``sample`` only observation/state tensors.
    Ground-truth controls and tokens are used after generation solely to score
    the completed trajectory.  A fixed, forked RNG makes each checkpoint face
    the same from-scratch noise/masking draw without perturbing training RNG.
    ``rtc_parent`` is retained in the signature for checkpoint compatibility,
    but RTC prefixes are deliberately not supplied during validation.
    """
    if codec is None:
        raise ValueError("from-scratch validation requires an action codec")
    model.eval(); totals: dict[str, float] = {}; count = 0
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.train.seed + 100_000)
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches: break
            batch = _move(batch, device)
            generation_input = _generation_input(batch)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(
                autocast_device,
                dtype=torch.bfloat16,
                enabled=cfg.train.precision == "bf16" and device.type == "cuda",
            ):
                prediction = model.sample(generation_input)
            controls = prediction.float() if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(prediction)
            action_mse = model.decoded_action_mse(controls, batch)
            control_valid = batch["control_valid_mask"].bool()
            control_mse = ((controls.float() - batch["continuous_target"].float()) ** 2)[control_valid].mean()
            metrics = {
                "loss": action_mse,
                "action_mse": action_mse,
                "generation_action_mse": action_mse,
                "generation_control_mse": control_mse,
            }
            if not is_continuous_architecture(cfg.policy.architecture):
                token_accuracy = (prediction == batch["discrete_target"]).masked_select(control_valid).float().mean()
                metrics["generation_token_accuracy"] = token_accuracy
            batch_size = len(batch["state"])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    model.train()
    if count == 0:
        raise RuntimeError("validation loader produced no batches")
    return {key: value / count for key, value in totals.items()}


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
    if cfg.data.observation_source == "rgb":
        rgb_root = Path(cfg.data.rgb_cache_path).resolve() if cfg.data.rgb_cache_path else prepared / "rgb"
        observation_versions = {"bsp_rgb_cache": json.loads((rgb_root / "manifest.json").read_text())}
    else:
        vision_root = Path(cfg.data.vision_cache_path).resolve() if cfg.data.vision_cache_path else prepared / "vision"
        vision_manifest = json.loads(next(vision_root.glob("worker_*_manifest.json")).read_text())
        observation_versions = {"dino": vision_manifest["dino"], "siglip": vision_manifest["siglip"]}
    return {
        "architecture": cfg.policy.architecture, "training_type": _training_type(cfg, parent is not None),
        "parent_checkpoint": parent, "file_path": str(path.resolve()), "sha256": file_hash,
        "configuration": config_dict(cfg), "code_commit": "root repository has no commit",
        "code_snapshot_sha256": payload["code_snapshot_sha256"], "data_split": splits,
        "encoder_version": normalization["encoder"], "vision_weight_versions": observation_versions,
        "training_update_count": payload["update"], "checkpoint_selection_metric": selection,
        "resume_command": payload["resume_command"], "parameter_counts": payload["parameter_counts"],
        "wandb": payload.get("wandb"), "determinism": payload.get("determinism"),
    }


def _training_payload(model, optimizer, cfg, *, ema: _EMAModel | None, architecture: str, rtc: bool, parent_checkpoint: str | None,
                      update: int, best: float, history: list[dict[str, Any]], counts: dict[str, int],
                      loss_trace: list[dict[str, Any]], started: float, world: int, peak: int, resume_command: str,
                      wandb_info: dict[str, Any] | None,
                      inference_state: dict[str, torch.Tensor] | None = None,
                      selected_update: int | None = None) -> dict[str, Any]:
    wall = time.time() - started
    return {
        "architecture": architecture, "training_type": _training_type(cfg, rtc), "parent_checkpoint": parent_checkpoint,
        "model": inference_state if inference_state is not None else (ema.model if ema is not None else model).state_dict(),
        "online_model": model.state_dict() if ema is not None else None,
        "ema_optimization_step": ema.optimization_step if ema is not None else None,
        "ema_decay": ema.decay if ema is not None else None,
        "optimizer": optimizer.state_dict(), "update": update, "best_validation": best,
        "selected_update": int(selected_update if selected_update is not None else update),
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
    require_active_architecture(cfg.policy.architecture, "training")
    require_active_model_size(cfg.policy.model_size, "training")
    if cfg.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world, local = _distributed(); device = torch.device("cuda", local); torch.cuda.set_device(device)
    # The optimized CUDA SDP backward became numerically singular for trained
    # DiT-S FM weights (finite forward/loss, NaN or ~1e20 parameter gradients).
    # Use the stable math kernel only in FM training processes.  Evaluation and
    # deployment remain on the fast inference backend because they are no-grad.
    if cfg.policy.architecture == "fm" and cfg.train.fm_math_sdp_training:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    seed_all(cfg.train.seed + rank, cfg.train.deterministic)
    rtc = parent_checkpoint is not None
    total_updates = int(updates or (cfg.train.rtc_updates if rtc else cfg.train.updates))
    parent_cache = _matching_parent_cache(cfg, parent_checkpoint) if rtc else None
    dataset = create_policy_dataset(cfg, "train", include_rtc_history=rtc, parent_prediction_path=parent_cache)
    valset = create_policy_dataset(cfg, "val", include_rtc_history=rtc, parent_prediction_path=parent_cache)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg.train.seed) if world > 1 else None
    val_sampler = DistributedSampler(valset, world, rank, shuffle=False) if world > 1 else None
    loader_generator = torch.Generator().manual_seed(cfg.train.seed + rank)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg.train.num_workers, pin_memory=True, persistent_workers=cfg.train.num_workers > 0,
                        collate_fn=collate_policy_batch, drop_last=True, generator=loader_generator)
    val_loader = DataLoader(valset, batch_size=cfg.train.batch_size, sampler=val_sampler, num_workers=min(2,cfg.train.num_workers),
                            pin_memory=True, collate_fn=collate_policy_batch)
    model = create_policy(cfg).to(device)
    codec = create_action_codec(cfg, device)
    parent = None
    if rtc:
        parent, _ = load_policy_checkpoint(parent_checkpoint, cfg, device)
        parent.requires_grad_(False).eval()
        model.load_state_dict(parent.state_dict())
    learning_rate = (
        cfg.train.rtc_learning_rate
        if rtc and cfg.train.rtc_learning_rate is not None
        else cfg.train.learning_rate
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(cfg.train.optimizer_beta1, cfg.train.optimizer_beta2),
        eps=cfg.train.optimizer_epsilon,
        weight_decay=cfg.train.weight_decay,
    )
    ema = _EMAModel(model, cfg) if cfg.train.use_ema else None
    start = 0; best = float("inf"); history = []; loss_trace = []; output_path = Path(output_path); resume_info = wandb_resume_info
    best_weights_path = output_path.with_suffix(output_path.suffix + ".best.weights.pt")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("online_model") or state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None:
            ema.model.load_state_dict(state["model"])
            ema.optimization_step = int(state.get("ema_optimization_step") or state["update"])
            ema.decay = float(state.get("ema_decay") or 0.0)
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
    config_path = getattr(cfg, "_source_path", "configs/default.yaml")
    resume_command = f"torchrun --standalone --nproc_per_node={world} -m robot_policy.cli {'finetune_rtc' if rtc else 'train_base'} --config {config_path} --architecture {cfg.policy.architecture} --output {output_path}"
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
        optimizer_step_skipped = _skip_optimizer_step(
            cfg.policy.architecture,
            grad_norm,
            (
                cfg.train.fm_rtc_grad_skip_threshold
                if rtc
                else cfg.train.fm_grad_skip_threshold
            ),
            step + 1,
            0 if rtc else cfg.train.fm_grad_skip_after_updates,
        )
        if not optimizer_step_skipped:
            optimizer.step()
            if ema is not None:
                ema.step(model)
        optimizer.zero_grad(set_to_none=True)
        aggregate["optimizer_step_skipped"] = float(optimizer_step_skipped)
        multiplier = _lr_multiplier(step,total_updates,cfg.train.warmup_updates,cfg.train.min_lr_ratio)
        for group in optimizer.param_groups: group["lr"] = learning_rate * multiplier
        peak = max(peak, torch.cuda.max_memory_allocated(device)); aggregate.update(update=step+1,lr=optimizer.param_groups[0]["lr"],grad_norm=grad_norm)
        if rank == 0:
            record = {"update": step + 1, **aggregate}
            loss_trace.append(record)
            tracker.log_train(step + 1, {k: v for k, v in aggregate.items() if k != "update"})
        if rank == 0 and ((step+1)%10==0 or step==start):
            print(json.dumps(aggregate),flush=True)
        if (step+1)%cfg.train.eval_every==0 or step+1==total_updates:
            metrics=validate(
                ema.model if ema is not None else model,
                val_loader,
                device,
                cfg,
                parent,
                codec,
                max_batches=cfg.train.validation_max_batches,
            ); score=metrics["action_mse"]
            if rank==0:
                history.append({"update":step+1,"train":aggregate,"validation":metrics}); print("validation",json.dumps(history[-1]),flush=True)
                tracker.log_validation(step + 1, metrics)
                if score < best:
                    best=score
                    selected_model = ema.model if ema is not None else model
                    _atomic_torch_save(
                        {
                            "model": selected_model.state_dict(),
                            "update": step + 1,
                            "validation_action_mse": score,
                        },
                        best_weights_path,
                    )
        if rank == 0 and (step + 1) % cfg.train.save_every == 0 and step + 1 < total_updates:
            resume_path = output_path.with_suffix(output_path.suffix + ".resume")
            _atomic_torch_save(_training_payload(model, optimizer, cfg, ema=ema, architecture=cfg.policy.architecture, rtc=rtc,
                       parent_checkpoint=parent_checkpoint, update=step+1, best=best, history=history, counts=counts,
                       loss_trace=loss_trace, started=started, world=world, peak=peak, resume_command=resume_command,
                       wandb_info=tracker.info if tracker else None), resume_path)
        if world>1: dist.barrier()
    if rank != 0:
        dist.destroy_process_group(); return None
    if best_weights_path.exists():
        selected = torch.load(best_weights_path, map_location="cpu", weights_only=False)
        inference_state = selected["model"]
        selected_update = int(selected["update"])
    else:
        inference_state = None
        selected_update = total_updates
    payload = _training_payload(model, optimizer, cfg, ema=ema, architecture=cfg.policy.architecture, rtc=rtc,
               parent_checkpoint=parent_checkpoint, update=total_updates, best=best, history=history, counts=counts,
               loss_trace=loss_trace, started=started, world=world, peak=peak, resume_command=resume_command,
               wandb_info=tracker.info if tracker else None, inference_state=inference_state,
               selected_update=selected_update)
    _atomic_torch_save(payload, output_path)
    # Prove clean-process loadability now, before publishing the manifest.
    reloaded,_=load_policy_checkpoint(output_path,cfg,"cpu"); del reloaded
    final_validation = history[-1]["validation"]["action_mse"] if history else None
    selection={"name":"from-scratch normalized generation action MSE","value":best,"split":"val",
               "best_observed_value":best,"selected_update":selected_update,
               "final_observed_value":final_validation}
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
        tracker.finish(output_path, {
            "final_validation_action_mse": final_validation,
            "best_validation_action_mse": best,
            "selected_update": selected_update,
            "validation_mode": "from_scratch_generation",
            "training_updates": total_updates,
        })
    best_weights_path.unlink(missing_ok=True)
    if world>1: dist.destroy_process_group()
    return entry
