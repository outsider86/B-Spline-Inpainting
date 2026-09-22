from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_policy.config import (
    TRAINABLE_ARCHITECTURES,
    is_continuous_architecture,
    load_config,
    require_active_architecture,
    require_active_model_size,
)
from robot_policy.data.dataset import collate_policy_batch, create_policy_dataset
from robot_policy.policies import load_policy_checkpoint
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import create_action_codec


def _summary(values: list[np.ndarray]) -> dict[str, float] | None:
    if not values:
        return None
    x=np.concatenate([v.reshape(-1) for v in values])
    return {"mae":float(np.abs(x).mean()),"mse":float(np.mean(x*x)),"rmse":float(np.sqrt(np.mean(x*x))),"max_abs":float(np.abs(x).max()),
            "p95_abs":float(np.quantile(np.abs(x),.95)),"p99_abs":float(np.quantile(np.abs(x),.99))}


def _stack(dataset, indices, device):
    batch=collate_policy_batch([dataset[i] for i in indices])
    return {k:v.to(device) for k,v in batch.items()}


def select_rtc_indices(
    dataset,
    count: int,
    minimum_frame: int,
    *,
    strategy: str = "leading",
    seed: int = 20260915,
) -> list[int]:
    """Select RTC windows while preserving the historical leading cohort.

    The episode-balanced strategy shuffles eligible windows within each
    episode, then draws one per episode in round-robin order. This prevents a
    small cohort from silently measuring only the first episode in a split.
    """
    if count < 1:
        raise ValueError("RTC sample count must be positive")
    eligible = [
        (index, episode_id)
        for index, (episode_id, frame) in enumerate(dataset.index)
        if frame >= minimum_frame
    ]
    if not eligible:
        raise ValueError("the requested split contains no RTC-eligible windows")
    if strategy == "leading":
        return [index for index, _ in eligible[:count]]
    if strategy not in {"episode_balanced", "episode_balanced_motion"}:
        raise ValueError(f"unknown RTC sample strategy {strategy!r}")

    by_episode: dict[int, list[int]] = {}
    for index, episode_id in eligible:
        by_episode.setdefault(episode_id, []).append(index)
    rng = np.random.default_rng(seed)
    if strategy == "episode_balanced":
        for values in by_episode.values():
            rng.shuffle(values)
    else:
        for episode_id, values in by_episode.items():
            action_path = dataset.root / "actions" / f"episode_{episode_id:06d}.npz"
            with np.load(action_path) as action_data:
                normalized = action_data["normalized_action"]
                horizon = int(action_data["action_valid_mask"].shape[1])
            scored: list[tuple[float, int, int]] = []
            for index in values:
                _, frame = dataset.index[index]
                # A motion diagnostic must not rank windows whose tail is
                # repeated padding from the final recorded action.
                if frame + horizon > len(normalized):
                    continue
                trajectory = normalized[frame : frame + horizon]
                score = float(np.abs(np.diff(trajectory, axis=0)).sum())
                scored.append((score, frame, index))
            scored.sort(key=lambda item: (-item[0], item[1]))
            # Greedily spread selections across time so the cohort does not
            # consist of many overlapping windows around one transition.
            minimum_spacing = max(1, horizon // 3)
            diverse: list[tuple[float, int, int]] = []
            deferred: list[tuple[float, int, int]] = []
            for candidate in scored:
                if all(abs(candidate[1] - chosen[1]) >= minimum_spacing for chosen in diverse):
                    diverse.append(candidate)
                else:
                    deferred.append(candidate)
            by_episode[episode_id] = [item[2] for item in (*diverse, *deferred)]
    episodes = sorted(by_episode)
    selected: list[int] = []
    cursor = {episode_id: 0 for episode_id in episodes}
    while len(selected) < count:
        progressed = False
        for episode_id in episodes:
            position = cursor[episode_id]
            values = by_episode[episode_id]
            if position < len(values):
                selected.append(values[position])
                cursor[episode_id] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _selection_metadata(dataset, indices: list[int], strategy: str, seed: int) -> dict[str, Any]:
    examples = [dataset.index[index] for index in indices]
    digest_input = "\n".join(f"{episode_id}:{frame}" for episode_id, frame in examples)
    motion_scores = []
    by_episode: dict[int, list[int]] = {}
    for episode_id, frame in examples:
        by_episode.setdefault(episode_id, []).append(frame)
    for episode_id, frames in by_episode.items():
        action_path = dataset.root / "actions" / f"episode_{episode_id:06d}.npz"
        with np.load(action_path) as action_data:
            normalized = action_data["normalized_action"]
            horizon = int(action_data["action_valid_mask"].shape[1])
        for frame in frames:
            end = min(frame + horizon, len(normalized))
            motion_scores.append(float(np.abs(np.diff(normalized[frame:end], axis=0)).sum()))
    return {
        "strategy": strategy,
        "seed": seed if strategy == "episode_balanced" else None,
        "episode_counts": {
            str(episode_id): count
            for episode_id, count in sorted(Counter(episode_id for episode_id, _ in examples).items())
        },
        "sample_ids_sha256": hashlib.sha256(digest_input.encode()).hexdigest(),
        "motion_score": {
            "definition": "sum(abs(diff(normalized ground-truth 30-step trajectory)))",
            "minimum": float(np.min(motion_scores)),
            "median": float(np.median(motion_scores)),
            "mean": float(np.mean(motion_scores)),
            "maximum": float(np.max(motion_scores)),
        },
    }


def evaluate(
    cfg,
    checkpoint: str,
    max_samples: int = 128,
    batch_size: int = 32,
    split: str = "test",
    sample_strategy: str = "leading",
    sample_seed: int = 20260915,
    condition_source: str = "previous_plan",
) -> dict[str, Any]:
    require_active_architecture(cfg.policy.architecture, "RTC evaluation")
    require_active_model_size(cfg.policy.model_size, "RTC evaluation")
    device=torch.device("cuda"); model,payload=load_policy_checkpoint(checkpoint,cfg,device); model.eval()
    if split not in {"train", "val", "test"}:
        raise ValueError("RTC split must be 'train', 'val', or 'test'")
    if condition_source not in {"ground_truth", "previous_plan"}:
        raise ValueError("RTC condition_source must be 'ground_truth' or 'previous_plan'")
    codec=create_action_codec(cfg,device); data=create_policy_dataset(cfg,split)
    lookup={pair:i for i,pair in enumerate(data.index)}
    current_indices=select_rtc_indices(
        data,
        max_samples,
        0 if condition_source == "ground_truth" else cfg.rtc.raw_delay_max,
        strategy=sample_strategy,
        seed=sample_seed,
    )
    stats=json.loads((Path(cfg.data.prepared_path)/"normalization.json").read_text())
    low=torch.tensor(stats["action_q01"],device=device); high=torch.tensor(stats["action_q99"],device=device)
    curves=[]
    for raw_delay in range(0,cfg.rtc.raw_delay_max+1):
        mapping=map_delay(raw_delay,frequency_hz=cfg.data.frequency_hz,span_length_steps=cfg.spline.span_length_steps,degree=cfg.spline.degree) if cfg.data.action_representation=="bspline" else None
        trajectory_errors=[]; suffix_errors=[]; preservation=[]; requantization=[]; fixed_errors=[]; switch_velocity=[]; switch_acceleration=[]
        torch.manual_seed(20260915+raw_delay)
        for offset in range(0,len(current_indices),batch_size):
            chosen=current_indices[offset:offset+batch_size]; current=_stack(data,chosen,device)
            fixed=prefix=shifted=prior_actions=None
            if raw_delay:
                raw=torch.full((len(chosen),),raw_delay,device=device,dtype=torch.long)
                if condition_source == "ground_truth":
                    shifted = current["continuous_target"].float()
                    prior_controls = shifted
                else:
                    previous_indices=[lookup[(data.index[i][0],data.index[i][1]-raw_delay)] for i in chosen]
                    previous=_stack(data,previous_indices,device)
                    with torch.no_grad():
                        prior=model.sample(previous)
                    prior_controls=prior.float() if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(prior)
                    shifted=codec.shift_and_refit(prior_controls,raw)
                if cfg.data.action_representation=="bspline":
                    affected=torch.full_like(raw,mapping.affected_spans)
                    fixed=control_support_mask(
                        affected,
                        cfg.spline.num_basis,
                        shifted.shape[-1],
                        cfg.spline.degree,
                    )
                else:
                    fixed=raw_action_prefix_mask(raw,cfg.data.action_horizon)
                prefix=(
                    shifted
                    if is_continuous_architecture(cfg.policy.architecture)
                    else current["discrete_target"]
                    if condition_source == "ground_truth"
                    else codec.encode_tokens(shifted)
                )
                if condition_source == "previous_plan":
                    prior_actions=codec.decode_controls(prior_controls)
            if (
                prefix is not None
                and is_continuous_architecture(cfg.policy.architecture)
                and str(payload.get("training_type", "base")).lower() == "base"
            ):
                predicted = model.sample_realtime_pigdm(
                    current,
                    prefix_values=prefix,
                    fixed_mask=fixed,
                )
            else:
                with torch.no_grad():
                    predicted=model.sample(current,prefix_values=prefix,fixed_mask=fixed,use_cache=True)
            controls=predicted.float() if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(predicted)
            decoded=codec.decode_controls(controls)
            physical=(decoded+1)*.5*(high-low)+low
            err=physical-current["target_trajectory"].float(); valid=current["action_valid_mask"].bool()
            trajectory_errors.extend([err[i,valid[i]].cpu().numpy() for i in range(len(err))])
            suffix_valid = valid & (
                torch.arange(valid.shape[1], device=device)[None] >= raw_delay
            )
            suffix_errors.extend(
                [err[i, suffix_valid[i]].cpu().numpy() for i in range(len(err))]
            )
            if raw_delay:
                reference_controls=shifted if is_continuous_architecture(cfg.policy.architecture) else codec.decode_tokens(prefix)
                reference_actions=codec.decode_controls(reference_controls); shifted_actions=codec.decode_controls(shifted)
                preservation.append((decoded[:,:raw_delay]-reference_actions[:,:raw_delay]).cpu().numpy())
                requantization.append((reference_actions[:,:raw_delay]-shifted_actions[:,:raw_delay]).cpu().numpy())
                fixed_errors.append((controls-reference_controls).masked_select(fixed).cpu().numpy())
                if prior_actions is not None:
                    scale=.5*(high-low)
                    jump=(decoded[:,0]-prior_actions[:,raw_delay-1])*scale*cfg.data.frequency_hz
                    switch_velocity.append(jump.cpu().numpy())
                    if raw_delay>=2:
                        prev_velocity=(prior_actions[:,raw_delay-1]-prior_actions[:,raw_delay-2])*scale*cfg.data.frequency_hz
                        switch_acceleration.append(((jump-prev_velocity)*cfg.data.frequency_hz).cpu().numpy())
        curves.append({"raw_delay_actions":raw_delay,"delay_ms":1000*raw_delay/cfg.data.frequency_hz,"whole_spans":mapping.whole_spans if mapping else None,
                       "phase_steps":mapping.phase_steps if mapping else None,"affected_spans":mapping.affected_spans if mapping else None,
                       "committed_control_count":mapping.committed_control_count if mapping else raw_delay,
                       "training_delay_status":"zero-delay check" if raw_delay==0 else (("seen exact-span" if raw_delay%cfg.spline.span_length_steps==0 else "unseen within-span") if mapping else "seen raw-action delay"),
                       "decoded_physical_error":_summary(trajectory_errors),
                       "generated_suffix_physical_error":_summary(suffix_errors),
                       "committed_interval_preservation_normalized":_summary(preservation),
                       "discrete_requantization_preservation_error_normalized":_summary(requantization),
                       "fixed_control_error":_summary(fixed_errors),"switch_velocity_physical_per_s":_summary(switch_velocity),
                       "switch_acceleration_physical_per_s2":_summary(switch_acceleration)})
    checkpoint_training_type = str(payload["training_type"]).lower()
    rtc_method = (
        "pigdm_binary_hard_mask"
        if is_continuous_architecture(cfg.policy.architecture)
        and checkpoint_training_type == "base"
        else "finetuned_ttrtc_direct_hard_mask"
        if is_continuous_architecture(cfg.policy.architecture)
        else "discrete_hard_mask"
    )
    return {"architecture":cfg.policy.architecture,"action_representation":cfg.data.action_representation,"training_type":payload["training_type"],"checkpoint":str(Path(checkpoint).resolve()),
            "split":split,"samples_per_delay":len(current_indices),"rtc_inference_method":rtc_method,
            "sample_selection":_selection_metadata(data,current_indices,sample_strategy,sample_seed),
            "condition_source":(
                "ground-truth current trajectory prefix encoded in the policy representation"
                if condition_source == "ground_truth"
                else "previous generated chunk shifted to the current observation time"
            ),
            "delay_evaluation_support":"raw d=0..10 actions",
            "bspline_hard_mask_rule":"affected span count + cubic degree (3) control rows" if cfg.data.action_representation=="bspline" else None,
            "scope":"open-loop dataset replay; no environmental feedback or robot success measurement","curves":curves}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--set",action="append",default=[]); p.add_argument("--architecture",required=True,choices=TRAINABLE_ARCHITECTURES); p.add_argument("--checkpoint",required=True); p.add_argument("--max-samples",type=int,default=128); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--split",choices=("train","val","test"),default="test"); p.add_argument("--sample-strategy",choices=("leading","episode_balanced","episode_balanced_motion"),default="leading"); p.add_argument("--sample-seed",type=int,default=20260915); p.add_argument("--condition-source",choices=("ground_truth","previous_plan"),default="previous_plan"); p.add_argument("--output",required=True)
    a=p.parse_args(argv); cfg=load_config(a.config,[*a.set,f"policy.architecture={a.architecture}"]); report=evaluate(cfg,a.checkpoint,a.max_samples,a.batch_size,a.split,a.sample_strategy,a.sample_seed,a.condition_source)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
