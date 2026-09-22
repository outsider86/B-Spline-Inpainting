# DINOv2 Joint Discrete Diffusion h1/h2 Status

Date: 2026-09-21

## Scope

The active matrix contains eight checkpoints:

| Representation | Observation history | Base | ttRTC |
|---|---:|---:|---:|
| raw actions | h1 | 50k updates | 5k updates |
| raw actions | h2 | 50k updates | 5k updates |
| cubic B-spline tokens | h1 | 50k updates | 5k updates |
| cubic B-spline tokens | h2 | 50k updates | 5k updates |

All variants use joint categorical discrete diffusion. Layer-wise DD is not in
scope. The training corruption is a 50/50 mixture of full-MASK sequences and
the existing block corruption, so the model is trained explicitly on the same
all-MASK initial state used by generation from scratch. Inference uses eight
confidence-based block-unmasking rounds with a block size of 21.

## Visual observation architecture

The frozen vision backbone is DINOv2 ViT-L/14 (`vit_large_patch14_reg4_dinov2`),
using the second-to-last extraction layer. SigLIP is not used in this family.
For each camera and timestep:

1. DINOv2 produces 256 patch tokens with width 1,024.
2. An independent trainable 32-query cross-attention resampler is used for each
   camera. The global and hand cameras do not share resampler parameters.
3. The 32 + 32 visual tokens are followed by one projected robot-state token.

This gives 65 observation tokens for h1 and 130 for h2. The h2 path preserves
the full time and camera axes and adds deterministic temporal position
embeddings; it does not average or flatten the two timesteps into one vector.

## Data and cache

- Dataset: `Data/stacking_cups_action_30hz`
- Split: 42 train / 5 validation / 5 test episodes
- Shared cache: `robot_policy/outputs/DINO_DD_JOINT_H12/cache/dinov2_patch16`
- Coverage: 52 episodes, 31,706 frames
- Stored shape per frame: 2 cameras x 256 tokens x 1,024 channels, FP16
- Cache size: approximately 31 GiB

The cache, DINOv2 weights, W&B files, checkpoints, and logs all reside in the
scratch workspace. No experiment data was moved to the home directory.

## Verification completed

- 74 selected repository tests passed. The only excluded legacy test requires
  the intentionally removed 36-checkpoint SWEEP directory.
- Four base GPU smoke trainings passed for raw/B-spline x h1/h2.
- Four RTC GPU smoke trainings passed, including history lookup and the raw and
  B-spline hard-mask paths.
- Save/reload and generation-from-scratch validation passed for all smoke
  checkpoints.
- Static diff checking passed.

## Active training

Only GPUs 0--3 are reserved. GPUs 4--6 remain free for evaluations owned by
other agents.

| GPU | Variant | W&B run |
|---:|---|---|
| 0 | raw h1 base | [84mf4xkm](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/84mf4xkm) |
| 1 | raw h2 base | [e8e5u39c](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/e8e5u39c) |
| 2 | B-spline h1 base | [44ah3k06](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/44ah3k06) |
| 3 | B-spline h2 base | [gr8i2pn6](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/gr8i2pn6) |

Early optimization is finite and has no skipped optimizer steps. The reported
gradient norm is measured before clipping; clipping at 1.0 is enabled. The
launcher will build a hash-keyed parent-prediction cache for each completed
base checkpoint and then start its corresponding ttRTC finetune automatically.

Runtime state is recorded in
`robot_policy/outputs/DINO_DD_JOINT_H12/status.json`; per-variant logs are under
`robot_policy/outputs/DINO_DD_JOINT_H12/logs`.

The four jobs now use batch 64 with accumulation 1, preserving the original
effective batch size of 64 while making substantially better use of each GPU.
Validation covers the complete 3,077-window validation split every 500 updates.
The original short-validation W&B runs are retained as historical records; the
table above lists the continuation runs used for the authoritative full-split
curves. At the latest recorded check, raw h1/h2 were beyond 12.5k updates and
B-spline h1/h2 were beyond 15.5k updates. All optimizer updates remained finite
and none were skipped.

The local DD-OpenVLA D2F implementation was audited again. Its training mask
probability is monotonic across blocks and its attention permits a target block
to read only the observation, preceding blocks, and itself. The implemented
joint-DD policy has the same block-causal information boundary and commits
completed blocks to a KV cache. It intentionally does not copy D2F's
teacher-distillation loss: this experiment uses hard categorical targets and
adds 50% full-MASK exposure to match generation-from-scratch deployment.

## Next steps

1. Monitor full-validation generation-from-scratch action MSE and gradient
   stability through 50k base updates.
2. Complete the exact-parent caches and four 5k ttRTC finetunes.
3. Evaluate all eight checkpoints on train/test open-loop generation, prefix
   inpainting RTC, latency, and the real deployment interface.
4. Publish the final bilingual comparison and checkpoint bundle.
