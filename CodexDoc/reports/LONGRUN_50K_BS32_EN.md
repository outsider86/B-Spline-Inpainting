# 50k Basic / 5k ttRTC, Batch-32 Training Report

**Report date:** 2026-09-16  
**Status:** Complete.

## Delivered scope

- Six basic checkpoints: three architectures × raw/B-spline, each trained for exactly **50,000 optimizer updates**.
- Six ttRTC checkpoints: the matching six variants, each fine-tuned for exactly **5,000 optimizer updates** from its own 50k parent.
- Both micro-batch and effective batch size are **32**; there is no hidden gradient accumulation.
- Exactly two new W&B projects were used, with exactly six production runs in each.
- Every run logs `train/action_mse` and `validation/action_mse` and contains a model artifact.

## W&B projects

- Basic, 6/6 finished: [robot-policy-50k-bs32-basic](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic)
- ttRTC, 6/6 finished: [robot-policy-5k-bs32-ttRTC](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC)

### Basic runs

| Representation | Architecture | Run ID | Updates | Final validation `action_mse` |
|---|---|---|---:|---:|
| Raw | FM | [`y7cijqp5`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/y7cijqp5) | 50,000 | 0.029513 |
| Raw | Layerwise DD | [`jnwtiwdf`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/jnwtiwdf) | 50,000 | 0.003621 |
| Raw | Joint DD | [`p3i9bwzy`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/p3i9bwzy) | 50,000 | 0.011324 |
| B-spline | FM | [`dtwgmcv5`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/dtwgmcv5) | 50,000 | 0.026211 |
| B-spline | Layerwise DD | [`jwco3ma9`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/jwco3ma9) | 50,000 | 0.002321 |
| B-spline | Joint DD | [`m6nqhqyl`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/m6nqhqyl) | 50,000 | 0.009827 |

### ttRTC runs

| Representation | Architecture | Run ID | Updates | Final validation `action_mse` |
|---|---|---|---:|---:|
| Raw | FM | [`mlaw9wev`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/mlaw9wev) | 5,000 | 0.030458 |
| Raw | Layerwise DD | [`orcbvl0u`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/orcbvl0u) | 5,000 | 0.004734 |
| Raw | Joint DD | [`zrscqaje`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/zrscqaje) | 5,000 | 0.007063 |
| B-spline | FM | [`5qxw6xgv`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/5qxw6xgv) | 5,000 | 0.025992 |
| B-spline | Layerwise DD | [`kjfy1n9h`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/kjfy1n9h) | 5,000 | 0.001907 |
| B-spline | Joint DD | [`oz24h02u`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/oz24h02u) | 5,000 | 0.013061 |

These are normalized corruption-conditioned validation diagnostics, not held-out physical-space rollout results. No new open-loop evaluation was requested in this training deliverable.

## Checkpoint artifacts

- Raw checkpoints and manifest: `robot_policy/outputs/longrun_50k_bs32/raw/checkpoints/`
- B-spline checkpoints and manifest: `robot_policy/outputs/longrun_50k_bs32/bspline/checkpoints/`
- Raw configuration: `robot_policy/configs/longrun_50k_bs32_raw.yaml`
- B-spline configuration: `robot_policy/configs/longrun_50k_bs32_bspline.yaml`

Raw ttRTC files use the explicit `_ttrtc.pt` suffix. B-spline files retain the repository's established `_rtc.pt` suffix; they are the requested ttRTC checkpoints and their W&B names use `ttrtc`.

## Integrity checks

- Local checkpoint records audited: **12/12 passed**.
- Independent CPU reload through the policy loader: **12/12 passed**.
- Training update counts: all basic `50,000`; all ttRTC `5,000`.
- Samples seen: basic `1,600,000`; ttRTC `160,000` per checkpoint.
- Batch size and effective batch size: both `32` in every payload and W&B config.
- Parent lineage: every ttRTC checkpoint points to the matching representation/architecture 50k parent; manifest SHA-256 values match.
- Joint parent caches: 52 episodes / 31,706 frames for raw and B-spline, with checkpoint hashes matched and exact integer tokens.
- W&B: exactly 6 + 6 runs, all `finished`, required action-MSE keys present, model artifact present for every run.
- Project tests: **14 passed**.

## Next steps

1. Run the same held-out open-loop, delay, replay, and latency evaluation suite on these 12 long-run checkpoints.
2. Compare 50k/5k results with the earlier 2k/800 experiments using decoded physical action MSE.
3. Add additional seeds if statistical variance, rather than deterministic single-seed training, is required.
