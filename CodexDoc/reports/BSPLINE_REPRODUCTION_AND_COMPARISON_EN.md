# B-spline Reproduction and Raw-Action Comparison

**Report date:** 2026-09-16  
**Status:** Complete for deterministic B-spline training, ttRTC fine-tuning, evaluation, replay, latency, visualization, and W&B comparison.

## Outcome

The B-spline experiment was rerun under the same controlled protocol used for the raw-action experiment: the same dataset and episode split, three policy architectures, optimizer and update budgets, deterministic seed `7`, evaluation windows, and GPU class. Six production checkpoints were generated under `robot_policy/outputs/bspline_reproduction/` without overwriting the earlier historical artifacts.

The comparison is available in W&B:

- [Raw vs B-spline comparison project](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-action-representation-comparison)
- [Authoritative comparison run `eqwedr12`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-action-representation-comparison/runs/eqwedr12)
- [B-spline basic project](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic)
- [B-spline ttRTC project](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC)

The dedicated comparison run logs a 12-row W&B table, the decoded physical action-MSE comparison, the ttRTC-effect comparison, and machine-readable JSON/CSV as a W&B artifact.

## Comparable `action_mse` contract

`action_mse` is now representation-aware but comparable across raw and B-spline training:

- It is computed in normalized decoded `30 × 7` raw-action space.
- Only raw actions affected by corruption-supervised controls are scored.
- For B-splines, non-supervised controls are filled with ground truth before decoding so unrelated spline support does not contaminate the metric.
- With the raw identity representation, this reduces exactly to the original raw-action metric; an automated test checks exact equivalence.
- W&B uses `train/action_mse` and `validation/action_mse`; architecture-specific optimization objectives remain separate.

Evaluation action MSE is a second, primary metric: decoded physical action MSE over all 3,149 held-out windows. It is not mixed with the normalized corruption-conditioned training diagnostic.

## B-spline training results

All basic models used 2,000 updates and all ttRTC models used 800 updates, with effective batch size 128, AdamW, learning rate `3e-4`, BF16, and seed `7`.

| Architecture | Stage | Final validation objective | Final normalized validation `action_mse` | W&B run |
|---|---:|---:|---:|---|
| Flow matching | Basic | 0.058429 | **0.018705** | [`17qdogt2`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/17qdogt2) |
| Flow matching | ttRTC | 0.081424 | **0.021648** | [`ke2jdwfe`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/ke2jdwfe) |
| Layerwise discrete | Basic | 13.625690 | **0.014264** | [`8go0inek`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/8go0inek) |
| Layerwise discrete | ttRTC | 13.590894 | **0.010950** | [`0d87eo7m`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/0d87eo7m) |
| Joint discrete | Basic | 12.705885 | **0.013798** | [`h774y2hw`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/h774y2hw) |
| Joint discrete | ttRTC | 14.613434 | **0.014376** | [`ng3rdmdp`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/ng3rdmdp) |

The three exact basic reruns are `2wt5pssh`, `bbpykwqj`, and `wcblowty`. Each matched 2,000 training points, 20 validation points, and every final tensor with maximum delta zero.

## Held-out B-spline evaluation

| Architecture | Stage | Physical action MSE ↓ | MAE ↓ | RMSE ↓ | Token accuracy |
|---|---:|---:|---:|---:|---:|
| Flow matching | Basic | **0.013835** | 0.053852 | 0.117623 | N/A |
| Flow matching | ttRTC | **0.013294** | 0.055207 | 0.115299 | N/A |
| Layerwise discrete | Basic | **0.032380** | 0.101878 | 0.179944 | 0.1425 |
| Layerwise discrete | ttRTC | **0.045018** | 0.131219 | 0.212174 | 0.1164 |
| Joint discrete | Basic | **0.023766** | 0.075348 | 0.154162 | 0.1708 |
| Joint discrete | ttRTC | **0.020955** | 0.073556 | 0.144758 | 0.1634 |

Within the B-spline representation, ttRTC changes full-test MSE by **−3.91%** for flow matching, **+39.03%** for layerwise discrete, and **−11.83%** for joint discrete, where a negative change is an improvement. Thus ttRTC helps flow matching and joint discrete, but materially hurts the reproduced layerwise model.

## Raw versus B-spline

The following uses decoded physical action MSE, so both representations are compared in the same output coordinates.

| Architecture | Stage | Raw MSE ↓ | B-spline MSE ↓ | B-spline change vs raw |
|---|---:|---:|---:|---:|
| Flow matching | Basic | **0.013315** | 0.013835 | 3.91% worse |
| Flow matching | ttRTC | **0.012975** | 0.013294 | 2.45% worse |
| Layerwise discrete | Basic | 0.058202 | **0.032380** | 44.37% better |
| Layerwise discrete | ttRTC | **0.044474** | 0.045018 | 1.22% worse |
| Joint discrete | Basic | 0.044148 | **0.023766** | 46.17% better |
| Joint discrete | ttRTC | 0.033632 | **0.020955** | 37.69% better |

Main finding: B-splines strongly improve the joint-discrete model and the basic layerwise model, while raw actions are slightly better for flow matching. The layerwise B-spline ttRTC result is the clear regression and should be diagnosed before treating ttRTC as generally beneficial.

## Delay, replay, and latency

The B-spline delay sweep uses the same 128 held-out samples for `d=0…10`. Even delays align with trained spline spans; odd delays are deliberately unseen within-span cases. Committed-prefix preservation error is zero throughout.

| Architecture | Basic mean delay MSE | ttRTC mean delay MSE | ttRTC effect |
|---|---:|---:|---:|
| Flow matching | 0.008309 | 0.008310 | 0.01% worse |
| Layerwise discrete | 0.026190 | 0.038691 | 47.74% worse |
| Joint discrete | 0.031480 | 0.019862 | 36.91% better |

All three B-spline ttRTC replays issued 120/120 commands and made 20 plan switches at `D=3` spans. These are dataset replays, not closed-loop robot success evaluations.

Default batch-one sampling p50 is 15.33/15.28 ms for basic/ttRTC flow matching, 12.58/11.87 ms for layerwise discrete, and 84.90/85.43 ms for cached joint discrete. The 126-scalar B-spline joint representation is about twice as fast as the 210-scalar raw joint representation at the default sampler setting.

## Verification and artifacts

- Project tests: **14 passed**.
- Independent B-spline checkpoint reload: **6/6 passed**.
- Exact same-seed basic reproduction: **3/3 passed**.
- W&B state: all six production, three rerun, and one comparison run verified as `finished`.
- B-spline fit error: normalized MAE `0.001384`, RMSE `0.008136`.
- Additional 256-bin quantization error: normalized MAE `0.001716`, RMSE `0.002247`.

Primary local artifacts:

- Configuration: `robot_policy/configs/bspline_reproduction.yaml`
- Checkpoints: `robot_policy/outputs/bspline_reproduction/checkpoints/`
- Evaluation: `robot_policy/outputs/bspline_reproduction/evaluation/`
- Reproducibility: `robot_policy/outputs/bspline_reproduction/reproducibility/`
- Visuals: `robot_policy/outputs/bspline_reproduction/visualizations/`
- Comparison JSON/CSV/figures: `robot_policy/outputs/action_representation_comparison/`

Earlier exploratory B-spline W&B runs with superseded metric semantics remain in project history. Only the production IDs listed above and the checkpoint manifest are authoritative.

## Next steps

1. Run at least three distinct deterministic seeds per representation to measure variance rather than only same-seed repeatability.
2. Diagnose the layerwise B-spline ttRTC regression with per-delay, per-horizon, joint-only, and gripper-only MSE breakdowns.
3. Add paired per-window confidence intervals and significance tests for raw versus B-spline comparisons.
4. Add closed-loop simulation or hardware evaluation only after runtime integration, calibration, and safety authorization are supplied.
