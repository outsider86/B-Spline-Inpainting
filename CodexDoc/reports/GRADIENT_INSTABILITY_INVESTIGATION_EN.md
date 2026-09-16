# Gradient-instability investigation

**Date:** 2026-09-16  
**Scope:** DiT-S/B/L 36-checkpoint capacity sweep

## Finding

The observed problem is a **layerwise-policy gradient explosion followed by FP32 norm-reduction overflow**. It is not, in the completed sweep, a NaN-loss crash.

Three policy variants are affected, covering six base/ttRTC runs:

| Size | Representation | Stage | First logged infinite norm | Infinite logged samples | Final validation `action_mse` |
|---|---|---:|---:|---:|---:|
| DiT-B | B-spline | Base | 6,510 | 3,952 | 0.1916951 |
| DiT-B | B-spline | ttRTC | 1 | 501 | 0.1748070 |
| DiT-B | Raw | Base | 13,100 | 3,688 | 0.5175689 |
| DiT-B | Raw | ttRTC | 1 | 501 | 0.5157025 |
| DiT-L | B-spline | Base | 6,360 | 4,366 | 0.1823873 |
| DiT-L | B-spline | ttRTC | 1 | 501 | 0.1752656 |

The ttRTC children inherit an already unstable parent and therefore report an infinite gradient norm from update 1. No FM run, joint-DD run, DiT-S layerwise run, or raw DiT-L layerwise run showed a non-finite norm. No run logged a non-finite loss or `action_mse`.

## Direct reproduction

A real batch-32 backward pass from the affected DiT-B B-spline layerwise base produced:

- finite loss: `94.2550`;
- **0 non-finite gradient elements**;
- maximum finite gradient element: `1.033e21`;
- robust FP64 global gradient norm: `1.021e23`;
- 28 parameter tensors whose individual FP32 norm reduction overflowed;
- PyTorch `clip_grad_norm_` result: `Infinity`;
- nonzero gradient elements before clipping: `103,258,569`;
- nonzero gradient elements after clipping: **0**.

Therefore, the gradients are genuinely enormous but still elementwise finite. Squaring them during an FP32 L2-norm reduction exceeds the FP32 range. The clip coefficient becomes zero, erasing all new gradient signal. AdamW can then only apply residual optimizer-state evolution and decoupled weight decay.

## Root cause

The failure originates in the layerwise DiT conditioning path:

`constant t=0` → trainable time embedding drifts → unbounded AdaNorm scale/shift → residual activation amplification → enormous finite gradients → FP32 norm overflow → zeroed gradients

The relevant implementation details are:

- `LayerwiseDiscretePolicy.logits` always sends a zero timestep to the backbone, even though corruption severity varies per batch.
- One shared trainable time embedding is reused in every block.
- AdaNorm modulation is default-initialized, unbounded, and has no adaLN-Zero residual gates.
- Final LayerNorm keeps logits and forward losses finite, hiding the exploding internal Jacobian.
- Training uses the default FP32 norm reduction in `clip_grad_norm_` and does not fail fast on non-finite norms.

Measured activation evidence:

| Quantity | Initial DiT-B B-spline | Affected final DiT-B B-spline | Stable final DiT-S B-spline |
|---|---:|---:|---:|
| Maximum time embedding | 0.365 | 794.836 | 1.307 |
| Maximum AdaNorm scale | 0.145 | 3,610.613 | 1.277 |
| Maximum block output | — | about 959,000 | about 147 |

The largest gradients occur in the time MLP, AdaNorm modulation, and early attention blocks, matching this mechanism. Ordinary parameter maxima remain near 5, so this is activation/Jacobian amplification rather than a single parameter becoming `Inf`.

Raw DiT-L remaining finite shows that this is a threshold reached along some optimization trajectories, not a deterministic consequence of size alone. Greater width/depth makes the unsafe pathway more vulnerable; initialization, corruption samples, representation, and optimization trajectory determine whether it crosses the threshold.

## Did the sweep crash?

No numerical crash occurred in the final 36 runs: all finished, saved reloadable checkpoints, and passed the W&B audit. Every traceback retained in the training logs ends in `KeyboardInterrupt` and came from the earlier controlled scheduler migration. There is no OOM, NCCL failure, NaN loss, or numerical `RuntimeError` in the completed logs.

The practical failure is still serious: affected runs silently lose new gradient updates after norm overflow. A larger or less stable variant can progress from huge finite gradients to actual `Inf`/`NaN` elements, at which point optimizer state and parameters can be contaminated and a true crash or NaN run can follow.

## Recommended correction order

1. **Fix the architecture:** pass the actual corruption/noise level instead of constant zero, and replace the current modulation with adaLN-Zero-style zero-initialized modulation and learned residual gates. Consider normalizing or bounding the time embedding as an additional guard.
2. **Make clipping numerically robust:** compute per-tensor and global L2 norms in FP64 (or with a scaled sum-of-squares algorithm), then apply the finite clip coefficient to FP32 gradients.
3. **Fail fast and preserve evidence:** explicitly check every gradient tensor for finite values; record robust norm, max absolute gradient, clip coefficient, time-embedding magnitude, and AdaNorm-scale magnitude. Save a diagnostic checkpoint before aborting on true non-finite values.
4. **Retune only after the structural fix:** a lower learning rate, longer warmup, or smaller clip threshold may improve margin, but none fixes constant unbounded conditioning or FP32 norm overflow by itself.
5. **Retrain the affected variants:** the six affected checkpoints remain valid reproduction artifacts, but they are not trustworthy converged capacity comparisons. Retrain the three affected base/ttRTC pairs after the fix and compare `action_mse` against these recorded runs.

## Suggested acceptance test

Run DiT-B raw/B-spline and DiT-L B-spline layerwise canaries through and beyond the original onset points (at least 15,000 base updates), requiring:

- finite gradient elements and finite robust global norm at every update;
- a strictly positive finite clip coefficient;
- bounded time/AdaNorm telemetry without sustained growth;
- no NaN/Inf loss or parameter values;
- continued improvement of validation `action_mse` after updates 6,500 and 13,100.

