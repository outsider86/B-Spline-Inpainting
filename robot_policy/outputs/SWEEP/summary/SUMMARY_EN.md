# Comprehensive 36-Checkpoint Sweep Summary

Generated from fresh evaluation artifacts in this directory on 2026-09-16.

## Evaluation contract

- All 36 checkpoints were evaluated on the same 3,149 held-out windows (five test episodes), using recorded observations and seed 20260915. These are open-loop action errors, not task-success rates.
- Primary quality metrics are decoded 30×7 trajectories: normalized MSE for representation-fair comparison and physical joint/gripper MSE after training-split q01/q99 scaling.
- Default sampling is FM 12 steps, layerwise diffusion 8 rounds, and joint diffusion 8 rounds with fused inter-block K/V transitions.
- Latency uses batch 1 on RTX PRO 6000 Blackwell: 10 warmups and 100 synchronized trials. `estimated online p50` is the sum of separately measured online vision, observation projector, policy sampling, and action decode medians; it is not a directly timed end-to-end percentile.
- Delay curves are included for all checkpoints.

## Main findings

- Best physical MSE: **0.009896**, DiT-L bspline fm base.
- B-spline beats its matched raw checkpoint in **10/18** pairs; median physical-MSE reduction is **3.94%** (negative means worse).
- Across d=0…10, B-spline has lower mean delayed MSE in **9/18** matched pairs; the median reduction is **-2.33%**.
- ttRTC improves zero-delay physical MSE in **4/18** pairs and mean d=0…10 MSE in **4/18** pairs. Median delayed-MSE reduction is **-6.11%**, so this fixed fine-tuning protocol does not reliably improve delay robustness across capacities.
- The D2F-aligned fused cache transition is exactly token-equivalent to the prior cached sampler in **12/12** real checkpoints. Median p50 speedup is **8.65%** (range 5.29% to 12.48%).
- The known DiT-B raw layerwise instability is visible in open-loop error; capacity scaling is therefore not monotonic across every policy family.

## B-spline decode proof

- Every B-spline checkpoint uses the same manifest-matched `UniformLeftBSplineConfig`, implementation `uniform_left_direct_fit_v1`, tokenizer `3282a6009a52b9a0`, and calibration.
- Decode order is token dequantization (discrete only) → fixed **30×18** basis × **18×7** controls → **30×7** normalized actions → physical scaling.
- Across every held-out window, recomputed continuous/stored decode max error is **9.934e-08**; recomputed token/stored decode max error is **5.959e-08**.
- The basis is rank 18; maximum partition-of-unity error is 1.110e-16. The basis figure uses a dense 1,201-point SciPy cubic evaluation, with the exact 30 decoder samples overlaid as dots; those integer samples match the decoder matrix to 5.551e-17. Full evidence is in `decoder_validation/decode_validation.json`.

## Top 10 by decoded physical MSE

| rank | size | repr. | policy | stage | physical MSE | sample p50 ms |
|---|---|---|---|---|---|---|
| 1 | DiT-L | bspline | fm | base | 0.009896 | 53.85 |
| 2 | DiT-L | raw | fm | base | 0.010727 | 55.99 |
| 3 | DiT-B | raw | fm | base | 0.010828 | 28.84 |
| 4 | DiT-S | bspline | fm | base | 0.010893 | 15.04 |
| 5 | DiT-L | raw | fm | ttrtc | 0.010904 | 55.97 |
| 6 | DiT-B | raw | fm | ttrtc | 0.011002 | 28.80 |
| 7 | DiT-S | raw | fm | base | 0.011016 | 15.66 |
| 8 | DiT-S | raw | fm | ttrtc | 0.011148 | 16.06 |
| 9 | DiT-L | bspline | fm | ttrtc | 0.011148 | 53.77 |
| 10 | DiT-B | bspline | fm | base | 0.011326 | 30.19 |

## All checkpoints

| size | repr. | policy | stage | M params | val action MSE | test norm. MSE | test physical MSE | sample p50 ms | estimated online p50 ms | W&B |
|---|---|---|---|---|---|---|---|---|---|---|
| DiT-S | raw | fm | base | 15.07 | 0.02363 | 0.03678 | 0.01102 | 15.66 | 26.49 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9p5hb9y) |
| DiT-S | raw | fm | ttrtc | 15.07 | 0.02645 | 0.03708 | 0.01115 | 16.06 | 26.90 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/v9ubgo5v) |
| DiT-S | raw | layerwise | base | 15.04 | 0.00556 | 0.04564 | 0.01411 | 12.90 | 23.73 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/zh84gs3w) |
| DiT-S | raw | layerwise | ttrtc | 15.04 | 0.00624 | 0.05046 | 0.01606 | 12.31 | 23.13 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/2o2kthwz) |
| DiT-S | raw | joint | base | 13.85 | 0.01256 | 0.06101 | 0.01849 | 146.18 | 157.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ykwmgupy) |
| DiT-S | raw | joint | ttrtc | 13.85 | 0.00699 | 0.06588 | 0.01984 | 144.53 | 155.37 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/dgxz1ipu) |
| DiT-S | B-spline | fm | base | 15.07 | 0.03394 | 0.03622 | 0.01089 | 15.04 | 25.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/2rb15m2t) |
| DiT-S | B-spline | fm | ttrtc | 15.07 | 0.03566 | 0.03806 | 0.01158 | 15.63 | 26.49 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/83pdq57s) |
| DiT-S | B-spline | layerwise | base | 15.00 | 0.00291 | 0.08652 | 0.02938 | 12.01 | 22.86 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ya0xdf4n) |
| DiT-S | B-spline | layerwise | ttrtc | 15.00 | 0.00377 | 0.09377 | 0.03349 | 12.55 | 23.39 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/q28aplkd) |
| DiT-S | B-spline | joint | base | 13.82 | 0.01093 | 0.05121 | 0.01580 | 78.48 | 89.32 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0q4g7cq4) |
| DiT-S | B-spline | joint | ttrtc | 13.82 | 0.01367 | 0.06052 | 0.01850 | 79.84 | 90.71 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/340xpp9a) |
| DiT-B | raw | fm | base | 108.06 | 0.02567 | 0.03672 | 0.01083 | 28.84 | 39.66 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/op6lpqi3) |
| DiT-B | raw | fm | ttrtc | 108.06 | 0.02561 | 0.03663 | 0.01100 | 28.80 | 39.64 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/duznrj8t) |
| DiT-B | raw | layerwise | base | 107.39 | 0.51757 | 0.49172 | 0.21971 | 28.46 | 39.30 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o2g40xq8) |
| DiT-B | raw | layerwise | ttrtc | 107.39 | 0.51570 | 0.49172 | 0.21970 | 28.53 | 39.40 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ajt8p9vs) |
| DiT-B | raw | joint | base | 102.68 | 0.03297 | 0.14679 | 0.07543 | 237.33 | 248.52 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/x9qrzmwd) |
| DiT-B | raw | joint | ttrtc | 102.68 | 0.03654 | 0.19250 | 0.06061 | 233.78 | 245.09 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/1hux9mwd) |
| DiT-B | B-spline | fm | base | 108.05 | 0.02710 | 0.03845 | 0.01133 | 30.19 | 41.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0c77dslj) |
| DiT-B | B-spline | fm | ttrtc | 108.05 | 0.02993 | 0.03931 | 0.01207 | 30.19 | 41.01 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/25fzgt8t) |
| DiT-B | B-spline | layerwise | base | 107.33 | 0.19170 | 0.42273 | 0.19773 | 25.21 | 36.03 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i40wvlpn) |
| DiT-B | B-spline | layerwise | ttrtc | 107.33 | 0.17481 | 0.42273 | 0.19773 | 24.77 | 35.57 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/wjpva0fp) |
| DiT-B | B-spline | joint | base | 102.62 | 0.01815 | 0.05071 | 0.01840 | 131.14 | 141.97 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tsg4colk) |
| DiT-B | B-spline | joint | ttrtc | 102.62 | 0.02253 | 0.04998 | 0.01695 | 131.74 | 142.60 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/jjch9idz) |
| DiT-L | raw | fm | base | 367.58 | 0.02927 | 0.03586 | 0.01073 | 55.99 | 66.83 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tziiazri) |
| DiT-L | raw | fm | ttrtc | 367.58 | 0.02781 | 0.03612 | 0.01090 | 55.97 | 66.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/redv2qv4) |
| DiT-L | raw | layerwise | base | 366.17 | 0.04226 | 0.04782 | 0.01460 | 64.23 | 75.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o0tj6klu) |
| DiT-L | raw | layerwise | ttrtc | 366.17 | 0.03814 | 0.04828 | 0.01446 | 63.80 | 74.62 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/7ejattim) |
| DiT-L | raw | joint | base | 357.84 | 0.05200 | 0.07649 | 0.02594 | 532.13 | 543.44 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/kvby9apd) |
| DiT-L | raw | joint | ttrtc | 357.84 | 0.05068 | 0.10409 | 0.03393 | 533.41 | 544.78 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/0i7ac2b6) |
| DiT-L | B-spline | fm | base | 367.57 | 0.02596 | 0.03251 | 0.00990 | 53.85 | 64.74 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/hso7oc5a) |
| DiT-L | B-spline | fm | ttrtc | 367.57 | 0.02935 | 0.03538 | 0.01115 | 53.77 | 64.65 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/s56r3udm) |
| DiT-L | B-spline | layerwise | base | 366.08 | 0.18239 | 0.36524 | 0.17671 | 46.81 | 57.59 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9zgz3hh) |
| DiT-L | B-spline | layerwise | ttrtc | 366.08 | 0.17527 | 0.36524 | 0.17671 | 46.83 | 57.66 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/pgnhmdfl) |
| DiT-L | B-spline | joint | base | 357.76 | 0.01948 | 0.06627 | 0.02239 | 309.14 | 320.50 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/us6r4jrf) |
| DiT-L | B-spline | joint | ttrtc | 357.76 | 0.02368 | 0.06986 | 0.02332 | 310.01 | 321.11 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ra5ijsh8) |

## Artifacts

- `checkpoint_metrics.csv`: one row per checkpoint with training, test, latency, memory, smoothness, decoder, and W&B fields.
- `pairwise_effects.csv`: matched B-spline/raw and ttRTC/base effects.
- `summary.json`: complete machine-readable aggregation.
- `performance_vs_latency.png`, `physical_mse_matrix.png`, `joint_cache_acceleration.png`: comparison figures.
- `decoder_validation/`: numerical decode audit and inspected basis/trajectory figures.
- `open_loop/`, `latency/`, `latency_baseline/`, and `rtc/` (when present): raw per-checkpoint reports.

## Limitations

The dataset exposes absolute joints and gripper but no calibrated Cartesian pose, so position/rotation error cannot be reported honestly. Velocity/acceleration statistics describe predicted open-loop trajectory smoothness, not executed robot dynamics. No simulator or physical robot success claim is made.
