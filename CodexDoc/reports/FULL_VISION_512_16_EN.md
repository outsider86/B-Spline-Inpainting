# Full-Vision FM and Joint-DD Results

## Protocol

- 16 checkpoints: DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC.
- Every observation contains two complete 16×16 patch grids: **512 vision tokens**, plus one state token.
- Base training uses 50,000 updates; ttRTC uses 5,000 updates; effective batch size is 32.
- `validation/action_mse` is decoded generation from scratch. The sampler receives only vision and state; it never receives target actions, corrupted ground truth, or teacher assistance.
- Evaluation includes full held-out open-loop generation, delay RTC, oracle-prefix inference RTC, and synchronized batch-1 latency.
- Within each DiT size, raw and B-spline RTCEVAL metric plots share one log physical-MSE y-axis; all trajectory plots share the same full-dataset physical min/max for each action dimension.

## Findings

- Best decoded physical MSE: **0.008187** (DiT-B / raw / fm / ttrtc).
- B-spline wins 5/8 matched comparisons; median physical-MSE reduction is 13.33%.
- ttRTC wins 6/8 matching zero-delay open-loop comparisons; median reduction is 0.55%.
- Joint cached/fused decoding is token-identical in 8/8 checkpoints.
- FM rejected 5625 anomalous optimizer updates after the 1,000-update warm-up; every rejected step remains recorded with its pre-clip norm.
- Deployment validation passes all 8 runtime cases for each of DiT-S and DiT-B; each audit inventories all 16 checkpoints and produces finite 30×7 actions from two 256-token camera streams.

## Evidence

- `completion_audit.json`: checkpoint identity, exact update counts, hashes, parent lineage, finite gradients, from-scratch validation, all four evaluation families, and local/remote W&B state.
- `summary/deployment_validation_dit_{s,b}.json`: real-GPU load and inference for every raw/B-spline × FM/joint-DD × base/ttRTC case.
- `summary/checkpoint_metrics.csv`: complete 16-row metric table; `ranking_by_physical_mse.csv` and `pairwise_effects.csv`: rankings and matched effects.
- Results are open-loop action-generation diagnostics, not closed-loop task-success measurements.

## All checkpoints

| size | repr. | policy | stage | val gen MSE | test physical MSE | p50 ms | W&B |
|---|---|---|---|---|---|---|---|
| DiT-S | raw | fm | base | 0.007611 | 0.010827 | 25.61 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/70newtq7) |
| DiT-S | raw | fm | ttrtc | 0.008254 | 0.008333 | 15.39 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zt8fra2l) |
| DiT-S | raw | discrete_joint | base | 0.026301 | 0.023261 | 150.83 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/x1vtxm13) |
| DiT-S | raw | discrete_joint | ttrtc | 0.025776 | 0.022973 | 156.25 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/6cyontjw) |
| DiT-S | bspline | fm | base | 0.008366 | 0.008506 | 24.01 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/k2plilzb) |
| DiT-S | bspline | fm | ttrtc | 0.008092 | 0.008409 | 24.31 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zxp7uffq) |
| DiT-S | bspline | discrete_joint | base | 0.024129 | 0.019029 | 80.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/ivclz1i6) |
| DiT-S | bspline | discrete_joint | ttrtc | 0.025541 | 0.019057 | 84.87 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/efg4vkqh) |
| DiT-B | raw | fm | base | 0.006808 | 0.008203 | 68.15 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/mz36ylrg) |
| DiT-B | raw | fm | ttrtc | 0.007001 | 0.008187 | 68.32 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zip5w0in) |
| DiT-B | raw | discrete_joint | base | 0.015395 | 0.019526 | 298.37 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/c7mgy7f0) |
| DiT-B | raw | discrete_joint | ttrtc | 0.017608 | 0.020072 | 295.84 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/26qllqsn) |
| DiT-B | bspline | fm | base | 0.007076 | 0.008330 | 37.31 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/f581nhyj) |
| DiT-B | bspline | fm | ttrtc | 0.006578 | 0.008320 | 37.38 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/wo8veppb) |
| DiT-B | bspline | discrete_joint | base | 0.025449 | 0.017233 | 167.97 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/lwoiqslq) |
| DiT-B | bspline | discrete_joint | ttrtc | 0.026583 | 0.017078 | 163.03 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/680zgije) |
