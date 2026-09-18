# DiT-S Full-Vision Results (8 Checkpoints)

## Status

The original DiT-S rerun is complete and independently audited. All eight
checkpoints use two cameras with a complete 16×16 token grid per camera
(256×2 = 512 vision tokens). Base policies were trained for 50,000 updates and
their exact-checkpoint parents were cached before 5,000-update ttRTC training.

`validation/action_mse` is generation from scratch: the policy receives only
vision and robot state and generates from noise (FM) or all-MASK tokens (joint
discrete diffusion). Target actions are used only after generation for scoring.

| representation | policy | stage | final validation MSE | best validation MSE | test physical MSE | sampling p50 (ms) | W&B |
|---|---|---:|---:|---:|---:|---:|---|
| raw | FM | base | 0.007611 | 0.005651 | 0.010827 | 25.61 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/70newtq7) |
| raw | FM | ttRTC | 0.008254 | 0.006683 | 0.008333 | 15.39 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zt8fra2l) |
| raw | joint DD | base | 0.026301 | 0.013556 | 0.023261 | 150.83 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/x1vtxm13) |
| raw | joint DD | ttRTC | 0.025776 | 0.020801 | 0.022973 | 156.25 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/6cyontjw) |
| B-spline | FM | base | 0.008366 | 0.005903 | 0.008506 | 24.01 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/k2plilzb) |
| B-spline | FM | ttRTC | 0.008092 | 0.007014 | 0.008409 | 24.31 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zxp7uffq) |
| B-spline | joint DD | base | 0.024129 | 0.017170 | 0.019029 | 80.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/ivclz1i6) |
| B-spline | joint DD | ttRTC | 0.025541 | 0.020515 | 0.019057 | 84.87 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/efg4vkqh) |

## Verification

- Exact inventory: 8/8 atomic checkpoints.
- Evaluation per checkpoint: full 3,149-chunk test open-loop generation,
  batch-1 latency, 128-sample delay RTC, and 64-sample oracle-prefix inference
  RTCEVAL at prefixes `{2,4,6,8,10}`.
- Real deployment validation: 8/8 policies load and produce finite `[1,30,7]`
  actions; vision encoder output is finite and uses 512 tokens.
- Local and remote W&B completion audits pass with no errors.
- Test suite: 44/44 passing.

The broader DiT-B/DiT-L replication remains active and will produce a separate
24-checkpoint aggregate comparison.
