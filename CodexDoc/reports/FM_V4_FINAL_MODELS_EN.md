# V4 Flow-Matching Final Local Models

Date: 2026-09-22 UTC

All training processes are stopped and all GPUs are free. W&B is metrics-only;
model artifact upload is disabled. The authoritative model for every variant
is its local `base.pt`, whose inference `model` field contains the best
observed validation EMA weights.

| Dataset | Variant | Selected point | Validation action MSE | Checkpoint type |
|---|---|---:|---:|---|
| stacking_cup_30hz | raw | epoch 41 | 0.0229889 | best model + epoch-100 resume state |
| stacking_cup_30hz | B-spline | epoch 50 | 0.0214426 | best model + epoch-100 resume state |
| classify_blocks_30hz | raw | epoch 64 | 0.0311488 | inference-only best model |
| classify_blocks_30hz | B-spline | epoch 16 | 0.0304322 | inference-only best model |
| hanging_mug_30hz | raw | update 33,360 | 0.0150083 | best model + epoch-100 resume state |
| hanging_mug_30hz | B-spline | update 41,283 | 0.0152258 | best model + epoch-100 resume state |

All six checkpoints pass clean policy loading. Both classify checkpoints also
pass the deployment inspection/load path and are explicitly marked
`training_resume_available=false`; this does not affect inference or
evaluation.

Detailed machine-readable stop records are stored beside the outputs as
`EARLY_STOP_VALIDATION_BEST.json`.
