# BSP U-Net V4 Flow-Matching Status

Date: 2026-09-22 UTC

## Outcome

The replacement batch-4 training on GPUs 2/3 was stopped after confirming the
previous discrepancy came from partial training-time validation coverage. The
authoritative evaluation pipeline was then run directly on the existing
completed batch-4 and batch-64 raw/B-spline checkpoints.

Every metric below covers the complete split: 28,557 train windows and 3,149
validation windows. Open-loop is generation from scratch. RTC uses the current
GT batch's first six actions as the committed prefix. Raw FM is conditioned on
the first six GT action rows; B-spline FM is conditioned on the six GT control
rows supporting the three affected spans (`3 spans + cubic degree 3`). Base FM
uses binary-hard-mask PiGDM, and RTC MSE is computed only over the generated
suffix.

| Batch | Representation | Split | Open-loop physical MSE | GT-prefix RTC suffix physical MSE |
|---:|---|---|---:|---:|
| 4 | raw | train | 0.00606603 | 0.00586514 |
| 4 | raw | val | 0.01132877 | 0.01151038 |
| 4 | B-spline | train | 0.00629346 | 0.00550323 |
| 4 | B-spline | val | 0.01130805 | 0.01028329 |
| 64 | raw | train | 0.00036776 | 0.00031687 |
| 64 | raw | val | 0.00945820 | 0.01051469 |
| 64 | B-spline | train | 0.00042084 | 0.00035019 |
| 64 | B-spline | val | 0.00936654 | 0.01002486 |

Batch 64 fits training data much more aggressively: its validation/train
open-loop MSE ratio is 25.72× for raw and 22.26× for B-spline, versus 1.87× and
1.80× for batch 4. Despite that larger generalization gap, batch 64 still has
lower absolute validation MSE: 16.51% lower for raw and 17.17% lower for
B-spline open-loop; 8.65% and 2.51% lower for GT-prefix RTC.

B-spline has almost no open-loop advantage over raw at the same batch size
(0.18% at batch 4 and 0.97% at batch 64), but helps RTC suffix generation:
10.66% lower validation MSE at batch 4 and 4.66% lower at batch 64.

## Visualization contract and audit

For each batch size, one deterministic motion-rich GT batch was sampled from
train and one from validation. Every subplot contains exactly ground truth,
from-scratch prediction, and GT-prefix RTC prediction. The blue RTC trajectory
is the exact committed GT prefix concatenated with the PiGDM-generated suffix;
all four NPZ artifacts pass bitwise prefix equality for raw and B-spline, and
all reported committed-prefix MSE values are exactly zero. Axis limits are the
same full-dataset physical min/max for both representations.

Artifacts:

- `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch4/`
- `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch64/`

Each folder contains `trajectory_comparison_{train,val}.png`, the source NPZ
and JSON files, complete per-checkpoint metric JSONs, `metrics.csv`, and a
bilingual summary.

## Next step

Use batch 64 as the stronger existing validation checkpoint, while treating
its large train/validation gap as explicit evidence of overfitting. For RTC,
keep B-spline as the preferred representation under this prefix-6 protocol.
