# Open-loop and RTC evaluation

All six checkpoints use the identical 3,149-window held-out test set. These are predictions under fixed recorded observations; they do not measure environmental feedback, simulator performance, or physical robot success.

| Checkpoint | Decoded physical MAE | RMSE | Max abs | Token accuracy | First-sample MAE |
|---|---:|---:|---:|---:|---:|
| FM base | 0.05162 | 0.11083 | 1.2286 | n/a | 0.02087 |
| FM RTC | 0.05642 | 0.11751 | 1.2517 | n/a | 0.03903 |
| layerwise base | 0.10179 | 0.18512 | 1.6747 | 0.1271 | 0.08548 |
| layerwise RTC | 0.10239 | 0.18721 | 1.6428 | 0.1201 | 0.15054 |
| joint base | 0.07076 | 0.14816 | 1.3477 | 0.1735 | 0.03086 |
| joint RTC | 0.06967 | 0.13918 | 1.4049 | 0.1606 | 0.05405 |

Each JSON under `outputs/evaluation/` additionally reports encoded-space error, per-dimension MAE/RMSE/max/quantiles, six joint channels separately from gripper, spline-fit and quantization baselines, and velocity/acceleration quantiles. The dataset exposes absolute joint coordinates and gripper only; Cartesian position/rotation metrics would be fabricated without kinematics/calibration and are therefore marked unavailable.

## Delay sweeps

Delay reports use the same first 128 eligible test windows at every d=s from 0 through 10 raw 30 Hz actions. Even delays 2/4/6/8/10 are the exact-span training support; odd delays are deliberately unseen within-span evaluations. The mapping uses `ceil(d/2)+3` cubic support controls. Maximum committed-interval error is exactly zero in all 66 checkpoint/delay combinations relative to the representation actually fixed; discrete reports retain the small separate re-quantization error to the continuous shifted prior.

| Checkpoint | d=0 MAE | d=10 MAE | Mean seen-even MAE | Mean unseen-odd MAE |
|---|---:|---:|---:|---:|
| FM base | 0.04397 | 0.04892 | 0.04654 | 0.04667 |
| FM RTC | 0.04650 | 0.04860 | 0.04704 | 0.04725 |
| layerwise base | 0.09398 | 0.09533 | 0.09240 | 0.09078 |
| layerwise RTC | 0.09044 | 0.08807 | 0.08777 | 0.08798 |
| joint base | 0.05543 | 0.06706 | 0.06009 | 0.05935 |
| joint RTC | 0.05602 | 0.06034 | 0.05589 | 0.05596 |

RTC fine-tuning improves delayed joint and layerwise predictions, while FM is essentially flat/slightly worse; this mixed outcome is retained rather than summarized as a universal gain. Dataset replay completes 120/120 commands with no empty queue for each architecture at D=S=3, plus an odd d=s=3 joint replay. Plan-switch timing and commands are in `outputs/replay/`.
