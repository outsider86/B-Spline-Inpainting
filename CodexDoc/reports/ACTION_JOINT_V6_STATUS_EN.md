# ActionJoint V6 Final Status

## Scope and protocol

V6 contains three tasks (`hanging_mug`, `stacking_cup`, and
`classify_blocks`), three observation-state variants, and raw/B-spline action
representations. All 18 base policies use BSP-UNet flow matching, predict the
actual 7D joint angle trajectory, and consume exactly two same-timestep camera
images (`observation_horizon=1`). The state dimensions are 7 for LastCommand
and State, and 14 for State+LastCommand.

Each base model trained for 100 epochs and ran full generation-from-scratch
validation at epochs 10, 20, ..., 100. Its validation winner initialized a
fresh-optimizer five-epoch training-time RTC stage, validated in full after
every epoch. There is no test split. One online W&B project is used per task,
and no W&B model artifacts were created.

## Validation-best results

Values are generation-from-scratch action MSE over the complete validation
split. `e` denotes the selected epoch.

| Task | Observation state | Representation | Base | ttRTC |
|---|---|---:|---:|---:|
| hanging_mug | LastCommand | raw | e100 / 0.0115253636 | e5 / 0.0115644013 |
| hanging_mug | LastCommand | B-spline | e90 / 0.0117124693 | e5 / 0.0116665477 |
| hanging_mug | State | raw | e90 / 0.0125336456 | e5 / 0.0126129311 |
| hanging_mug | State | B-spline | e80 / 0.0126252889 | e5 / 0.0124728527 |
| hanging_mug | State+LastCommand | raw | e70 / 0.0116072344 | e5 / 0.0115960392 |
| hanging_mug | State+LastCommand | B-spline | e100 / 0.0114557928 | e1 / 0.0115313689 |
| stacking_cup | LastCommand | raw | e70 / 0.0187636974 | e4 / 0.0185712790 |
| stacking_cup | LastCommand | B-spline | e100 / 0.0175009139 | e5 / 0.0175499370 |
| stacking_cup | State | raw | e100 / 0.0189064756 | e5 / 0.0190488806 |
| stacking_cup | State | B-spline | e70 / 0.0194656161 | e1 / 0.0199390450 |
| stacking_cup | State+LastCommand | raw | e60 / 0.0186401576 | e5 / 0.0187613057 |
| stacking_cup | State+LastCommand | B-spline | e100 / 0.0176345984 | e5 / 0.0176392303 |
| classify_blocks | LastCommand | raw | e90 / 0.0233113019 | e5 / 0.0230967354 |
| classify_blocks | LastCommand | B-spline | e70 / 0.0225643763 | e5 / 0.0224542118 |
| classify_blocks | State | raw | e70 / 0.0255008774 | e5 / 0.0255440672 |
| classify_blocks | State | B-spline | e80 / 0.0250842212 | e5 / 0.0251315308 |
| classify_blocks | State+LastCommand | raw | e60 / 0.0235361312 | e5 / 0.0235573525 |
| classify_blocks | State+LastCommand | B-spline | e20 / 0.0229856817 | e4 / 0.0226660629 |

## Final audit and publication

- Final local inventory: 18 `base.pt` and 18 `ttrtc.pt` files. No resume,
  periodic, epoch/step snapshot, or rolling-best weight file remains.
- Independent CPU load and contract audit: 36/36 passed. It verified the 7D
  joint output, 7D/14D state input, h1 two-camera input, cubic uniform-left
  B-spline metadata, 100+5 epoch budgets, complete validation histories,
  ttRTC parent linkage, and task-specific online W&B projects.
- Complete validation sizes are 2,722 samples for hanging, 4,211 for stacking,
  and 9,467 for classify. Every optimizer completed with zero skipped steps.
- `DiscreteRTC/dRTC/NewModel/V6` contains 36 models and 72 server/training JSON
  files (108 total). No remote file is missing, and all 36 remote model LFS
  SHA-256 values match their local files.
- Final Hugging Face dataset revision:
  `08a54d5c34cda652cf29f8451afd94ec0228c5a6`.

The V6 ActionJoint training and publication goal is complete. Deployment,
open-loop, and GT-prefix RTC evaluation are separate follow-up work.
