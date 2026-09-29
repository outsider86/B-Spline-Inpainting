# Absolute-Action V8 Completion Report

## Outcome

V8 is complete. Its scope is `classify_blocks` only, with eight final models:

`{StateAbsJoint, StateAbsEE} × {raw, B-spline} × {base, ttRTC}`.

Every model is BSP-UNet flow matching with independent scratch ResNet-18
global/hand encoders and SpatialSoftmax. Each example uses one observation
timestep (h1), containing exactly the current global and hand images. Training
used GPUs 0--3 and ran from 2026-09-29 04:52 to 08:02 UTC.

## Contracts and data

| Variant | Observation state | Predicted action |
|---|---|---|
| StateAbsJoint | 7D measured joints (6) + gripper (1) | 7D next-step absolute measured joints (6) + gripper (1) |
| StateAbsEE | The same 7D measured-joint state | 8D next-step absolute TCP xyz (3) + quaternion xyzw (4) + gripper (1) |

V8 makes action width explicit in config/checkpoint metadata throughout data
preparation, raw/B-spline codecs, policy heads, metrics, RTC hard masks,
sidecar validation, and deployment. Legacy checkpoints still default to 7D.

Each source tree has 85 episodes, 185,913 frames, and 170 MP4 files. The
corresponding videos match byte-for-byte by SHA-256 and their timestamp,
frame, and episode arrays are identical. V8 therefore shares one audited
RGB84 cache while isolating action caches by state/action variant and
representation. The deterministic seed-20260915 split is 76 training episodes
(166,875 windows), 9 validation episodes (19,038 windows), and no test split.

## Protocol and selected checkpoints

- Batch/effective batch size: 64/64.
- Base: 100 epochs and 260,700 updates, with complete validation every 10
  epochs.
- ttRTC: initialized from the matching validation-best base, then trained for
  5 epochs and 13,035 updates, with complete validation every epoch.
- Raw and B-spline both use reference hard-mask ttRTC. B-spline preserves the
  `D+3` cubic control rows supporting the delayed spans.
- All eight runs had zero skipped optimizer steps and finite loss/gradient
  diagnostics.

These are the validation-selected checkpoints actually published, rather
than necessarily the last validation displayed by a W&B run summary:

| State/action | Representation | Stage | Selected epoch | Selected update | Validation action MSE |
|---|---|---|---:|---:|---:|
| StateAbsJoint | raw | base | 50 | 130,350 | 0.02690121 |
| StateAbsJoint | raw | ttRTC | 5 | 13,035 | 0.02677587 |
| StateAbsJoint | B-spline | base | 50 | 130,350 | 0.02701399 |
| StateAbsJoint | B-spline | ttRTC | 5 | 13,035 | 0.02683749 |
| StateAbsEE | raw | base | 40 | 104,280 | 0.15562337 |
| StateAbsEE | raw | ttRTC | 3 | 7,821 | 0.15484847 |
| StateAbsEE | B-spline | base | 10 | 26,070 | 0.14707805 |
| StateAbsEE | B-spline | ttRTC | 5 | 13,035 | 0.14614573 |

Validation action MSE is the complete validation-split
generation-from-scratch metric. W&B summaries show the final validation by
default, so they can differ when the selected optimum occurred earlier. The
authoritative selections are in `V8_AUDIT.json` and each `*.training.json`.

## Audit and publication

- Local root: `robot_policy/output/NEW/V8Full`.
- The final tree contains exactly eight `.pt` files and no resume, periodic,
  or best-epoch training checkpoints.
- All 8/8 checkpoints strict-load for CPU deployment. Joint and EE models have
  87,319,495 and 87,321,288 parameters, respectively.
- All four ttRTC parent paths and parent SHA-256 values match the exact base
  models.
- The complete suite collected 92 tests: 91 passed and one historical optional
  test was skipped.
- The W&B project `robot-policy-bsp-unet-v8-classify-blocks` has exactly eight
  V8 runs, all finished, with zero logged artifacts.
- The portable Hugging Face release is
  `DiscreteRTC/dRTC/NewModel/V8Full`: 32 files comprising 8 models, 16
  server/training JSON files, and 8 encoder/normalization sidecars.
- Remote revision: `da92d9d3899320d132177d4911ed1efae5837bfc`. An independent
  readback found 32/32 files and matched every model LFS SHA-256 and regular
  Git blob hash.

Authoritative evidence is in `V8_STATUS.json`, `V8_DATASET_AUDIT.json`,
`V8_AUDIT.json`, and `V8_HF_UPLOAD.json` under the output root.

## Next

The V8 training, ttRTC finetuning, cleanup, portable deployment packaging, and
publication goal is complete. A separate evaluation can compare open-loop
generation from scratch against GT-prefix RTC on identical examples and
report per-dimension Joint and EE errors; that evaluation is outside this
training goal.
