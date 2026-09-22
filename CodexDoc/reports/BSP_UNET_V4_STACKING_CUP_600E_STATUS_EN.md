# BSP U-Net V4 stacking_cup_30hz 600-Epoch Status

Date: 2026-09-22 UTC

Raw and B-spline h2 Flow-Matching runs are active on GPUs 2/3. They use the
61-episode, 45,285-frame `Data/stacking_cup_30hz` dataset, a deterministic
55-train / 6-validation / 0-test split, batch/effective batch 64, and two
timesteps × two cameras with scratch ResNet-18 encoders.

There are 40,748 train and 4,537 validation windows. With `drop_last=True`, one
loader epoch is 636 updates; 600 epochs are exactly 381,600 updates. Complete
validation runs every 10 epochs. Exact EMA/online snapshots are retained at
epochs 100, 200, 300, 400, 500, and 600; final `base.pt` separately carries
validation-best inference weights.

W&B project: `robot-policy-bsp-unet-v4-stacking-cup-600e`

- Raw run: `bsm6y5gh` on GPU 2.
- B-spline run: `1272qqoh` on GPU 3.

All action targets, normalization, spline calibration, and RGB data were
rebuilt for this dataset. The RGB cache contains 61 length-verified episode
files and all 45,285 frames.

Next: audit the first full validation at epoch 10, then verify the first
immutable checkpoint at epoch 100.
