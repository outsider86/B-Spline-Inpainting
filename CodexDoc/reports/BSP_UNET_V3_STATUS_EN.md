# BSP scratch-vision U-Net v3 status

Updated: 2026-09-19 UTC

## Status

The original 16-checkpoint matrix completed. FM checkpoints are final. The
first DD pass exposed a training/selection mismatch and is preserved under
`outputs/BSP_UNET_V3/diagnostics/legacy_d2f_partialval_50k`; four corrected DD
base+RTC queues are now running on GPUs 0--3. The frozen FM checkpoints are
uploading to `DiscreteRTC/dRTC/NewModel/v3`.

## Confirmed reference architecture

The reference BSP policy trains its vision encoders jointly from scratch. Each
camera has a separate non-pretrained ResNet-18 whose BatchNorm layers are
replaced by GroupNorm. A learned 1x1 map and 32-keypoint SpatialSoftmax produce
64 values per camera, followed by `Linear(64,64)+ReLU`. Robot state is
concatenated directly. Consecutive observations are flattened into one global
FiLM condition for a `[256,512,1024]` temporal U-Net.

The v3 implementation uses the same image encoder and U-Net scale. For this
dataset, each observation step is `2×64 + 7 = 135` values; the one-frame and
two-frame conditions are therefore 135-D and 270-D. DINOv2 and SigLIP are not
part of this experiment.

## Experiment matrix

- Policy: continuous FM or 256-bin discrete diffusion.
- Action representation: raw 30×7 or cubic B-spline 18×7 controls.
- Observation: current frame or consecutive previous/current frames.
- Training stage: FM uses 50k base; corrected DD uses 10k base selected by
  complete validation; both use 5k RTC children.
- Total: 16 checkpoints.

Corrected discrete diffusion uses 50% fully masked examples and 50% monotonic
D2F block corruption. This retains partial inpainting/RTC training while
matching generation's all-MASK initial state. Inference deterministically
unmasks over eight MaskGIT rounds. Both policy families use the same RTC hard
mask; B-spline RTC fixes only the exact control-point support of affected spans.

## DD diagnosis and evidence

The old selector evaluated only 256 of 3,077 validation examples. Its noisy
early minima did not predict full-test behavior, while teacher-corruption loss
continued improving after rollout quality saturated. Complete-validation A/B
tests selected 50% full-MASK exposure. At 10k, its held-out test normalized
action MSE was 0.04773/0.04977 for raw h1/h2 and 0.06547/0.06460 for B-spline
h1/h2. The archived 50k B-spline baselines were 0.08192/0.07738.

DD-OpenVLA categorical sampling and annealed Gumbel remasking were also tested.
Changes were small and inconsistent across representations, so deterministic
argmax/confidence decoding remains the validation-selected default. BSP-UNet is
convolutional, so transformer KV caching does not apply; evaluation metadata
now reports `joint_kv_cache=false`.

## Verified preflight

- Full-size FM parameters: 89,254,855.
- Full-size discrete-DD parameters: 90,055,360.
- Full forward/backward and from-scratch generation pass in BF16.
- Real batch-64 training, EMA, validation, checkpoint save/reload, and manifest
  publication pass.
- Project tests: 65/65 pass.
- RGB cache: 52 episodes / 31,706 frames / two cameras / 84×84 uint8 CHW.

## Next

Finish the corrected DD queues, run the final 16-checkpoint open-loop,
latency, train/test RTC-trajectory and deployment audits, then complete the
bilingual release and `NewModel/v3` upload.
