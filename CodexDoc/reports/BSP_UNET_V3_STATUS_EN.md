# BSP scratch-vision U-Net v3 status

Updated: 2026-09-19 UTC

## Status

Implementation, dataset caching, deployment integration, and preflight testing
are complete. The 16-checkpoint six-GPU experiment is starting.

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
- Training stage: 50k base or 5k RTC child.
- Total: 16 checkpoints.

Discrete diffusion trains with monotonic block corruption and reconstructs
discrete action tokens. Inference starts from every mutable token masked and
iteratively unmasks it over eight rounds. Both policy families use the same RTC
hard mask; B-spline RTC fixes only the exact control-point support of affected
spans.

## Verified preflight

- Full-size FM parameters: 89,254,855.
- Full-size discrete-DD parameters: 90,055,360.
- Full forward/backward and from-scratch generation pass in BF16.
- Real batch-64 training, EMA, validation, checkpoint save/reload, and manifest
  publication pass.
- Project tests: 62/62 pass.
- RGB cache: 52 episodes / 31,706 frames / two cameras / 84×84 uint8 CHW.

## Next

Run six GPU queues continuously, monitor gradients and validation convergence,
then evaluate all 16 checkpoints, verify the deployed one/two-frame RTC
interface, prepare bilingual reports, and publish `NewModel/v3`.
