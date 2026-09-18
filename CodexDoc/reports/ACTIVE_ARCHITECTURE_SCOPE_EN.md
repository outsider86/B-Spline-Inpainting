# Active architecture scope: FM and joint DD

Date: 2026-09-17

## Decision

All future training, RTC fine-tuning, open-loop evaluation, inference-RTC
evaluation, latency benchmarking, and aggregate comparisons will include only:

- `fm`: continuous flow matching.
- `discrete_joint`: joint-sequence discrete diffusion with blockwise KV cache.

`discrete_layerwise` is retired from active experiments. The implementation
remains loadable only to reproduce the existing checkpoints and historical
reports. Existing files and remote artifacts are not deleted.

## Evidence

The completed 36-checkpoint sweep found severe gradient-norm overflow in six
layerwise DiT-B/L runs, while the joint policy provides the discrete generation
path relevant to continued cache and unmasking work. Keeping FM preserves the
continuous baseline and keeping joint DD preserves the target discrete method.

## Operational effect

- A raw/B-spline x base/ttRTC experiment now contains eight variants.
- A DiT-S/B/L x raw/B-spline x base/ttRTC capacity sweep now contains 24
  checkpoints.
- Historical layerwise results remain valid audit evidence but must not be
  mixed into new aggregate comparisons.

## Next step

Run new experiments and W&B comparisons with FM and joint DD only.
