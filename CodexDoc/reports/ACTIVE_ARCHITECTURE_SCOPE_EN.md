# Active research scope: FM/joint DD on DiT-S/B

Date: 2026-09-18

## Decision

All future training, RTC fine-tuning, open-loop evaluation, inference-RTC
evaluation, latency benchmarking, and aggregate comparisons will include only:

- `fm`: continuous flow matching.
- `discrete_joint`: joint-sequence discrete diffusion with blockwise KV cache.
- DiT-S and DiT-B capacity presets.

`discrete_layerwise` is retired from active experiments. The implementation
remains loadable only to reproduce the existing checkpoints and historical
reports. Existing files and remote artifacts are not deleted.

DiT-L is likewise retired from active research. Its configs, completed
checkpoints, partial interrupted outputs, and historical reports are preserved,
but active training/evaluation entry points reject it as legacy/load-only.

## Evidence

The completed 36-checkpoint sweep found severe gradient-norm overflow in six
layerwise DiT-B/L runs, while the joint policy provides the discrete generation
path relevant to continued cache and unmasking work. Keeping FM preserves the
continuous baseline and keeping joint DD preserves the target discrete method.

The full-vision DiT-L extension was stopped on 2026-09-18 by research-scope
decision. Continuing with DiT-S/B retains the small/base capacity comparison
while avoiding the substantially larger DiT-L compute cost in follow-up work.

## Operational effect

- A raw/B-spline x base/ttRTC experiment now contains eight variants.
- The active DiT-S/B x raw/B-spline x FM/joint-DD x base/ttRTC full-vision
  matrix contains 16 checkpoints.
- Historical layerwise results remain valid audit evidence but must not be
  mixed into new aggregate comparisons.
- Historical DiT-L results remain available as archived evidence but must not
  be scheduled or mixed into new active summaries.

## Next step

Finish and compare DiT-S/B using FM and joint DD only.
