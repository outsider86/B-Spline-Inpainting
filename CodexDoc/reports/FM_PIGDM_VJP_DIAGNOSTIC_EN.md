# FM PiGDM VJP Step-by-Step Diagnostic

Date: 2026-09-22 UTC

## Conclusion

The visual concern is correct for the batch-64 checkpoints: base-FM PiGDM is
nearly equivalent to directly stitching the GT prefix onto the from-scratch
suffix. The VJP is computed, but the endpoint cross-Jacobian transmits too
little correction into mutable suffix coordinates for the change to be visible
at the trajectory plot scale.

The implementation was checked term-by-term against Kinetix:
`x1=x_t+(1-t)v`, `error=hard_mask*(condition-x1)`,
`correction=J_x1^T error`, and
`x_next=x_t+dt*(v+guidance*correction)`. The coefficient schedule and maximum
weight 5 also match. A scalar linear-policy regression matches the JAX equation
at every step, so this is not an evident sign, mask, or Euler transcription
error.

On the same four motion-rich validation chunks with prefix 6 and paired noise,
raw/B-spline cross-VJP gain is only 0.00925/0.01918 relative to the condition
error. Final PiGDM-versus-naive-stitch suffix RMS is 0.001597/0.003117. Suffix
GT MSE improves only from 0.027876 to 0.027646 for raw and from 0.029731 to
0.029575 for B-spline.

Artifacts are under
`robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch64/vjp_diagnostic/`.
Each variant contains per-step JSON/CSV, a VJP trace plot, a direct
PiGDM-versus-naive-stitch plot, and paired trajectory arrays.

## Next step

Do not treat batch-64 base-FM PiGDM as effective RTC yet. Run paired guidance
strength / flow-step sensitivity and prioritize the ttRTC-finetuned sampler;
accept an RTC path only when it produces a stable suffix change and improves
GT suffix error over naive stitching.
