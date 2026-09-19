# B-spline RTC hard-mask audit

Updated: 2026-09-19 UTC

## Result

Both continuousRTC (FM) and discreteRTC (joint discrete diffusion) use the same hard B-spline support mask. For `D` affected left-boundary spans of a degree-`p` spline, the fixed control rows are exactly `[0, D+p)`. With the active cubic setup, that is `[0, D+3)`.

This is the exact union of control rows whose basis functions are nonzero in those affected spans. No control row outside that union is conditioned on the previous plan.

## Enforcement

- Training creates one representation-aware `fixed_mask` in `rtc/training.py` and sends it to either policy.
- FM overwrites only fixed rows before the velocity prediction and excludes them from the mutable loss. During inference it reapplies the mask before and after every Euler update.
- Joint DD inserts only the fixed discrete tokens, excludes them from corruption supervision, marks them immutable, and reapplies them through iterative unmasking.
- Deployment, inference, RTC evaluation, and latency paths now pass explicit checkpoint/config spline geometry rather than relying on default constants.
- RTC deployment metadata exposes `rtc_mask_type=hard` and the exact B-spline mask scope.

## Delay mapping

For B-spline RTC, a raw delay is mapped to `D = ceil(raw_delay / span_length_steps)`. The active span length is two raw action steps. An odd delay therefore conditions the entire partially affected span, which is required because a spline span is jointly defined by its local controls.

Basis overlap means those same mathematically required controls may also influence adjacent future spans. This is inherent spline support, not extra conditioning; the implementation does not fix another control row beyond the affected-span support union.

## Regression evidence

- The hard mask is compared against the nonzero columns of the authoritative B-spline basis for affected spans 1 through 5.
- For both FM and joint DD, changing every prefix value outside the exact fixed support leaves same-seed sampling and training loss exactly unchanged.
- Existing checkpoints already used the same `D+3` behavior. The change makes geometry config-derived, documents the contract, and adds regression protection; it does not require retraining.

## Key files

- `robot_policy/src/robot_policy/rtc/delay_mapping.py`
- `robot_policy/src/robot_policy/rtc/training.py`
- `robot_policy/src/robot_policy/policies/fm.py`
- `robot_policy/src/robot_policy/policies/discrete_joint.py`
- `robot_policy/tests/test_bspline_and_rtc.py`
- `robot_policy/tests/test_inference_rtc.py`

