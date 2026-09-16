# Data contract

## Dataset and alignment

The source is LeRobot v2.1, Piper, 52 episodes / 31,706 frames at 30 Hz. `action[t]` and `observation[t]` share a row and timestamp. The metadata does not independently prove whether that command was collected before or after the state sample, so this repository records the row-alignment convention rather than asserting a stronger causal convention.

Splits are made by episode before windows. Training-only statistics normalize state and action. A sample never crosses an episode boundary; missing terminal actions repeat the final action and `action_valid_mask` identifies recorded versus padded raw time.

| Field | Shape | dtype | units / meaning |
|---|---:|---|---|
| cached fused vision | `[B,2,16,2176]` | float16 on disk | two cameras, fixed 4x4 spatial grid, DINO(1024)+SigLIP(1152) |
| normalized state | `[B,7]` | float32 | six absolute joints + absolute gripper, train mean/std |
| observation tokens | `[B,33,D]` | float32/bfloat16 compute | 16 global + 16 hand + one state token |
| observation valid mask | `[B,33]` | bool | true means usable key/token |
| raw action window | `[B,30,7]` | float32 | 30 samples at 30 Hz; joint and gripper units remain separate |
| raw action validity | `[B,30]` | bool | false only for repeat-padded terminal targets |
| continuous spline target | `[B,18,7]` | float32 | normalized control points before discretization |
| discrete spline target | `[B,18,7]` | uint8 disk / int64 model | ordered bin positions 0..255 |
| control validity | `[B,18,7]` | bool | Greville location lies in recorded time; used for loss |
| model action positions | FM `[B,18,7]`; discrete flattened `[B,126]` | float / int | no knot/span/special targets |

## Fixed geometry

- Degree 3; 18 control points; 15 executable spans; three right-support spans.
- Span length 2 raw steps = 66.667 ms. Executable domain is `[0,30)` raw steps; full fixed knot vector has 22 entries.
- Open span `j` has cubic support controls `{j,j+1,j+2,j+3}`. The union for D committed spans is controls `0..D+2` (D+3 controls), with three-control overlap between adjacent spans.
- Fixed knots, degree, sample period, duration/spans, implementation version, phase and calibration are decode metadata, never prediction targets.

## Mask meanings

- `valid`: recorded target exists.
- `masked/supervised`: valid discrete position selected by corruption; CE and expected distance use exactly this same set.
- `fixed/immutable`: prior-plan support that may neither receive loss nor be remasked.
- `active block`: positions progressively recovered now.
- `future block`: unavailable to current joint action queries.
- `PAD` is a batch/storage concept only; 256 is MASK input only and output heads contain exactly the 256 numeric bins.

RTC uses uppercase D=S in knot spans, 1..5, and lowercase d=s in raw steps. Exact whole-span experiments map `d=2D`; raw-delay evaluation additionally covers d=1..10 and records `phase_steps=d mod 2`. Delays beyond the 30-step prediction horizon return a coverage error and are never silently clamped.

