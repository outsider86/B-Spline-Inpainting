# dd-openvla Block-Diffusion / K/V-Cache Audit

Date: 2026-09-16

## Reference behavior inspected

The local `RefCode/dd-openvla` D2F implementation combines four ideas:

1. A block-causal attention mask: observation/prompt tokens attend within the condition, and each action block attends to the condition plus completed/current blocks (`prismatic/extern/hf/modeling_prismatic.py:1248`; training counterpart in `vla-scripts/finetune_d2f.py:316`).
2. A state machine that can expose later blocks while earlier blocks are being decoded (`modeling_prismatic.py:1329-1365`).
3. A cropped K/V cache that retains stable condition/completed-block states and discards active-block states after each iteration (`modeling_prismatic.py:1371-1418`).
4. Confidence-based token skipping and progressive block completion (`modeling_prismatic.py:1420-1454`).

## Compatibility assessment

The current `JointDiscretePolicy` was already trained with monotonic block corruption and uses a matching block-causal topology. It also cached observation and completed action blocks. The remaining avoidable work was at each inter-block boundary:

- legacy path: recompute the completed block to commit its K/V, then make a separate first forward for the next masked block;
- fused path: process the completed block and next masked block in one block-masked forward, retain only the completed block K/V, and reuse the next-block logits as its first denoising iteration.

This implements the cache-transition portion of the reference design without introducing confidence thresholds or changing the trained sampler schedule. Adding reference-style adaptive early block exposure or token skipping would change model outputs and therefore requires a separate quality evaluation or training goal.

## Implementation

- `src/robot_policy/policies/discrete_joint.py:137-174`: optional `fuse_cache_transition` sampler path, enabled by default.
- `src/robot_policy/evaluation/latency.py:65-107`: benchmarks uncached, legacy cached, and fused cached modes independently and checks token equality.
- `tests/test_policies.py`: verifies uncached, legacy cached, and fused cached sampling agreement on the deterministic test model.

## Measured result

Protocol: all 12 real joint checkpoints; batch 1; 8 rounds; RTX PRO 6000 Blackwell; 10 warmups plus 100 synchronized trials.

- Exact legacy-cached versus fused-cached token matches: **12/12**.
- Maximum legacy/fused token difference: **0 bins**.
- Median p50 speedup: **8.65%**.
- Range: **5.29% to 12.48%**.
- Full pre-change reports: `latency_baseline/`.
- Full optimized reports: `latency/`.
- Per-checkpoint visualization: `joint_cache_acceleration.png`.

The uncached full-sequence and cached kernels are not universally bit-identical on the real large checkpoints because different attention shapes can alter floating-point ordering; this is distinct from the new optimization. The deployed legacy cached and fused cached paths are exactly token-identical in all measured checkpoints.
