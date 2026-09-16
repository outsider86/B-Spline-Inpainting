# Final requirement audit

1. **Modular repository/environment:** complete under `robot_policy/`; pinned environment and commands are in `environment.lock.yml`, `README.md`, and `docs/environment.md`.
2. **Source audit/attribution:** complete in `docs/source_map.md` and `THIRD_PARTY_NOTICES.md`. Restrictively licensed references were studied without copied code.
3. **Observation milestone:** complete for all 31,706 frames, with real-batch shapes/gradients, frozen-weight checks, cache precision metrics, and inspected images under `outputs/visualizations/`.
4. **B-spline integration:** complete through the authoritative BSplineEncoder adapter; fixed cubic geometry, quantization, terminal masks, and inspected reconstruction/error evidence are recorded.
5. **Three policies/fairness:** complete with configuration selection, shared trainer, 7.4% parameter spread, tiny-overfit diagnostics, attention topology, forward/backward/sample and loss-gradient tests.
6. **Six checkpoints:** complete and independently CPU-reloaded. `outputs/checkpoints/checkpoint_manifest.json` has six hashes, exact configuration, source snapshot, parent hash, split/version metadata, training accounting, and selection metrics.
7. **Training/inference RTC:** complete for all three architectures using parent-policy predictions, decreasing-exponential D=S draws, cubic D+3 support, raw within-span mapping, and mock executor/replay.
8. **Block diffusion/KV cache:** complete only for `discrete_joint`, with a cache-disabled oracle, exact token equality, invalidation checks, and measured crossover behavior.
9. **Evaluation/latency/visuals:** complete for all six; see `docs/evaluation_report.md`, `docs/cache_and_latency.md`, and `docs/visual_qa.md`.
10. **Reproduction:** command-level workflow is in `README.md`; training supports periodic resume artifacts and explicit `--resume`.

## Honest limitations and follow-ups

- Results are open-loop and dataset replay, never closed-loop success rates.
- Cartesian XYZ/rotation metrics and image-space projections are unavailable without calibrated kinematics/camera geometry.
- DDP initialization failed in this installed runtime, so parallel independent one-GPU runs preserved fairness instead.
- The supplied `dd-openvla` snapshot did not contain discrete training-time RTC; the project documents its clean-room combination of audited block/protection semantics with deployment-available prior predictions.
- A credential was found embedded in a restrictive reference launch script. It was not copied, printed, or used; the reference owner should rotate it and remove it from history.
