# Sweep Result Organization and Output Cleanup

Date: 2026-09-16

## Status

Complete. The 36-checkpoint DiT-S/B/L sweep is now the sole canonical result set under `robot_policy/outputs/SWEEP`.

## Preserved results

- 36 final checkpoints: 6 each for DiT-S/DiT-B/DiT-L × raw/B-spline.
- Six checkpoint manifests, W&B identity sidecars, compact training logs, final scheduler status, and the completed audit.
- Online W&B histories and model artifacts in `robot-policy-50k-bs32-basic` and `robot-policy-5k-bs32-ttRTC`.
- A small compatibility symlink, `outputs/model_size_sweep_50k_bs32 -> SWEEP`, so immutable absolute parent-lineage paths embedded in checkpoints remain resolvable without data duplication.

## Verification

- `completion_audit.json`: passed, 36 expected / 36 found, zero errors.
- W&B: 36 unique run IDs with the required 18 base / 18 ttRTC split.
- Post-cleanup SHA-256 verification: 36 checked / 0 mismatches.
- Per-group count: exactly six final checkpoints in each of the six capacity/representation groups.
- No `.pt.resume` snapshots or local W&B cache directories remain.

## Deleted artifacts

The cleanup permanently removed 36 redundant resume snapshots, six local W&B cache trees, stale lock/partial-audit files, and twelve superseded top-level output trees: `action_representation_comparison`, `bspline_reproduction`, `checkpoints`, `evaluation`, `latency`, `longrun_50k_bs32`, `pilot`, `pilot_fast`, `prepared`, `raw_actions`, `replay`, and `visualizations`.

Total reclaimed: 83,622,878,276 bytes (about 77.9 GiB). Remaining canonical sweep size: 69,930,109,574 bytes (about 65.1 GiB). Deleted local artifacts are not recoverable from this workspace; online W&B data remains available.

## Next step

Implement the documented stability changes and run DiT-B/L canaries before retraining the six affected layerwise variants. Do not change the preserved 36-checkpoint baseline.
