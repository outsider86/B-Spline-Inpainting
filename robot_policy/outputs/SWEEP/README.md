# Model-Size Sweep Results

This is the canonical local result set for the completed DiT-S/B/L capacity sweep.

## Contents

| Capacity | Representation | Final checkpoints |
|---|---|---:|
| DiT-S | raw | 6 |
| DiT-S | B-spline | 6 |
| DiT-B | raw | 6 |
| DiT-B | B-spline | 6 |
| DiT-L | raw | 6 |
| DiT-L | B-spline | 6 |

Each group is under `dit_{s,b,l}/{raw,bspline}/`. Its `checkpoints/` directory contains FM, layerwise diffusion, and joint diffusion base/ttRTC checkpoints, plus the manifest and any W&B identity sidecars created during interruption-safe training. Compact training logs are retained under `logs/`.

Top-level evidence:

- `completion_audit.json`: passed 36/36 with zero errors, 36 unique W&B run IDs, and an exact 18 base / 18 ttRTC project split.
- `sweep_status.json`: final scheduler status and run metadata.
- `summary/`: fresh 36-checkpoint open-loop, delay, latency, B-spline decode, and dd-openvla-aligned cache-acceleration evaluation.
- `summary/deployment/`: Piper-compatible statistics plus the 36-checkpoint
  contract audit and real-GPU server runtime matrix.

The W&B projects are `robot-policy-50k-bs32-basic` and `robot-policy-5k-bs32-ttRTC`. Online run histories and model artifacts remain in W&B; redundant local W&B caches were removed.

## Integrity

On 2026-09-16, all 36 final files were checked against the SHA-256 values in `completion_audit.json` after reorganization and cleanup: 36 checked, 0 mismatches.

`../model_size_sweep_50k_bs32` is a compatibility symlink to this directory. It is required because immutable checkpoint lineage and audit records contain the original absolute path; it does not duplicate checkpoint data.

## Cleanup

Removed permanently:

- 36 completed-run `.pt.resume` snapshots and all six local W&B cache trees;
- stale lock files and the superseded partial audit;
- the previous `action_representation_comparison`, `bspline_reproduction`, `checkpoints`, `evaluation`, `latency`, `longrun_50k_bs32`, `pilot`, `pilot_fast`, `prepared`, `raw_actions`, `replay`, and `visualizations` output trees.

The cleanup reclaimed 83,622,878,276 bytes (about 77.9 GiB). The canonical sweep occupies 69,930,109,574 bytes (about 65.1 GiB).

## Deployment

All 36 checkpoints are served by the general interface documented at
`../../deployment/README.md`. The server auto-resolves this directory's shared
raw/B-spline sidecars and reads architecture/model-size/stage directly from
each checkpoint. `summary/deployment/deployment_validation.json` audits all 36
headers and records finite real-GPU inference for the complete 12-case DiT-S
architecture x representation x stage matrix.

## Next step

Before retraining the six numerically affected layerwise variants, implement corruption-level conditioning, adaLN-Zero-style modulation, robust FP64 gradient clipping, finite-gradient telemetry, and DiT-B/L canary runs.
