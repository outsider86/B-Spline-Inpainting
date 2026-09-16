# Deployment validation

Status: **passed**, 2026-09-16 UTC.

## Completion audit

| Requirement | Evidence | Result |
| --- | --- | --- |
| Same Piper server interface | WebSocket handshake plus NumPy-msgpack `infer` / `infer_realtime` envelopes; real server process connected with `PolicyClient` | Pass |
| All checkpoint types | Embedded config/sidecar inspection found exactly 36 unique DiT-S/B/L x raw/B-spline x FM/layerwise/joint x base/ttRTC contracts | Pass |
| Correct live observations | `[global, hand]` RGB order, server-side 224x224 DINOv2+SigLIP preprocessing, legacy min/max state -> training z-score conversion, fixed task validation | Pass |
| Correct action decoding | raw identity temporal decode or authoritative cubic 30x18 B-spline basis, then training q01/q99 physical unnormalization | Pass |
| Raw ttRTC | physical `[B,30,7]` prior chunk; 3-step validation preserved exactly 3 shifted action rows | Pass |
| B-spline ttRTC | normalized `[B,18,7]` prior controls; decoded actions rejected; 3 raw steps map to 2 spans and preserve 5 cubic-support rows | Pass |
| Transport behavior | ping, metadata/init, reset, inference, structured errors, ndarray round-trip, real localhost handshake | Pass |
| Automated tests | 16 deployment cases; complete repository suite 35/35 passed | Pass |
| Real checkpoint runtime | Complete 12-case DiT-S matrix on CUDA plus actual frozen vision encoder; all outputs finite and `[1,30,7]` | Pass |

The machine-readable evidence is
[`deployment_validation.json`](deployment_validation.json). It includes all 36
resolved checkpoint contracts and the complete real DiT-S runtime matrix. The
runtime probe is a correctness check, not a replacement for the 100-trial
latency benchmark in the parent summary.

## Runtime matrix

Every row below uses a real final checkpoint. `fixed` is reported only for the
RTC request with a three-raw-step delay.

| Representation | Architecture | Stage | Vanilla finite | RTC finite | Fixed rows |
| --- | --- | --- | --- | --- | ---: |
| raw | FM | base | yes | n/a | n/a |
| raw | FM | ttRTC | yes | yes | 3 |
| raw | layerwise DD | base | yes | n/a | n/a |
| raw | layerwise DD | ttRTC | yes | yes | 3 |
| raw | joint DD | base | yes | n/a | n/a |
| raw | joint DD | ttRTC | yes | yes | 3 |
| B-spline | FM | base | yes | n/a | n/a |
| B-spline | FM | ttRTC | yes | yes | 5 |
| B-spline | layerwise DD | base | yes | n/a | n/a |
| B-spline | layerwise DD | ttRTC | yes | yes | 5 |
| B-spline | joint DD | base | yes | n/a | n/a |
| B-spline | joint DD | ttRTC | yes | yes | 5 |

The frozen vision probe returned finite `[2,16,2176]` fused features for the
two camera views. A separate launched-server test loaded the actual DiT-S
B-spline FM checkpoint, completed the real handshake, and returned finite
`actions [1,30,7]` plus `normalized_control_rows [1,18,7]`.

## Deployment files

- Operator guide: `robot_policy/deployment/README.md`
- Launcher: `robot_policy/deployment/run_policy_server.sh`
- Server package: `robot_policy/src/robot_policy/deployment/`
- Existing-client stats: [`piper_dataset_statistics.json`](piper_dataset_statistics.json)
- Start-pose stats: [`stacking_cups_start_statistics.json`](stacking_cups_start_statistics.json)
- Reproducible runtime validator: `robot_policy/scripts/validate_deployment.py`

No physical command was published during validation. The next operational
step is the reference Piper dry-run, followed by explicit camera/state/gripper
and safety checks before any `--execute` authorization.
