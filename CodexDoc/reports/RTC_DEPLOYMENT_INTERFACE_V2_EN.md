# RTC deployment-interface verification

Date: 2026-09-18

## Result

The deployed msgpack/WebSocket interface supports RTC inference correctly for
both active action representations. Verification used the real policy server,
the public `PolicyClient`, real DiT-S ttRTC checkpoints on GPU, and localhost
wire serialization—not only direct Python wrapper calls.

| Representation | Tested policy | Required history | Delay 3 result |
| --- | --- | --- | --- |
| raw | FM ttRTC | physical `prev_action_chunk`, `[1,30,7]` | finite `[1,30,7]`; 3 fixed action rows |
| B-spline | joint-DD ttRTC | normalized `prev_control_rows`, `[1,18,7]` | finite `[1,30,7]`; 2 affected spans and 5 fixed cubic-support rows |

The server advertises the contract in its handshake. `PolicyClient` reads
`rtc_requires_previous_field` and sends the matching field through
`infer_realtime`. Delay is measured in raw 30 Hz action steps and is strictly
limited to 1–10. Base checkpoints reject realtime calls. The server keeps no
cross-client plan cache; the previous plan is explicit in every request.

## Important B-spline constraint

B-spline RTC must receive the normalized parameter rows returned by the prior
request. It intentionally rejects decoded 30×7 actions. Re-fitting and cubic
support expansion map raw delay 3 to two affected spans and five preserved
control rows, avoiding the incorrect assumption that raw action indices equal
spline-control indices.

## Hardening performed

The live wire test revealed that msgpack-decoded NumPy arrays can be read-only.
The deployment boundary now creates an owned float32 copy before converting
previous actions or control rows to torch tensors. This prevents transport
buffer aliasing and removes PyTorch's non-writable-array warning. Versioned
Hugging Face layouts now auto-resolve `vN/sidecars/{raw,bspline}`.

## Evidence

- `outputs/FULL_VISION_512/summary/deployment/rtc_websocket_validation.json`
- `outputs/FULL_VISION_512/summary/deployment_validation_dit_{s,b}.json`
- Full project test suite, including read-only transport arrays and versioned
  release sidecar discovery.

This validates policy-server inference only; it does not constitute a
closed-loop robot safety or task-success validation.
