# NewModel v2: active full-vision policies

This release contains the 16 audited active checkpoints:

`DiT-S/B × raw/B-spline × FM/joint-DD × base/ttRTC`.

Every observation uses two complete 16×16 vision-token grids: 256 tokens per
camera and 512 total. Base checkpoints were trained for 50,000 updates and
ttRTC children for 5,000 updates at effective batch size 32. DiT-L and
layerwise DD are retired and intentionally absent from v2; they remain in v1
for historical reproduction.

## Layout

- `dit_{s,b}/{raw,bspline}/checkpoints/`: final checkpoints, W&B identity
  sidecars, and checkpoint manifests.
- `sidecars/{raw,bspline}/`: action codec, calibration, normalization, and
  split metadata required by the loader.
- `configs/`: the four active full-vision configurations.
- `evaluation/`: completion audit, 16-row metrics, plots, deployment
  validation, decoder validation, and bilingual summaries.
- `deployment/`: server/client contract and RTC validation report.

## RTC deployment contract

All checkpoints support ordinary `infer`. Only RTC/ttRTC checkpoints support
`infer_realtime`.

| Representation | Previous value required by `infer_realtime` | Shape | Coordinates |
| --- | --- | --- | --- |
| raw | `prev_action_chunk` | `[B,30,7]` | physical actions returned by the previous request |
| B-spline | `prev_control_rows` | `[B,18,7]` | normalized spline controls returned as `normalized_control_rows` |

`inference_delay` is an integer from 1 through 10 in raw 30 Hz action steps.
The B-spline endpoint intentionally rejects decoded action chunks: RTC must
preserve and shift the parameter-space controls. The server is stateless;
clients explicitly return the previous chunk or control rows with each RTC
request.

The repository loader automatically discovers `sidecars/{raw,bspline}` when
the versioned directory structure is retained. See `deployment/README.md` for
server and client examples.

## Verification

- 16/16 local and remote-W&B completion audit.
- DiT-S and DiT-B deployment matrices: 8/8 runtime cases each.
- Raw and B-spline RTC verified through the real msgpack/WebSocket interface
  with finite `[1,30,7]` actions and correct hard-prefix metadata.
- Joint cached/fused decoding is token-identical for all eight joint-DD
  checkpoints.
