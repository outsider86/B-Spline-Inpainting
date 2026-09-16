# Policy-server deployment

This server exposes every checkpoint in `outputs/SWEEP` through the same
msgpack-over-WebSocket envelope used by the previous Piper deployment:

- handshake: server metadata is sent immediately after connection;
- request: `{"type":"infer","request_id":...,"payload":{"examples":[...]}}`;
- response: `response["data"]["actions"]`, shape `[B,30,7]`, already in
  physical absolute Piper coordinates;
- realtime request type: `infer_realtime`;
- transport: WebSocket, compression disabled, NumPy-aware msgpack.

The implementation is in `src/robot_policy/deployment`. It loads architecture,
model size, representation, sampling settings, and normalization from the
checkpoint itself. An operator does not select a separate model YAML. For the
organized 36-checkpoint sweep, it automatically resolves the shared immutable
sidecars from `outputs/SWEEP/summary/cache/{raw,bspline}`.

## Observation and action contract

Each example is:

```python
{
    "image": [global_rgb, hand_rgb],  # HWC uint8, exact order
    "state": normalized_state[None, :],  # shape [1,7]
    "lang": "Stack the cups.",
}
```

The server resizes both RGB images to 224x224 and applies the same frozen
DINOv2 + SigLIP preprocessing as training. The default `state` contract is the
legacy Piper client's min/max normalization to `[-1,1]`. The server inverts
that transform and then applies this codebase's training z-score. Requests can
instead set `state_coordinates="physical"` or `state_coordinates="zscore"`.

The language string is checked because this checkpoint is task-specific, but
these compact policies were not trained with language tokens. A different
instruction is rejected rather than silently implying language conditioning.

Actions are decoded inside the server:

- raw checkpoints: 30x7 normalized actions -> q01/q99 unnormalization;
- B-spline checkpoints: 18x7 controls -> authoritative smooth cubic 30x18
  basis multiplication -> 30x7 normalized actions -> q01/q99
  unnormalization;
- the gripper is projected to 0/1 at physical threshold 0.3 by default.

## Start a server

From `robot_policy`:

```bash
export CKPT="$PWD/outputs/SWEEP/dit_l/bspline/checkpoints/fm_base.pt"
CUDA_VISIBLE_DEVICES=0 deployment/run_policy_server.sh
```

Equivalent direct invocation:

```bash
PYTHONPATH=src /home/wangpc/miniconda3/envs/starVLA/bin/python \
  -m robot_policy.deployment.server \
  --ckpt_path outputs/SWEEP/dit_l/bspline/checkpoints/fm_base.pt \
  --port 10093 --device cuda --precision bf16
```

Useful switches are `--no-binary-gripper`, `--gripper-threshold`,
`--prepared-path`, `--host`, and `--idle-timeout`.

Before using the existing Piper live client, export its expected statistics:

```bash
PYTHONPATH=src /home/wangpc/miniconda3/envs/starVLA/bin/python \
  -m robot_policy.deployment.export_stats \
  --ckpt_path "$CKPT" \
  --output outputs/SWEEP/summary/deployment/piper_dataset_statistics.json \
  --start-output outputs/SWEEP/summary/deployment/stacking_cups_start_statistics.json
```

Pass that JSON as the Piper client's `--stats` file. It deliberately exposes
the training-episode physical state range for the legacy client transform;
the server converts the resulting min/max-normalized state to the z-score used
during this policy's training.

Pass the second JSON as `--start-stats`; it contains min/max/median over the
first physical state of all 52 stacking-cups demonstrations, as required by
the handoff's guarded start-pose check.

## Base and ttRTC behavior

All 36 checkpoints support ordinary `infer`. An RTC/ttRTC checkpoint also
supports `infer_realtime`, with a strict representation contract:

| Checkpoint | Required previous field | Coordinates | Delay |
| --- | --- | --- | --- |
| raw ttRTC | `prev_action_chunk`, `[B,30,7]` | physical robot actions returned by the prior call | 1..10 raw 30 Hz steps |
| B-spline ttRTC | `prev_control_rows`, `[B,18,7]` | normalized spline controls returned as `normalized_control_rows` by the prior call | 1..10 raw 30 Hz steps |

The B-spline endpoint rejects decoded 30x7 actions. A delay of `d` raw steps
shifts and refits the prior curve, maps to `ceil(d/2)` affected spline spans,
and preserves `ceil(d/2)+3` cubic-support control rows. It never treats a
30 Hz action index as a control-row index.

The former Piper async client can be used unchanged for raw ttRTC. It cannot
be used unchanged for B-spline ttRTC because it discards control rows. Use
`robot_policy.deployment.PolicyClient`, which retains them:

```python
from robot_policy.deployment import PolicyClient

with PolicyClient("127.0.0.1", 10093) as client:
    first = client.predict_action([example])
    second = client.predict_action_realtime(
        [next_example],
        inference_delay=3,
        previous=first["normalized_control_rows"],
    )
```

The server is stateless across requests and clients. RTC history is explicit
in each request, preventing one robot/client from inheriting another client's
plan.

## Safety and dry-run boundary

This package performs inference only. It does not publish robot commands. Keep
the reference Piper client in dry-run mode until camera order, state units,
task string, 30 Hz execution, action range, start pose, gripper direction, and
emergency-stop behavior have been verified. Do not add `--execute` merely
because server inference succeeds.

With the server running, the existing reference live client can be exercised
without robot commands as follows (there is intentionally no `--execute`):

```bash
cd /scratch/wangpc/DiscreteRTCv2
/home/wangpc/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/Piper/eval_files/piper_live_client.py \
  --host 127.0.0.1 --port 10093 \
  --stats /scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/SWEEP/summary/deployment/piper_dataset_statistics.json \
  --start-stats /scratch/wangpc/B-Spline-Inpainting/robot_policy/outputs/SWEEP/summary/deployment/stacking_cups_start_statistics.json \
  --task "Stack the cups." --rate-hz 30 --duration 30 \
  --save-viewer-data
```

For a raw ttRTC checkpoint, the reference
`piper_async_training_rtc_client.py` recognizes the advertised
`rtc_mode=training_time_hard_prefix` and can use `infer_realtime`. The server
advertises this sweep's trained maximum of 10 raw action steps and never
silently accepts a larger delay.

## Verification

```bash
cd /scratch/wangpc/B-Spline-Inpainting/robot_policy
PYTHONPATH=src /home/wangpc/miniconda3/envs/starVLA/bin/python -m pytest -q
```

Deployment coverage includes checkpoint-sidecar validation for all 36 files,
every architecture x representation family, both RTC conditioning spaces,
prefix preservation, invalid-shape/task/delay rejection, NumPy serialization,
router errors, and a real localhost WebSocket handshake/inference round-trip.
