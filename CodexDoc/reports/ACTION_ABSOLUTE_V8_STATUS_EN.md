# Absolute-Action V8 Status

V8 covers only `classify_blocks` and produces eight final checkpoints:

`{StateAbsJoint, StateAbsEE} × {raw, B-spline} × {base, ttRTC}`.

All jobs use BSP-UNet flow matching with independent scratch ResNet-18 camera
encoders and SpatialSoftmax. Each example has one observation timestep (h1)
containing exactly the current global and hand RGB frames.

StateAbsJoint maps a 7D measured joint/gripper state to a 7D absolute
next-step joint/gripper action. StateAbsEE uses the same 7D measured-joint
state and predicts an 8D absolute next-step TCP xyz, quaternion-xyzw, and
gripper action. The implementation now derives action width from config and
checkpoint metadata throughout preparation, raw/B-spline codecs, policy heads,
metrics, RTC masks, sidecar validation, and deployment. Legacy checkpoints
remain 7D by default. An explicit 8D B-spline/ttRTC regression passes.

Both dataset trees contain 85 episodes, 185,913 frames, and 170 global/hand
MP4 files. State/actions are finite, frame indices are consecutive, and
timestamps increase strictly. Every corresponding video matches byte-for-byte
by SHA-256, and temporal row identity also matches, so V8 safely shares one RGB
cache while isolating all action caches.

The deterministic seed-20260915 split is 76 train episodes / 166,875 windows
and 9 validation episodes / 19,038 windows, with no test split. Batch size and
effective batch size are both 64. Bases run for 100 epochs (260,700 updates)
with complete validation every 10 epochs. Their exact validation-selected
weights initialize five-epoch ttRTC children (13,035 updates), which validate
over the complete split every epoch. Four base lanes run concurrently on GPUs
0--3, followed by four matching ttRTC lanes.

All runs share the W&B project
`robot-policy-bsp-unet-v8-classify-blocks`; model artifacts are disabled.
Local outputs are under `robot_policy/output/NEW/V8Full`, and the planned
portable publication target is `DiscreteRTC/dRTC/NewModel/V8Full`.

Current status: the source/action contract audit and variable-action-width
tests have passed. The driver is preparing four isolated action caches and the
audited shared RGB cache. Next it will train, strict-load and lineage-audit the
eight validation-selected models, remove transient checkpoints, and publish
models plus portable loading sidecars.
