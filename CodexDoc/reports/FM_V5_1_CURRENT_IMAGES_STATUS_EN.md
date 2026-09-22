# Flow Matching V5.1: current-timestep two-camera input

Updated: 2026-09-22 UTC

## Objective and sole V5 difference

- V5 uses previous and current global/hand frames: four images total and `observation_horizon=2`.
- V5.1 uses only the current global/hand pair: two images total and `observation_horizon=1`.
- Both use the 14D state `measured_state[0:7] + previous_command[7:14]`.
- Action targets, train/validation episode splits, batch size 64, learning rate, 100 epochs, full validation every epoch, and validation-best publication every 10 epochs are unchanged.
- The vision frontend remains two independent scratch-trained ResNet-18 + SpatialSoftmax encoders; it is not DINOv2.

## Cache and output isolation

V5.1 reuses V5's read-only action-target, normalization, split, and RGB caches because none depends on the observation horizon. V5.1 checkpoints, logs, W&B runtime data, and recovery state are isolated under `BSP_UNET_V5_1_*_100E` and cannot overwrite V5.

## Final training state

Training was stopped as requested and no training process remains. All six jobs now publish a `base.pt`: five completed 100 epochs; the final classify-blocks B-spline run was stopped after its epoch-80 checkpoint and retains the best full-validation weights observed through that point.

| Task | Representation | Trained through | Selected update | validation/action_mse |
|---|---|---:|---:|---:|
| classify_blocks | raw | 100 epochs | 70304 | 0.0285403539 |
| classify_blocks | B-spline | 80 epochs | 56784 | 0.0268160413 |
| hanging_mug | raw | 100 epochs | 30160 | 0.0138457487 |
| hanging_mug | B-spline | 100 epochs | 33930 | 0.0132550516 |
| stacking_cup | raw | 100 epochs | 20650 | 0.0191705185 |
| stacking_cup | B-spline | 100 epochs | 31270 | 0.0186288262 |

All runs use the original V5 W&B project `robot-policy-bsp-unet-v5-state-command-100e`, and no W&B artifact was uploaded.

The authoritative machine status is `robot_policy/outputs/BSP_UNET_V5_1_STATUS.json`.

## Unified deployment interface

Each final model is paired with:

- `base.pt`: weights and embedded training configuration;
- `base.server.json`: launch settings and the expected architecture, raw/B-spline representation, observation horizon, 14D state width, and camera order.

Launch command:

```bash
export CKPT=/absolute/path/to/base.pt
export POLICY_CONFIG=/absolute/path/to/base.server.json
CUDA_VISIBLE_DEVICES=0 robot_policy/deployment/run_policy_server.sh
```

The same server entry point supports V5 and V5.1. A mismatched model/JSON pair is rejected so the V5 four-image and V5.1 two-image contracts cannot be silently mixed.

## Completed

- Added and validated all six V5.1 configurations with `observation_horizon=1` and `state_dim=14`.
- Verified shared caches and scratch-only runtime paths.
- Generated `base.server.json` for all six existing V5 models.
- Passed model+JSON contract tests for both horizon 1 and horizon 2.
- All six online V5.1 W&B runs either completed or were stopped as requested.
- Completed a real WebSocket round trip with the finished hanging-mug raw V5.1 model: the handshake advertised horizon 1, two cameras, and 14D state; the request supplied only the current global/hand pair and returned finite `[1,30,7]` actions in about 58 ms locally.
- Made msgpack-decoded read-only NumPy images into owned writable request buffers before `torch.from_numpy`, removing the deployment warning.

## Cleanup result

- Exactly six model checkpoints remain, all named `base.pt`.
- All `.resume`, `base.best_epoch_*.pt`, and `base.pt.best.weights.pt` files were removed.
- The classify-blocks B-spline `base.pt` was verified as a `best_validation_model` with the h1, 14D-state, B-spline contract; its adjacent `base.server.json` parses successfully.
- The small `base.pt.wandb.json` and `base.server.json` files are not model copies; they are retained for run provenance and the deployment contract.
