# FM V5 state + previous-command training status

Updated: 2026-09-22 UTC

## Input contract

V5 changes only the low-dimensional input of the V4 BSP Flow-Matching policy:

- `observation.state[0:7]`: current measured state;
- `observation.state[7:14]`: previous command;
- h2 still consumes two timestamps and two cameras per timestamp (four images);
- each camera still uses an independent scratch ResNet-18;
- only raw FM and B-spline FM are trained.

The per-timestep observation width changes from `2*64+7=135` to
`2*64+14=142`; the h2 global condition changes from 270D to 284D.

## Data and splits

| Task | Dataset | Train/val episodes | Train/val windows | Updates/epoch |
|---|---|---:|---:|---:|
| classify blocks | `Data/Processed/classify_blocks_30hz_cleanup` | 45 / 5 | 86,561 / 9,493 | 1,352 |
| hanging mug | `Data/Processed/hanging_mug_30hz_cleanup` | 55 / 6 | 24,150 / 2,738 | 377 |
| stacking cup | `Data/Processed/stacking_cup_30hz_cleanup` | 55 / 6 | 37,780 / 4,245 | 590 |

No test split is used. The action sidecars, 14D state normalization, and RGB84
caches are complete and verified for all three datasets.

## Training and checkpoint protocol

- 100 loader epochs; batch size = effective batch size = 64;
- generation-from-scratch `action_mse` over the complete validation split each
  epoch;
- retain the exact per-epoch best EMA state in memory;
- every ten epochs write the best observed state through that boundary as a
  standalone `base.best_epoch_010.pt`, ..., `base.best_epoch_100.pt`;
- final `base.pt` also contains the globally best validation weights;
- W&B records metrics/summary only and uploads no model artifacts.

## Six authoritative runs

Shared W&B project: `robot-policy-bsp-unet-v5-state-command-100e`

| GPU | Task | Representation | W&B run ID |
|---:|---|---|---|
| 0 | classify blocks | raw | `to20cz26` |
| 1 | classify blocks | B-spline | `8348qdwj` |
| 2 | hanging mug | raw | `rwnzg4hv` |
| 3 | hanging mug | B-spline | `gb0et8i7` |
| 4 | stacking cup | raw | `v3s5a0an` |
| 5 | stacking cup | B-spline | `6fjdzy3p` |

Runtime truth is recorded in
`robot_policy/outputs/BSP_UNET_V5_STATUS.json`.

The interrupted local training checkpoints/logs/W&B runtime files were removed
as requested while verified prepared/RGB caches were retained. All six runs in
the table restarted together from update zero.

## Verification completed

- Dataset audits and 14D full-size forward/loss checks passed for all configs.
- 51 relevant regression tests passed.
- A full-size end-to-end GPU smoke verified training, epoch-10 best saving,
  standalone checkpoint loading, 14D deployment input, and `[B,30,7]` action
  output; its temporary files were then deleted.

## Next

1. Complete all six 100-epoch runs.
2. Verify ten epoch-numbered best models plus final `base.pt` per run.
3. Report final/best validation generation `action_mse` and selected epoch for
   all six models.
4. Remove completed optimizer recovery files while retaining model checkpoints
   and audit metadata.

## Hanging-mug completion and cleanup

- Raw completed 100/100 epochs and selected epoch 100 with validation action
  MSE `0.0135608550`.
- B-spline completed 100/100 epochs and selected epoch 68 with validation
  action MSE `0.0130568118`.
- After verification, the epoch-10...100 and rolling-best model checkpoints
  were deleted as requested. Each directory retains only its best `base.pt`
  model plus small W&B/manifest audit metadata.

## Stacking-cup completion and cleanup

- Raw completed 100/100 epochs and selected epoch 98 with validation action
  MSE `0.0185145810`.
- B-spline completed 100/100 epochs and selected epoch 49 with validation
  action MSE `0.0188139177`.
- After verification, the epoch-10...100 and rolling-best model checkpoints
  were deleted as requested. Each directory retains only its best `base.pt`
  model plus small W&B/manifest audit metadata.
