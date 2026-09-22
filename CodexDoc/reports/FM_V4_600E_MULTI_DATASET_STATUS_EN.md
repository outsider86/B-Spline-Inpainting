# V4 Flow-Matching Multi-Dataset Training Status

Date: 2026-09-22 UTC

Six batch-64 h2 Flow-Matching runs are active with complete validation after
every loader epoch. Each dataset has raw and B-spline variants; each example
uses two timesteps and two cameras with scratch ResNet-18 encoders.

| Dataset | Raw GPU / W&B | B-spline GPU / W&B | Updates per epoch | Full validation samples |
|---|---|---|---:|---:|
| stacking_cup_30hz | 2 / `zjesf7u7` | 3 / `9zt4k9ha` | 636 | 4,537 |
| classify_blocks_30hz | 0 / `twqqp4w4` | 1 / `rniugssl` | 1,379 | 9,673 |
| hanging_mug_30hz | 4 / `erubna5k` | 5 / `aqxr7k6b` | 417 | 2,979 |

The old stacking runs with 10-epoch validation were stopped and preserved.
The fresh runs validate every epoch and retain exact snapshots every 100
epochs. GPU 6 remains free.

Both hanging-mug variants are complete at update 41,700 (epoch 100). Raw
selected update 33,360 with validation action MSE 0.0150083; B-spline selected
update 41,283 with 0.0152258. Both validation-best bases and epoch-100 snapshots
clean-load. GPUs 4/5 are free.

Model checkpoint artifacts are local-only under scratch. W&B records metrics
and summaries but no longer stages or uploads model artifacts.

The classify B-spline throughput issue is a CPU input-pipeline bottleneck, not
GPU throttling: its compressed per-episode target cache averages 3.32 MiB,
versus 0.256 MiB for raw. Random window shuffling with a three-episode worker
cache repeatedly decompresses these files; GPU 1 consequently spends long
periods idle while its eight loader workers are CPU-busy.

Next: verify the automatic hanging finalization and continue monitoring the
dense validation curves for the requested early-stop decision.
