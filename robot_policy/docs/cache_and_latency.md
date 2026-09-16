# Cache correctness and inference latency

Timing uses batch size one on an NVIDIA RTX PRO 6000 Blackwell, compilation disabled, automatic PyTorch SDPA/MultiheadAttention backends, 10 warmups and 100 measured iterations with CUDA synchronization. Policy inference is FP32; the frozen vision frontend uses BF16 autocast. All raw JSON includes p50/p95/p99 and peak memory.

The online dual-camera frontend costs about 10.7–11.1 ms p50, of which about 0.107 ms is image preprocessing and about 11.0 ms is DINOv2+SigLIP encoding/fusion/pooling. Cached-feature projector/state processing is about 0.115 ms, action decoding about 0.019 ms, and RTC shift/refit/mask scheduling about 0.091 ms. Peak benchmark memory is about 3.01 GB including both frozen vision models.

| Path | Default setting | Calls | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| FM base | 12 Euler | 12 | 15.09 | 15.19 |
| FM RTC | 12 Euler | 12 | 15.05 | 15.28 |
| layerwise base | 8 rounds | 8 | 12.00 | 12.45 |
| layerwise RTC | 8 rounds | 8 | 12.00 | 12.21 |
| joint base full | 8×6 blocks | 48 | 83.34 | 83.99 |
| joint base cached | 8×6 + 6 commits | 54 | 85.30 | 85.88 |
| joint RTC full | 8×6 blocks | 48 | 83.37 | 84.56 |
| joint RTC cached | 8×6 + 6 commits | 54 | 83.86 | 84.28 |

The exact projected K/V cache stores observation plus completed blocks; active/remasked tokens are recomputed. A completed block is evaluated once with its final token IDs before its K/V becomes reusable. Observation/state/model identity changes reject reuse, and the plan-local cache disappears on plan completion or episode reset.

Correctness checks cover per-layer suffix logits (`atol=rtol=2e-5` on CPU), three remasking rounds, all six block transitions, immutable prefixes, changed-observation rejection, and cache-disabled versus cached checkpoint sampling. Both measured checkpoints return exactly identical integer tokens (`max_token_difference=0`).

For this compact width-192 batch-one model, cache setup/commit overhead dominates at four rounds and remains slightly slower at eight. At 12 rounds it becomes modestly beneficial: joint base drops from 125.73 to 123.73 ms p50 and joint RTC from 126.16 to 124.61 ms. This crossover is reported directly; no theoretical speedup is substituted for measurement.
