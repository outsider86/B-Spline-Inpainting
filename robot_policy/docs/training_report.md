# Training report

All six deliverable checkpoints are genuinely trained and independently reloaded from disk. A 20-update throughput pilot exists under `outputs/pilot*` but is not counted. Validation episodes alone were used for monitoring; the saved artifacts are the final-update weights, so the manifest reports both the final and best-observed validation objectives without falsely claiming best-weight selection.

## Comparable setup

- Episode split: 42 train / 5 validation / 5 test, fixed before windowing with seed 20260915.
- Base runs: 2,000 optimizer updates, 256,000 samples seen each.
- RTC runs: 800 optimizer updates, 102,400 samples seen each.
- AdamW, learning rate 3e-4, cosine decay, 100-update warmup, weight decay 1e-4, gradient clip 1.0, BF16 autocast.
- Batch 64 with two-step gradient accumulation gives effective batch 128 for every model.
- A two-GPU NCCL pilot failed during initialization with an illegal-memory-access error on this Blackwell/PyTorch combination. No other user's process was touched. Final fairness was preserved by running independent single-GPU architecture jobs concurrently on physical GPUs 4 and 5 with the same effective batch and update budget.
- RTC delay draws use `p(D) ∝ exp(5-D)` over D=S=1..5: approximately `[0.6364, 0.2341, 0.0861, 0.0317, 0.0117]`. Exact training delays map to raw d=s=2D. Prefixes come from frozen parent-policy predictions at the corresponding earlier recorded observation, not privileged ground truth.

## Final artifacts

| Architecture | Stage | Updates | Final val objective | Best observed | Wall s | GPU h | Peak train GB | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FM | base | 2000 | 0.05923 | 0.05247 | 347.3 | 0.0965 | 0.304 | 3,997,383 |
| FM | RTC | 800 | 0.07753 | 0.06694 | 152.6 | 0.0424 | 0.372 | 3,997,383 |
| discrete layerwise | base | 2000 | 16.47439 | 16.47439 | 345.7 | 0.0960 | 0.655 | 4,038,528 |
| discrete layerwise | RTC | 800 | 16.15584 | 16.09588 | 159.0 | 0.0442 | 0.724 | 4,038,528 |
| discrete joint | base | 2000 | 12.59834 | 12.36003 | 348.0 | 0.0967 | 0.724 | 3,741,120 |
| discrete joint | RTC | 800 | 14.28329 | 14.28329 | 602.7 | 0.1674 | 0.791 | 3,741,120 |

The maximum trainable-parameter spread is 7.4%. Joint RTC costs more wall time because each training condition obtains a prior prediction through blockwise joint sampling. The authoritative hashes, exact configurations, parent hashes, histories, hardware accounting, and resume commands are in `outputs/checkpoints/checkpoint_manifest.json`.

The tiny-subset acceptance diagnostic in `outputs/evaluation/tiny_overfit.json` uses four real train windows and reduces objective ratios to 0.111 (FM), 0.00077 (layerwise discrete), and 0.0333 (joint discrete). Those diagnostic weights are not saved or represented as final models.
