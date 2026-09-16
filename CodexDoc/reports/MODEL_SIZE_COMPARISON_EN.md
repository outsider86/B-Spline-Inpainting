# Current Policy versus StarVLA DiT Model Size

**Report date:** 2026-09-16

## Comparison boundary

The current-policy counts include the complete trainable policy: observation/state projector, action transformer, embeddings, and output head. They exclude the frozen cached DINOv2 and SigLIP vision encoders.

The StarVLA counts below isolate its action DiT and exclude the Qwen VLM. This is the closest architecture-to-architecture comparison. The local StarVLA source is pinned at `2f17402a5ccaa09907516ae5e542b0fa6ee5d155`.

## Current policies

| Representation | Architecture | Observation/state | Action backbone | Embedding/output | Total trainable |
|---|---|---:|---:|---:|---:|
| Raw | FM | 0.502M | 3.414M | 0.084M | **4.000M** |
| Raw | Layerwise DD | 0.502M | 3.414M | 0.139M | **4.055M** |
| Raw | Joint DD | 0.502M | 3.117M | 0.139M | **3.757M** |
| B-spline | FM | 0.502M | 3.414M | 0.082M | **3.997M** |
| B-spline | Layerwise DD | 0.502M | 3.414M | 0.123M | **4.039M** |
| B-spline | Joint DD | 0.502M | 3.117M | 0.123M | **3.741M** |

All current policies use six transformer layers with hidden width 192 and six attention heads.

For a strict action-head-only comparison, remove the 0.502M observation/state projector. The six current action heads are:

| Representation | Architecture | Current action head |
|---|---|---:|
| Raw | FM | **3.498M** |
| Raw | Layerwise DD | **3.553M** |
| Raw | Joint DD | **3.256M** |
| B-spline | FM | **3.496M** |
| B-spline | Layerwise DD | **3.537M** |
| B-spline | Joint DD | **3.239M** |

## StarVLA DiT-S/B/L presets

StarVLA's legacy `DiTActionHeader` defines three explicit presets. The following exact counts were obtained from the pinned constructors using the same 7-D action and 30-step action horizon as the current experiments. The diffusion scheduler has no learned parameters, so each number is also the complete trainable action-head count for this implementation.

| StarVLA preset | Shape | Action-head parameters | Versus current 3.239–3.553M heads | BF16 weights |
|---|---|---:|---:|---:|
| DiT-S | 6 layers, width 384, 4 heads | **11.101M** | **3.12–3.43× larger** | 21.2 MiB |
| DiT-B | 12 layers, width 768, 12 heads | **86.535M** | **24.36–26.71× larger** | 165.1 MiB |
| DiT-L | 24 layers, width 1024, 16 heads | **304.759M** | **85.77–94.08× larger** | 581.3 MiB |

Thus the present policies are smaller than even DiT-S. Architecturally, they are closest to a custom extra-small preset: the same six-layer depth as DiT-S, but half its width (192 versus 384).

## StarVLA references

The newer QwenPI family below does not use the legacy S/B/L shapes directly; these figures should be treated as a separate StarVLA action-head family.

| StarVLA variant | Shape | DiT-only parameters | Relative to current 3.74–4.05M policies |
|---|---|---:|---:|
| QwenPI_v3 compressed action DiT | 36 layers, width 1024, 16 heads | **532.326M** | **131–142× larger** |
| QwenPI_v3 complete action model | DiT plus action encoders/decoder | **538.678M** | **133–144× larger** |
| QwenPI / QwenDiscrete action DiT | 36 layers, width 2048, 32 heads | **2.128B** | **525–569× larger** |

The QwenPI_v3 source also reports 94.593M parameters for its per-layer VLM-to-DiT projectors and 5.071B parameters for the complete Qwen3-VL-4B system. Those are outside the DiT-only comparison.

## Weight-memory interpretation

- Current policy weights: approximately **14.3–15.5 MiB in FP32** or **7.1–7.7 MiB in BF16**.
- StarVLA compressed DiT: approximately **1.98 GiB in FP32** or **1.00 GiB in BF16**.
- StarVLA 2048-wide DiT: approximately **7.93 GiB in FP32** or **3.96 GiB in BF16**.

Optimizer states, gradients, activations, VLM/vision encoders, and training traces are not included in these weight-only numbers. The 45–55 MB current checkpoint files are larger than inference weights because they also store AdamW state and metadata.

## Conclusion

The present codebase intentionally implements a compact StarVLA-style policy, not a parameter-matched StarVLA DiT. Its total trainable policy is less than 1% of even the compressed QwenPI_v3 action model. This makes the current results useful for architecture and action-representation comparisons, but not a controlled model-capacity comparison against full StarVLA.
