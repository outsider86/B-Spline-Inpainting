# Architecture decision

The shared frontend is frozen DINOv2 ViT-L/14-reg4 plus frozen SigLIP ViT-SO400M/14 at 224 px. A small trainable projector and trainable state tokenizer form `[global patches | hand patches | state]`. No language model, tokenizer, prompt, or language weights are present.

| ID | Action path | Observation interaction | Diffusion / sampling | Specialized cache |
|---|---|---|---|---|
| `fm` | 18 tokens, each a continuous 7-D control | StarVLA-style alternating cross-attention to observation and action self-attention | Beta-time flow velocity, Euler 5/8/12 (12 default) | no |
| `discrete_layerwise` | 126 scalar bin/MASK tokens | same alternating conditioned DiT interface as FM | cosine MaskGIT, irreversible reveal | no |
| `discrete_joint` | append 126 scalar bin/MASK tokens to all 33 observation tokens | one shared compact DiT sequence | monotonic block corruption and blockwise MaskGIT | exact prefix cache for obs + completed blocks |

The joint attention topology is:

```text
query \ key       observation   prior blocks   same block   future blocks
observation           yes            no           no             no
action block j        yes            yes          yes            no
padding               no             no           no             no
```

This makes observation hidden states and committed earlier blocks independent of changing later tokens. Their per-layer inputs/KV are exactly cacheable. An active block is never cached across remasking. Observation, state, episode, or model changes invalidate the cache. This is the same topology in training and inference.

## Parameter matching plan

All policies use width 192, depth 6, six heads, MLP ratio 4, identical observation projector/state tokenizer, fixed vision, and equal 256-bin output semantics. FM and layerwise discrete use the same alternating backbone and 32 learned future queries. Joint discrete uses six equivalent-width self-attention/MLP blocks plus a small residual mixer in every block. The mixer is active in both full and cached paths and closes the otherwise 18% parameter gap caused by the layerwise time/AdaLN modules without changing cache-safe attention. Its longer shared sequence still changes FLOPs despite comparable width/depth. The final trainable counts are 3,997,383 (`fm`), 4,038,528 (`discrete_layerwise`), and 3,741,120 (`discrete_joint`): a 7.4% max spread. Reports separate observation adaptation, action backbone, embedding/head, trainable total, frozen vision (730,911,680 parameters), sequence length, and measured latency.

## Intentional adaptations

- StarVLA's 7B-width/global configuration coupling is removed; the compact action network receives explicit dimensions.
- State is already an observation token and is not duplicated in a second state path.
- BSplineEncoder bins replace StarVLA's separate action binning.
- Expected bin distance replaces reference argmax MAE.
- DINO/SigLIP features use fixed 4x4 average pooling before the trainable projector. This preserves shared aligned dense regions and reduces a full-dataset cache from roughly 275 GB to roughly 4.4 GB.
