# B-Spline Inpainting Robot Policy

Modular robotics-policy code for preprocessing, training, ttRTC fine-tuning,
open-loop evaluation, and Piper-compatible deployment. The implementation
supports raw actions and smooth cubic B-spline actions across FM, layerwise
discrete diffusion, and joint block diffusion policy heads.

The primary package is [`robot_policy`](robot_policy/README.md). Deployment
instructions are in
[`robot_policy/deployment/README.md`](robot_policy/deployment/README.md).

Clone with the pinned encoder:

```bash
git clone --recurse-submodules https://github.com/outsider86/B-Spline-Inpainting.git
```

Datasets, pretrained caches, W&B runtime files, and multi-gigabyte checkpoints
are intentionally excluded from Git. Their local layout and measured results
are documented in `CodexDoc` and the lightweight SWEEP summaries.
