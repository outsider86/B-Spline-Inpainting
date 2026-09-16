# Reproducible environment audit

Audit time: 2026-09-15 UTC.

- Host environment used: Conda `starVLA`, Python 3.10.19.
- PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, timm 1.0.25, Transformers 4.57.0.
- NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3, PyArrow 14.0.1, OpenCV 4.11.0, PyAV 12.3.0, Matplotlib 3.10.8.
- NVIDIA driver 595.71.05. Six NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, each 97,887 MiB.
- At launch GPUs 0–3 were occupied by existing Python jobs (roughly 29 GiB each); GPUs 4–5 were unoccupied and selected. Existing processes were not modified.
- A two-rank NCCL throughput pilot on GPUs 4–5 failed at process-group initialization with an illegal-memory-access error. Final jobs therefore ran as concurrent, independent one-GPU processes on physical devices 4 and 5. Effective batch 128 was held constant with gradient accumulation; no occupied GPU or unrelated process was disturbed.
- Filesystem: `/scratch`, 14 TB total, 11 TB available at audit time.
- Dataset version: LeRobot `v2.1`, Piper, 52 episodes, 31,706 frames, 104 MP4s, 30.0 Hz. Episode lengths 500–786. Timestamp deltas min 0.0333328247 s, median 0.0333333015 s, max 0.0333347321 s, with no >1 ms deviations and no nonconsecutive frame indices.

Install the package without mutating the reference repositories:

```bash
conda activate starVLA
cd /scratch/wangpc/B-Spline-Inpainting/robot_policy
pip install -e ../BSplineEncoder
pip install -e '.[test]'
```

Weights are cached outside source under `.cache/huggingface`. Public timm IDs are `vit_large_patch14_reg4_dinov2.lvd142m` and `vit_so400m_patch14_siglip_224`. Artifact manifests record those IDs, preprocessing, extraction layer and cache precision. Redistributing pretrained weights is outside this repository's deliverables and remains subject to their upstream licenses.
