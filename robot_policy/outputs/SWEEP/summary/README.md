# SWEEP Evaluation Summary

- [English comprehensive report](SUMMARY_EN.md)
- [中文综合报告](SUMMARY_CN.md)
- [Machine-readable summary](summary.json)
- [All checkpoint metrics](checkpoint_metrics.csv)
- [Pairwise effects](pairwise_effects.csv)
- [B-spline decode validation](decoder_validation/decode_validation.json)
- [Dense cubic B-spline basis visualization](decoder_validation/bspline_basis.png)
- [dd-openvla acceleration audit](REFERENCE_ACCELERATION_AUDIT.md)
- [Performance/latency frontier](performance_vs_latency.png)
- [Physical-MSE matrix](physical_mse_matrix.png)
- [Joint-cache acceleration](joint_cache_acceleration.png)
- [Deployment validation report](deployment/README.md)
- [Deployment validation data](deployment/deployment_validation.json)
- [Piper client statistics](deployment/piper_dataset_statistics.json)

Raw per-checkpoint evidence is retained in `open_loop/`, `latency/`, `latency_baseline/`, and `rtc/` when delay evaluation is present. Regenerable evaluation caches are under `cache/`.
