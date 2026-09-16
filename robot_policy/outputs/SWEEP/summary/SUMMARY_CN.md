# 36 个检查点综合汇总

本报告由本目录中的重新评测结果生成，日期为 2026-09-16。

## 评测协议

- 36 个检查点均使用相同的 3,149 个测试窗口（5 个测试 episode）与随机种子 20260915。这里报告的是开环动作误差，不是任务成功率。
- 主要质量指标来自解码后的 30×7 轨迹：归一化 MSE 用于公平比较动作表示；物理 MSE 使用训练集 q01/q99 还原关节与夹爪单位。
- 默认采样：FM 12 步、layerwise diffusion 8 轮、joint diffusion 8 轮并启用融合的跨 block K/V cache 转换。
- 延迟在 RTX PRO 6000 Blackwell 上以 batch 1 测量：10 次预热、100 次同步计时。`estimated online p50` 是在线视觉、投影器、策略采样和动作解码各自 p50 的加和，并非直接测得的端到端分位数。

## 主要结论

- 最佳物理 MSE 为 **0.009896**：DiT-L / bspline / fm / base。
- 在 18 组严格配对中，B-spline 有 **10/18** 组优于 raw；物理 MSE 降幅中位数为 **3.94%**（负值表示变差）。
- ttRTC 在零延迟下有 **4/18** 组改善；ttRTC 的核心目标是延迟重规划，因此不能只凭零延迟指标判断。
- 在 d=0…10 的平均延迟 MSE 上，B-spline 有 **9/18** 组优于 raw，中位降幅为 **-2.33%**。
- ttRTC 在平均延迟 MSE 上仅有 **4/18** 组改善，中位降幅为 **-6.11%**；当前固定微调协议不能稳定提升不同容量模型的延迟鲁棒性。
- 参考 `dd-openvla` D2F 实现的融合 cache 转换在 **12/12** 个真实检查点上与原缓存采样器逐 token 完全一致；p50 加速中位数 **8.65%**，范围 5.29%–12.48%。

## B-spline 解码正确性

- 18 个 B-spline 检查点的 manifest 均与重新生成的 encoder 类型、配置、tokenizer `3282a6009a52b9a0` 和 calibration 完全一致。
- 解码顺序：离散 token 反量化（仅离散策略）→ 固定 30×18 basis × 18×7 控制点 → 30×7 归一化动作 → 物理尺度还原。
- 全部测试窗口中，连续重算与缓存重建的最大差值为 **9.934e-08**；离散 token 重算的最大差值为 **5.959e-08**。
- basis 秩为 18，行和最大误差为 1.110e-16。basis 图使用 SciPy 在连续时间上密集计算 1,201 个点，并以圆点叠加解码器实际使用的 30 个离散采样；离散采样与解码矩阵的最大差值为 5.551e-17。

## 全部检查点

| size | repr. | policy | stage | M params | val action MSE | test norm. MSE | test physical MSE | sample p50 ms | estimated online p50 ms | W&B |
|---|---|---|---|---|---|---|---|---|---|---|
| DiT-S | raw | fm | base | 15.07 | 0.02363 | 0.03678 | 0.01102 | 15.66 | 26.49 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9p5hb9y) |
| DiT-S | raw | fm | ttrtc | 15.07 | 0.02645 | 0.03708 | 0.01115 | 16.06 | 26.90 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/v9ubgo5v) |
| DiT-S | raw | layerwise | base | 15.04 | 0.00556 | 0.04564 | 0.01411 | 12.90 | 23.73 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/zh84gs3w) |
| DiT-S | raw | layerwise | ttrtc | 15.04 | 0.00624 | 0.05046 | 0.01606 | 12.31 | 23.13 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/2o2kthwz) |
| DiT-S | raw | joint | base | 13.85 | 0.01256 | 0.06101 | 0.01849 | 146.18 | 157.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ykwmgupy) |
| DiT-S | raw | joint | ttrtc | 13.85 | 0.00699 | 0.06588 | 0.01984 | 144.53 | 155.37 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/dgxz1ipu) |
| DiT-S | B-spline | fm | base | 15.07 | 0.03394 | 0.03622 | 0.01089 | 15.04 | 25.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/2rb15m2t) |
| DiT-S | B-spline | fm | ttrtc | 15.07 | 0.03566 | 0.03806 | 0.01158 | 15.63 | 26.49 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/83pdq57s) |
| DiT-S | B-spline | layerwise | base | 15.00 | 0.00291 | 0.08652 | 0.02938 | 12.01 | 22.86 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/ya0xdf4n) |
| DiT-S | B-spline | layerwise | ttrtc | 15.00 | 0.00377 | 0.09377 | 0.03349 | 12.55 | 23.39 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/q28aplkd) |
| DiT-S | B-spline | joint | base | 13.82 | 0.01093 | 0.05121 | 0.01580 | 78.48 | 89.32 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0q4g7cq4) |
| DiT-S | B-spline | joint | ttrtc | 13.82 | 0.01367 | 0.06052 | 0.01850 | 79.84 | 90.71 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/340xpp9a) |
| DiT-B | raw | fm | base | 108.06 | 0.02567 | 0.03672 | 0.01083 | 28.84 | 39.66 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/op6lpqi3) |
| DiT-B | raw | fm | ttrtc | 108.06 | 0.02561 | 0.03663 | 0.01100 | 28.80 | 39.64 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/duznrj8t) |
| DiT-B | raw | layerwise | base | 107.39 | 0.51757 | 0.49172 | 0.21971 | 28.46 | 39.30 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o2g40xq8) |
| DiT-B | raw | layerwise | ttrtc | 107.39 | 0.51570 | 0.49172 | 0.21970 | 28.53 | 39.40 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ajt8p9vs) |
| DiT-B | raw | joint | base | 102.68 | 0.03297 | 0.14679 | 0.07543 | 237.33 | 248.52 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/x9qrzmwd) |
| DiT-B | raw | joint | ttrtc | 102.68 | 0.03654 | 0.19250 | 0.06061 | 233.78 | 245.09 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/1hux9mwd) |
| DiT-B | B-spline | fm | base | 108.05 | 0.02710 | 0.03845 | 0.01133 | 30.19 | 41.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/0c77dslj) |
| DiT-B | B-spline | fm | ttrtc | 108.05 | 0.02993 | 0.03931 | 0.01207 | 30.19 | 41.01 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/25fzgt8t) |
| DiT-B | B-spline | layerwise | base | 107.33 | 0.19170 | 0.42273 | 0.19773 | 25.21 | 36.03 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i40wvlpn) |
| DiT-B | B-spline | layerwise | ttrtc | 107.33 | 0.17481 | 0.42273 | 0.19773 | 24.77 | 35.57 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/wjpva0fp) |
| DiT-B | B-spline | joint | base | 102.62 | 0.01815 | 0.05071 | 0.01840 | 131.14 | 141.97 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tsg4colk) |
| DiT-B | B-spline | joint | ttrtc | 102.62 | 0.02253 | 0.04998 | 0.01695 | 131.74 | 142.60 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/jjch9idz) |
| DiT-L | raw | fm | base | 367.58 | 0.02927 | 0.03586 | 0.01073 | 55.99 | 66.83 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/tziiazri) |
| DiT-L | raw | fm | ttrtc | 367.58 | 0.02781 | 0.03612 | 0.01090 | 55.97 | 66.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/redv2qv4) |
| DiT-L | raw | layerwise | base | 366.17 | 0.04226 | 0.04782 | 0.01460 | 64.23 | 75.04 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/o0tj6klu) |
| DiT-L | raw | layerwise | ttrtc | 366.17 | 0.03814 | 0.04828 | 0.01446 | 63.80 | 74.62 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/7ejattim) |
| DiT-L | raw | joint | base | 357.84 | 0.05200 | 0.07649 | 0.02594 | 532.13 | 543.44 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/kvby9apd) |
| DiT-L | raw | joint | ttrtc | 357.84 | 0.05068 | 0.10409 | 0.03393 | 533.41 | 544.78 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/0i7ac2b6) |
| DiT-L | B-spline | fm | base | 367.57 | 0.02596 | 0.03251 | 0.00990 | 53.85 | 64.74 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/hso7oc5a) |
| DiT-L | B-spline | fm | ttrtc | 367.57 | 0.02935 | 0.03538 | 0.01115 | 53.77 | 64.65 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/s56r3udm) |
| DiT-L | B-spline | layerwise | base | 366.08 | 0.18239 | 0.36524 | 0.17671 | 46.81 | 57.59 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/i9zgz3hh) |
| DiT-L | B-spline | layerwise | ttrtc | 366.08 | 0.17527 | 0.36524 | 0.17671 | 46.83 | 57.66 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/pgnhmdfl) |
| DiT-L | B-spline | joint | base | 357.76 | 0.01948 | 0.06627 | 0.02239 | 309.14 | 320.50 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/us6r4jrf) |
| DiT-L | B-spline | joint | ttrtc | 357.76 | 0.02368 | 0.06986 | 0.02332 | 310.01 | 321.11 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/ra5ijsh8) |

## 文件索引

- `checkpoint_metrics.csv`：36 个检查点的训练、测试、延迟、显存、平滑性、解码和 W&B 字段。
- `pairwise_effects.csv`：B-spline/raw 与 ttRTC/base 严格配对差异。
- `summary.json`：完整机器可读汇总。
- `performance_vs_latency.png`、`physical_mse_matrix.png`、`joint_cache_acceleration.png`：可视化比较。
- `decoder_validation/`：B-spline 数值解码审计与图形。

## 限制

数据集只提供绝对关节与夹爪量，没有经过标定的笛卡尔位姿，因此不能可靠报告位置/旋转误差。速度和加速度仅描述开环预测轨迹平滑性，不代表机器人真实执行动力学。本报告不声称仿真器或真实机器人任务成功率。
