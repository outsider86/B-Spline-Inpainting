# BSP U-Net V4 Flow-Matching 状态

日期：2026-09-22 UTC

## 结果

在确认此前差异来自训练期间 validation 覆盖不完整后，GPU 2/3 上的替代 batch=4
训练已经停止。随后直接对既有、已经完成的 batch=4 和 batch=64 raw/B-spline
checkpoint 执行了新的权威评估流水线。

下表所有指标均覆盖完整 split：train 28,557 windows，validation 3,149 windows。
Open-loop 为完全 from-scratch generation。RTC 从当前 GT batch 取前 6 个 action 作为
已经 committed 的 prefix：raw FM 使用前 6 个 GT action rows 做 condition；B-spline
FM 使用覆盖 3 个受影响 spans 的 6 个 GT control rows（`3 spans + cubic degree 3`）
做 condition。Base FM 使用 binary-hard-mask PiGDM；RTC MSE 只在新生成的 suffix 上
计算。

| Batch | 表示 | Split | Open-loop physical MSE | GT-prefix RTC suffix physical MSE |
|---:|---|---|---:|---:|
| 4 | raw | train | 0.00606603 | 0.00586514 |
| 4 | raw | val | 0.01132877 | 0.01151038 |
| 4 | B-spline | train | 0.00629346 | 0.00550323 |
| 4 | B-spline | val | 0.01130805 | 0.01028329 |
| 64 | raw | train | 0.00036776 | 0.00031687 |
| 64 | raw | val | 0.00945820 | 0.01051469 |
| 64 | B-spline | train | 0.00042084 | 0.00035019 |
| 64 | B-spline | val | 0.00936654 | 0.01002486 |

Batch=64 对训练集的拟合明显更强：raw 和 B-spline 的 validation/train open-loop
MSE 比值分别为 25.72×、22.26×；batch=4 分别为 1.87×、1.80×。虽然 batch=64 的
generalization gap 更大，但其 validation 绝对误差仍更低：raw/B-spline open-loop
分别低 16.51%/17.17%，GT-prefix RTC 分别低 8.65%/2.51%。

同一 batch 内，B-spline 相对 raw 的 open-loop 优势很小（batch=4 为 0.18%，
batch=64 为 0.97%），但 RTC suffix 更好：batch=4 validation MSE 低 10.66%，
batch=64 低 4.66%。

## 可视化协议与核验

对每个 batch size，分别从 train 和 validation 固定采样一个高运动量 GT batch。
每个 subplot 严格包含 ground truth、from-scratch prediction、GT-prefix RTC
prediction 三条曲线。蓝色 RTC 轨迹由“完全精确的 committed GT prefix + PiGDM
生成 suffix”拼接；四个 NPZ 中 raw/B-spline 的 prefix 都通过 bitwise equality，
所有 committed-prefix MSE 也严格为 0。raw 与 B-spline 统一使用整个数据集的
physical min/max 作为 y 轴。

结果目录：

- `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch4/`
- `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch64/`

每个目录包含 `trajectory_comparison_{train,val}.png`、对应 NPZ/JSON、完整
per-checkpoint metric JSON、`metrics.csv` 和中英文总结。

## 下一步

既有 checkpoint 中，batch=64 仍是 validation 更强的选择，但必须明确记录其很大的
train/validation gap。Prefix=6 的 RTC 协议下，B-spline 应作为优先表示。
