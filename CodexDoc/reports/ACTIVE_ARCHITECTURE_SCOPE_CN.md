# 当前架构范围：FM 与 Joint DD

日期：2026-09-17

## 决策

后续训练、RTC 微调、开环评估、推理时 RTC 评估、延迟测试以及汇总对比仅包含：

- `fm`：连续流匹配。
- `discrete_joint`：带分块 KV cache 的联合序列离散扩散。

`discrete_layerwise` 不再用于新的实验。其实现仅保留用于复现已有 checkpoint
和历史报告；现有本地文件及远端产物不会被删除。

## 依据

已完成的 36-checkpoint sweep 显示，六个 layerwise DiT-B/L 任务出现严重的
梯度范数溢出。Joint DD 则对应后续需要继续研究的离散生成、分块解码与 cache
路径。保留 FM 可继续提供稳定的连续生成基线。

## 实际影响

- raw/B-spline x base/ttRTC 的单组实验由 12 个变体缩减为 8 个。
- DiT-S/B/L x raw/B-spline x base/ttRTC 的容量 sweep 由 36 个 checkpoint
  缩减为 24 个。
- 历史 layerwise 结果继续作为审计证据保留，但不再纳入新的汇总比较。

## 下一步

新的实验及 W&B 对比只运行 FM 和 Joint DD。
