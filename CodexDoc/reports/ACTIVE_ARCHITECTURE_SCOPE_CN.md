# 当前研究范围：DiT-S/B 上的 FM 与 Joint DD

日期：2026-09-18

## 决策

后续训练、RTC 微调、开环评估、推理时 RTC 评估、延迟测试以及汇总对比仅包含：

- `fm`：连续流匹配。
- `discrete_joint`：带分块 KV cache 的联合序列离散扩散。
- 容量配置仅保留 DiT-S 与 DiT-B。

`discrete_layerwise` 不再用于新的实验。其实现仅保留用于复现已有 checkpoint
和历史报告；现有本地文件及远端产物不会被删除。

DiT-L 同样退出当前研究范围。其配置、已完成 checkpoint、被中断的部分输出和
历史报告都会保留，但主动训练与评测入口会将其视为 legacy/load-only 并拒绝运行。

## 依据

已完成的 36-checkpoint sweep 显示，六个 layerwise DiT-B/L 任务出现严重的
梯度范数溢出。Joint DD 则对应后续需要继续研究的离散生成、分块解码与 cache
路径。保留 FM 可继续提供稳定的连续生成基线。

2026-09-18 按研究范围决策停止全视觉 DiT-L 扩展。后续保留 DiT-S/B 的小型/基础
容量对比，避免在后续研究中继续承担显著更高的 DiT-L 计算成本。

## 实际影响

- raw/B-spline x base/ttRTC 的单组实验由 12 个变体缩减为 8 个。
- 当前 DiT-S/B x raw/B-spline x FM/joint-DD x base/ttRTC 全视觉矩阵共
  16 个 checkpoint。
- 历史 layerwise 结果继续作为审计证据保留，但不再纳入新的汇总比较。
- 历史 DiT-L 结果继续作为归档证据保留，但不再调度，也不纳入新的主动汇总。

## 下一步

完成并比较 DiT-S/B，且仅运行 FM 与 Joint DD。
