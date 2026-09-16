# Sweep 结果整理与输出清理

日期：2026-09-16

## 状态

已完成。包含 36 个检查点的 DiT-S/B/L sweep 现已统一整理到 `robot_policy/outputs/SWEEP`，并作为唯一的本地标准结果集。

## 保留内容

- 36 个最终检查点：DiT-S/DiT-B/DiT-L × raw/B-spline 六组，每组 6 个。
- 六份检查点 manifest、W&B 身份 sidecar、精简训练日志、最终调度状态和完整审计结果。
- W&B 项目 `robot-policy-50k-bs32-basic` 与 `robot-policy-5k-bs32-ttRTC` 中的在线历史及模型 artifact。
- 兼容符号链接 `outputs/model_size_sweep_50k_bs32 -> SWEEP`，用于保持检查点内不可变的绝对父模型路径可解析；该链接不会复制数据。

## 验证结果

- `completion_audit.json`：通过，预期 36 / 找到 36，错误数为 0。
- W&B：36 个唯一 run ID，严格满足 18 个 base / 18 个 ttRTC。
- 清理后 SHA-256 复核：36 个检查点全部匹配，0 个不一致。
- 六个容量/动作表示分组均恰好包含 6 个最终检查点。
- 已无 `.pt.resume` 快照和本地 W&B 缓存目录。

## 已删除内容

本次永久删除了 36 个冗余 resume 快照、六个本地 W&B 缓存树、过期锁文件和部分审计文件，以及十二个已被替代的顶层输出目录：`action_representation_comparison`、`bspline_reproduction`、`checkpoints`、`evaluation`、`latency`、`longrun_50k_bs32`、`pilot`、`pilot_fast`、`prepared`、`raw_actions`、`replay` 和 `visualizations`。

共释放 83,622,878,276 字节（约 77.9 GiB）；当前标准 sweep 占用 69,930,109,574 字节（约 65.1 GiB）。已删除的本地文件无法从当前工作区恢复，但 W&B 在线数据仍然保留。

## 下一步

在重训六个受数值稳定性影响的 layerwise 变体前，先实现已记录的稳定性修复并完成 DiT-B/L canary 测试；不要修改当前保留的 36-checkpoint 基线。
