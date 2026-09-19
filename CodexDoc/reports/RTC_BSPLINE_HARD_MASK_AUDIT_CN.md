# B-spline RTC hard mask 审计

更新时间：2026-09-19 UTC

## 结论

continuousRTC（FM）与 discreteRTC（joint discrete diffusion）使用同一份 B-spline hard mask。对 degree 为 `p`、从左端开始受影响的 `D` 个 spans，固定的 control rows 精确为 `[0, D+p)`；当前 cubic 设置即 `[0, D+3)`。

它严格等于这些受影响 spans 中非零 basis functions 所对应 control rows 的并集。并集之外的 control rows 不会获得 previous plan 条件。

## 两种 RTC 如何执行

- 训练路径在 `rtc/training.py` 统一生成 representation-aware `fixed_mask`，再交给 FM 或 joint DD。
- FM 只覆盖 fixed rows，并把这些 rows 排除在 mutable loss 外；推理时在每个 Euler update 前后都重新施加 hard mask。
- joint DD 只插入 fixed discrete tokens，把它们排除出 corruption supervision，并在迭代 unmasking 中始终保持 immutable。
- deployment、inference、RTC evaluation 和 latency 路径现在显式传入 checkpoint/config 中的 spline geometry，不再依赖默认常数。
- deployment metadata 明确暴露 `rtc_mask_type=hard` 与 B-spline mask scope。

## Delay 映射

B-spline RTC 将 raw delay 映射为 `D = ceil(raw_delay / span_length_steps)`。当前每个 span 是两个 raw action steps，因此 odd delay 会固定整个“部分受影响”的 span。这是正确行为，因为一个 spline span 由其局部 control points 联合定义。

由于 cubic basis 支撑区间重叠，这些数学上必需的 control points 也可能影响相邻的未来 span。这是 B-spline 本身的局部支撑性质，不是额外泄漏；实现不会在受影响 span 的支撑并集之外多固定一个 control row。

## 回归验证

- 对 `D=1..5`，hard mask 已逐项与 authoritative B-spline basis 的非零列做完全相等比较。
- 对 FM 与 joint DD，任意大幅修改 exact support 之外的 prefix values，在相同随机种子下都不会改变 sampling 结果或 training loss。
- 现有 checkpoints 原本已经采用相同的 `D+3` 行为。本次修改使 geometry 来自配置、明确记录契约并加入回归保护，不需要重新训练。

## 关键文件

- `robot_policy/src/robot_policy/rtc/delay_mapping.py`
- `robot_policy/src/robot_policy/rtc/training.py`
- `robot_policy/src/robot_policy/policies/fm.py`
- `robot_policy/src/robot_policy/policies/discrete_joint.py`
- `robot_policy/tests/test_bspline_and_rtc.py`
- `robot_policy/tests/test_inference_rtc.py`

