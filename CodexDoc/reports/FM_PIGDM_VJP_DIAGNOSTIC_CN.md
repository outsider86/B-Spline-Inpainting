# FM PiGDM VJP 逐步诊断

日期：2026-09-22 UTC

## 结论

用户对 batch=64 可视化的判断是正确的：当前 base-FM PiGDM 在 suffix 上几乎等价于
`GT prefix + from-scratch suffix` 的直接拼接。它并非完全没有计算 VJP，而是 endpoint
VJP 从 conditioned coordinates 向 mutable suffix 的传播增益太小，因此最终变化在原始
轨迹图的尺度上基本不可见。

## 与 Kinetix 的公式核对

当前实现逐项对应参考实现：

`x1 = x_t + (1-t) v(x_t,t)`

`error = hard_mask * (condition - x1)`

`correction = J_x1(x_t)^T error`

`x_next = x_t + dt * (v + guidance_weight * correction)`

Guidance coefficient 也与 Kinetix 相同，`max_guidance_weight=5`。独立线性模型测试验证了
PyTorch 实现与 Kinetix/JAX 标量转写逐步一致，因此当前现象不是明显的符号、mask 或
Euler 公式错误。

## Batch=64 的实测结果

使用 validation 中相同的 4 个高运动量 GT chunks、prefix=6、paired initial noise：

| 表示 | VJP fixed gain / condition error | VJP cross gain / condition error | 平均 mutable guidance update RMS | 最终 RTC − naive stitch suffix RMS | Scratch suffix GT MSE | RTC suffix GT MSE |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.02045 | 0.00925 | 0.000863 | 0.001597 | 0.027876 | 0.027646 |
| B-spline | 0.02698 | 0.01918 | 0.001902 | 0.003117 | 0.029731 | 0.029575 |

也就是说，condition error 经 VJP 传播到 mutable suffix 后只剩约 0.9%（raw）或 1.9%
（B-spline）的 RMS 增益。最终 PiGDM suffix 相对 naive stitch 的变化只有 0.0016 / 0.0031
physical RMS；在该 cohort 上对 GT MSE 的改善也只有约 0.83% / 0.53%。

## 为什么会这样

Hard mask 本身只在 prefix coordinates 上提供 error。Suffix 能被改变，完全依赖
denoised endpoint `x1` 对 mutable state 的 cross-Jacobian。对训练充分的 batch=64
checkpoint，`x1` 对输入噪声/state 的 Jacobian 很弱；identity 项只直接修正 fixed
coordinates，跨时间传播到 suffix 的部分非常小。因此 VJP 确实每步被计算，但其实际
作用接近 no-op。

这也解释了为什么 batch=4 checkpoint 的早期 trace 更明显：它的 guidance update 约为
普通 Euler velocity update 的 12%，而 batch=64 只有约 1.3%（raw）和 2.3%
（B-spline）。较弱/欠拟合的 flow endpoint 仍保留更强的 state Jacobian，并不代表其
基础生成质量更好。

## 产物

- Raw batch=64：
  `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch64/vjp_diagnostic/fm_raw_h2/`
- B-spline batch=64：
  `robot_policy/outputs/BSP_UNET_V4/summary/authoritative_fullsplit/batch64/vjp_diagnostic/fm_bspline_h2/`

每个目录包含：

- `vjp_trace.json` / `vjp_trace.csv`：12 个 Euler step 的完整数值；
- `vjp_trace.png`：guidance coefficient、fixed/mutable VJP update、解码后的 prefix/suffix
  单步变化、condition residual；
- `rtc_vs_naive_stitch.png`：直接显示 PiGDM RTC 与 naive stitch 的差；
- `paired_trajectories.npz`：GT、from-scratch、naive stitch、PiGDM RTC 和差值。

## 下一步

不要把当前 batch=64 base-FM PiGDM 称为有效 RTC。下一步应在同一个 paired cohort 上
做 guidance strength / flow-step sensitivity，并优先验证 ttRTC finetune；只有当 suffix
相对 naive stitch 出现稳定、可量化的改变且 GT suffix error 改善时，才将其作为 RTC
部署方案。
