# V4Full、V5、V5.1 的 5-epoch Training-time RTC

更新时间：2026-09-23 UTC

## 完成情况

三个版本的全部 policy 均已完成 ttRTC：每版包含 classify-blocks、
hanging-mug、stacking-cup 三个任务，以及 raw、B-spline 两种动作表示，
合计 18 个 checkpoint。这里的 V4 明确使用 `BSP_UNET_V4_FULL` 中的六个
模型，而不是早期 `BSP_UNET_V4` 中的两个模型。

最终目录：

- `robot_policy/outputs/V4/{task}/{raw,bspline}/ttrtc.pt`
- `robot_policy/outputs/V5/{task}/{raw,bspline}/ttrtc.pt`
- `robot_policy/outputs/V51/{task}/{raw,bspline}/ttrtc.pt`

## ttRTC 训练语义

- 只载入 parent 的模型权重；optimizer 和 EMA 从头建立，不载入 base 的
  optimizer state。
- 每个 batch 随机模拟 delay，并用 GT action/control points 构造 hard
  prefix。
- prefix 的 flow time 固定为 `t=1`；suffix 使用随机采样的 flow time。
- loss mask 排除全部 prefix 坐标，只在 suffix 上反向传播。
- 每个模型训练 5 epoch，每个 epoch 在完整 validation split 上评估一次，
  最后发布五次 validation 中 action_mse 最低的模型。

## 审计结论

- 18/18 训练成功，0 个失败。
- 18/18 均完成 5 次全量 validation。
- 18/18 parent SHA-256 与训练前一致。
- 18/18 checkpoint 可严格加载，且部署 JSON 与模型合同一致。
- 所有运行均为 0 skipped optimizer update。
- 已删除 optimizer recovery、周期 snapshot 和 rolling best 临时文件，只保留
  `ttrtc.pt`、W&B provenance、部署 JSON、日志和审计 metadata。

## 发布与直接推理

- 18 个 `ttrtc.pt` 已上传到 Hugging Face 数据集
  `DiscreteRTC/dRTC` 的对应 `NewModel/v4Full`、`NewModel/V5Full` 和
  `NewModel/V5_1Full` leaf；没有覆盖任何 `base.pt`。
- 远端 18 个文件的大小和 LFS SHA-256 均与本地逐项一致；发布后的仓库
  revision 为 `c47d9f5cb4394d1ce86ea4bca54f2965db44df1a`。
- ttRTC FM 推理不使用 PiGDM/VJP，但仍执行多步 ODE/Euler integration，而
  不是单次网络 forward。每步将 prefix time 固定为 `t=1`，suffix time 设为
  当前 `t=i/N`，并在更新前后 hard-clamp prefix。部署采样代码和回归测试已
  按该训练时异步 time map 对齐。

完整指标表见 `robot_policy/outputs/TTRTC_5E_SUMMARY.md`，机器可读审计见
`robot_policy/outputs/TTRTC_5E_AUDIT.json`。

## 下一步

对 18 组 base/ttRTC 配对执行统一的 GT-prefix RTC suffix evaluation；训练时
用于选 best 的 validation/action_mse 是 generation-from-scratch 指标，不能代替
RTC suffix quality。
