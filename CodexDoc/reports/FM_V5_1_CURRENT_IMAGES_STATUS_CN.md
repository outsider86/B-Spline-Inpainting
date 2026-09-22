# Flow Matching V5.1：仅当前时刻双相机输入

更新时间：2026-09-22 UTC

## 目标与 V5 的唯一区别

- V5：前一时刻与当前时刻，各有 global/hand 两张图，共 4 张图；`observation_horizon=2`。
- V5.1：只使用当前时刻的 global/hand 两张图；`observation_horizon=1`。
- 两版都使用 14D state：`measured_state[0:7] + previous_command[7:14]`。
- 动作目标、train/validation episode 划分、batch size 64、学习率、100 epoch、每 epoch 全量 validation、每 10 epoch 固化 validation-best 的规则均不变。
- 视觉编码器仍是两个独立、从零训练的 ResNet-18 + SpatialSoftmax；没有切换到 DINOv2。

## 缓存与输出隔离

V5.1 复用 V5 的只读 action target、normalization、split 和 RGB cache，因为这些内容不依赖 observation horizon。V5.1 的 checkpoint、日志、W&B runtime 和临时恢复文件写入独立的 `BSP_UNET_V5_1_*_100E` 目录，不会覆盖 V5。

## 最终训练状态

训练已按用户要求停止，目前没有训练进程。六个任务均已发布为 `base.pt`：其中五个完成 100 epoch；最后一个 classify-blocks B-spline 在 epoch 80 checkpoint 落盘后停止，并保留截至该时刻全量 validation 最优的权重。

| 任务 | 表示 | 训练到 | 最优权重 update | validation/action_mse |
|---|---|---:|---:|---:|
| classify_blocks | raw | 100 epoch | 70304 | 0.0285403539 |
| classify_blocks | B-spline | 80 epoch | 56784 | 0.0268160413 |
| hanging_mug | raw | 100 epoch | 30160 | 0.0138457487 |
| hanging_mug | B-spline | 100 epoch | 33930 | 0.0132550516 |
| stacking_cup | raw | 100 epoch | 20650 | 0.0191705185 |
| stacking_cup | B-spline | 100 epoch | 31270 | 0.0186288262 |

这些 run 均位于原 V5 W&B project `robot-policy-bsp-unet-v5-state-command-100e`，且没有上传 W&B artifact。

权威机器状态文件：`robot_policy/outputs/BSP_UNET_V5_1_STATUS.json`。

## 统一部署接口

每个最终模型会形成一对文件：

- `base.pt`：模型权重及内嵌训练配置；
- `base.server.json`：部署参数及期望的 architecture、raw/B-spline、observation horizon、14D state 和相机顺序。

启动方式：

```bash
export CKPT=/absolute/path/to/base.pt
export POLICY_CONFIG=/absolute/path/to/base.server.json
CUDA_VISIBLE_DEVICES=0 robot_policy/deployment/run_policy_server.sh
```

同一个 server 入口支持 V5 和 V5.1。如果模型与 JSON 不匹配，server 会拒绝启动，避免把 V5 的四图输入合同和 V5.1 的两图输入合同混用。

## 已完成

- 已实现 V5.1 六套配置并验证 `observation_horizon=1`、`state_dim=14`。
- 已验证 V5.1 复用缓存完整且不写入 home。
- 已生成六个 V5 模型对应的 `base.server.json`。
- model+JSON 合同的 horizon=1/2 单元测试通过。
- 六个 V5.1 W&B run 均已完成或按用户要求停止。
- 已用完成的 hanging-mug raw V5.1 模型做真实 WebSocket 端到端验证：server handshake 正确声明 `observation_horizon=1`、两相机与 14D state；仅发送当前 global/hand 两张图即可返回有限的 `[1,30,7]` action，本机往返约 58 ms。
- 已处理 msgpack 只读 NumPy 图像，部署预处理会建立可写请求 buffer，不再触发 `torch.from_numpy` warning。

## 清理结果

- 仅保留六个模型 checkpoint，且文件名统一为 `base.pt`。
- 已删除所有 `.resume`、`base.best_epoch_*.pt` 和 `base.pt.best.weights.pt`。
- classify-blocks B-spline 的 `base.pt` 已验证为 `best_validation_model`，配置为 h1、14D state、B-spline；其相邻 `base.server.json` 可正常解析。
- 小型 `base.pt.wandb.json` 与 `base.server.json` 不是模型副本，分别用于 run 审计和部署合同，因此予以保留。
