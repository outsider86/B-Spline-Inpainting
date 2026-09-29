# Absolute Action V8 状态

## 目标

本轮只包含 `classify_blocks`，训练以下 8 个最终 checkpoint：

`{StateAbsJoint, StateAbsEE} × {raw, B-spline} × {base, ttRTC}`。

所有模型都采用 BSP-UNet Flow Matching；视觉 encoder 是从零训练的双相机
ResNet-18 + SpatialSoftmax。每个样本只使用一个当前时刻（h1），该时刻包含
global 和 hand 两张图，不使用历史图像。

## 两种 action contract

| Variant | Observation state | Predicted action |
|---|---|---|
| StateAbsJoint | 7D measured joints(6) + gripper(1) | 7D 下一时刻 absolute measured joints(6) + gripper(1) |
| StateAbsEE | 同一 7D measured-joint state | 8D 下一时刻 absolute TCP xyz(3) + quaternion xyzw(4) + gripper(1) |

代码原先默认 action 固定为 7D。V8 已将 action width 改为 checkpoint/config
显式字段，并贯通数据审计、raw/B-spline 编码、policy head、action MSE、RTC
hard mask、sidecar 校验和部署 metadata。旧 checkpoint 缺少该字段时仍默认 7D，
因此保持向后兼容。新增的 8D B-spline codec/ttRTC hard-mask 回归测试已通过。

## 数据审计与拆分

两个数据集分别位于：

- `/scratch/wangpc/B-Spline-Inpainting/Data/StateAbsJoint/classify_blocks_30hz_cleanup`
- `/scratch/wangpc/B-Spline-Inpainting/Data/StateAbsEE/classify_blocks_30hz_cleanup`

每个数据集均有 85 episodes、185,913 frames、170 MP4（global/hand）。所有
state/action 均为 finite；frame index 连续；timestamp 严格递增。两个 source
tree 的视频相对路径、大小和逐文件 SHA-256 完全一致，timestamp/frame/episode
数组也完全一致，因此共用一个 RGB84 cache；两种 action 的 SHA-256 不同，
action cache 按 state/action variant 和 representation 隔离。

固定随机种子 `20260915` 将数据拆为：

- train：76 episodes，166,875 frames/windows；
- validation：9 episodes，19,038 frames/windows；
- test：0。

## 训练协议

- batch/effective batch：64/64；
- base：100 epochs，共 260,700 optimizer updates；每 10 epochs 遍历完整
  validation split；只发布 validation-best 权重；
- ttRTC：从相同 leaf 的 validation-best base 初始化，训练 5 epochs，共
  13,035 updates；每个 epoch 遍历完整 validation split；
- raw 和 B-spline 都使用 reference hard-mask ttRTC；B-spline 固定受 delay
  spans 支持的 `D+3` 个 cubic control rows；
- 四个 base 在 GPU 0--3 并行，全部完成后四个 ttRTC 同样并行；
- W&B project：`robot-policy-bsp-unet-v8-classify-blocks`；不上传 W&B model artifact。

本地输出根目录：
`/scratch/wangpc/B-Spline-Inpainting/robot_policy/output/NEW/V8Full`。

计划远端目录：`DiscreteRTC/dRTC/NewModel/V8Full`。

## 当前进度与下一步

数据 contract 审计和 8D action 支持测试已通过，driver 正在准备四套独立 action
cache 和一个经过等价性证明的共享 RGB cache。随后会自动完成 base、ttRTC、严格
load/parent hash 审计、checkpoint 清理及带 portable sidecar 的 Hugging Face 发布。
