# Absolute Action V8 完成报告

## 结论

V8 已完成。范围仅为 `classify_blocks`，最终保留并发布了 8 个 checkpoint：

`{StateAbsJoint, StateAbsEE} × {raw, B-spline} × {base, ttRTC}`。

所有模型均为 BSP-UNet Flow Matching，视觉部分为从零训练的双相机
ResNet-18 + SpatialSoftmax。每个样本仅输入当前一个时刻（h1）的 global 与 hand
两帧图像，不使用图像历史。训练使用 GPU 0--3，于 2026-09-29 04:52--08:02 UTC
完成。

## Action contract 与数据

| Variant | Observation state | Predicted action |
|---|---|---|
| StateAbsJoint | 7D measured joints(6) + gripper(1) | 7D 下一时刻 absolute measured joints(6) + gripper(1) |
| StateAbsEE | 同一 7D measured-joint state | 8D 下一时刻 absolute TCP xyz(3) + quaternion xyzw(4) + gripper(1) |

V8 已将 action width 从历史固定 7D 改为 checkpoint/config 显式字段，并贯通
数据准备、raw/B-spline codec、policy head、action MSE、RTC hard mask、sidecar
检查和部署。旧 checkpoint 未声明该字段时仍默认 7D。

两个 source tree 各有 85 episodes、185,913 frames、170 个 MP4。逐文件视频
SHA-256 及 timestamp/frame/episode 序列完全相同，因此安全共用 RGB84 cache；
action cache 按 StateAbsJoint/StateAbsEE 与 raw/B-spline 隔离。固定 seed
`20260915` 的拆分为：

- train：76 episodes，166,875 windows；
- validation：9 episodes，19,038 windows；
- test：0。

## 训练协议与最终选择

- batch/effective batch：64/64；
- base：100 epochs、260,700 updates，每 10 epochs 对完整 validation split
  验证一次；
- ttRTC：从同一 leaf 的 validation-best base 初始化，5 epochs、13,035
  updates，每个 epoch 完整验证；
- raw 与 B-spline 都使用 reference hard-mask ttRTC；B-spline delay spans 保留
  `D+3` 个 cubic control rows；
- 所有 8 个 run 均为 0 skipped optimizer step，loss 与 gradient diagnostics
  均为 finite。

下表是实际发布 checkpoint 的 validation-best 选择，不是 W&B 页面上的最后一个
epoch 数值：

| State/action | Representation | Stage | Selected epoch | Selected update | Validation action MSE |
|---|---|---|---:|---:|---:|
| StateAbsJoint | raw | base | 50 | 130,350 | 0.02690121 |
| StateAbsJoint | raw | ttRTC | 5 | 13,035 | 0.02677587 |
| StateAbsJoint | B-spline | base | 50 | 130,350 | 0.02701399 |
| StateAbsJoint | B-spline | ttRTC | 5 | 13,035 | 0.02683749 |
| StateAbsEE | raw | base | 40 | 104,280 | 0.15562337 |
| StateAbsEE | raw | ttRTC | 3 | 7,821 | 0.15484847 |
| StateAbsEE | B-spline | base | 10 | 26,070 | 0.14707805 |
| StateAbsEE | B-spline | ttRTC | 5 | 13,035 | 0.14614573 |

这里的 validation action MSE 是 generation-from-scratch 的完整 validation
统计。W&B run summary 默认显示最后一次验证，因此当最佳点较早出现时会与上表不同；
发布模型以 `V8_AUDIT.json` 和各 `*.training.json` 中的 selected checkpoint 为准。

## 审计与发布

- 本地目录：
  `/scratch/wangpc/B-Spline-Inpainting/robot_policy/output/NEW/V8Full`；
- 本地最终树恰好包含 8 个 `.pt`，没有 `.resume`、periodic 或 best-epoch
  临时 checkpoint；
- 8/8 checkpoint 均在 CPU 上通过严格部署加载；Joint/EE 参数量分别为
  87,319,495 / 87,321,288；
- 4 个 ttRTC checkpoint 的 parent path 与 parent SHA-256 均与对应 base 完全匹配；
- 完整测试为 92 collected、91 passed、1 个历史 optional test skipped；
- W&B project：`robot-policy-bsp-unet-v8-classify-blocks`，恰好 8 个 V8 run，
  全部 `finished`，0 个 logged artifact；
- Hugging Face：`DiscreteRTC/dRTC/NewModel/V8Full`；共 32 个文件，包括
  8 models、16 server/training JSON 和 8 portable encoder/normalization sidecars；
- 远端 revision：`da92d9d3899320d132177d4911ed1efae5837bfc`。独立复查结果为
  32/32 文件存在，8/8 model LFS SHA-256 及所有 JSON blob hash 匹配。

权威运行证据位于输出根目录下的 `V8_STATUS.json`、`V8_DATASET_AUDIT.json`、
`V8_AUDIT.json` 和 `V8_HF_UPLOAD.json`。

## 下一步

V8 的训练、ttRTC finetune、清理、可移植部署打包和上传目标已经完成。后续如需
比较实际策略质量，应单独运行同一批样本上的 open-loop generation-from-scratch
与 GT-prefix RTC 评估，并分别报告 Joint 与 EE 的分维度误差；该评估不属于本次
训练目标。
