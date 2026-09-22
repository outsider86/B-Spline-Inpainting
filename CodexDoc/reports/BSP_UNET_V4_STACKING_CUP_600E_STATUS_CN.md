# BSP U-Net V4 stacking_cup_30hz 600 Epoch 状态

日期：2026-09-22 UTC

## 当前状态

Raw 与 B-spline 两条 h2 Flow-Matching 训练已在 GPU 2/3 启动。每个样本使用相邻两个
时刻、每个时刻 global/hand 两个相机，共四帧 RGB；视觉编码器仍为从随机初始化开始
联合训练的独立 ResNet-18。

W&B project：`robot-policy-bsp-unet-v4-stacking-cup-600e`

| 表示 | GPU | W&B run ID |
|---|---:|---|
| raw | 2 | `bsm6y5gh` |
| B-spline | 3 | `1272qqoh` |

## 数据与 epoch 协议

- 数据集：`Data/stacking_cup_30hz`。
- 审计结果：61 episodes、45,285 frames、30 Hz，无时间戳或 frame-index 异常。
- 固定 split seed `20260915`：55 train / 6 validation / 0 test。
- Validation episodes：8、18、19、22、34、54。
- Train windows：40,748；validation windows：4,537。
- Batch/effective batch：64/64。
- Training DataLoader 使用 `drop_last=True`，因此一个严格 loader epoch 为
  `floor(40,748/64)=636` updates；600 epochs 为 381,600 updates。
- 每 10 epochs（6,360 updates）执行一次完整 validation；validation batch 64，
  共 71 batches，覆盖全部 4,537 windows。
- 每 100 epochs（63,600 updates）保存一个精确 epoch snapshot。

## Checkpoint 计划

| Epoch | Update | 文件 |
|---:|---:|---|
| 100 | 63,600 | `base.epoch_100.pt` |
| 200 | 127,200 | `base.epoch_200.pt` |
| 300 | 190,800 | `base.epoch_300.pt` |
| 400 | 254,400 | `base.epoch_400.pt` |
| 500 | 318,000 | `base.epoch_500.pt` |
| 600 | 381,600 | `base.epoch_600.pt` |

每个 epoch snapshot 是该时刻精确的 EMA/online training payload。最终另存的
`base.pt` 使用全量 validation generation-from-scratch action MSE 最低点的推理权重。

## 独立 cache

新数据集没有复用旧 V4 的 split-dependent 内容。Raw/B-spline action targets、
normalization、B-spline calibration 和 RGB cache 全部重新生成。RGB cache 已逐 episode
校验：61 files、45,285 frames、双相机 `[2,3,84,84]`，与 action 长度完全一致。

## 下一步

监控首次 10-epoch 完整 validation、梯度与 optimizer skip；到 epoch 100 时核验第一个
不可变 checkpoint 的 update、hash、干净 reload 和 W&B 身份。
