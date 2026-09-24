# ActionEE V7 状态

## 目标矩阵

本轮只训练一个 observation-state 版本：`State_EE`。三个 task 分别训练 raw
与 uniform left-clamped B-spline，因此共有 6 个 base 和 6 个 ttRTC 最终模型。

- state：8D，`TCP position(3) + quaternion xyzw(4) + gripper(1)`；
- action：7D，local `delta translation(3) + delta rotation rotvec(3) + next gripper(1)`；
- observation：h1，同一当前时刻的 global/hand 两张图；
- architecture：BSP-UNet Flow Matching；
- base：100 epochs，每 10 epochs 对完整 validation split 评估一次；
- ttRTC：从 validation-best base 开始，以全新 optimizer 微调 5 epochs，每个
  epoch 做完整 validation；
- test split：不使用；
- W&B：全部 12 个 run 进入同一个新 project
  `robot-policy-bsp-unet-v7-ee`，不上传 W&B artifact。

## 数据审计

已逐 episode 验证 parquet schema、维度、finite 值、frame index、episode index
和 timestamp：

| Task | Episodes | Frames | Videos | State | Action |
|---|---:|---:|---:|---:|---:|
| hanging_mug | 61 | 26,698 | 122 | 8D | 7D |
| stacking_cup | 61 | 41,752 | 122 | 8D | 7D |
| classify_blocks | 50 | 95,900 | 100 | 8D | 7D |

三个 dataset 的 `embodiment.json` 和 `modality.json` 均明确匹配上述 EE/delta-EE
语义。没有非有限 state/action，frame index 连续，timestamp 严格递增。

## 并行和输出协议

本地根目录：

`/scratch/wangpc/B-Spline-Inpainting/robot_policy/output/NEW/V7Full`

六个 base 已在 GPU 0--5 同时训练；已完成父模型的 ttRTC 会提前使用空闲 GPU，
其余 ttRTC 由同一个 driver 幂等续接。各 task 的
raw/B-spline 共用该 task 自己的 RGB84 cache，action cache 按 representation
隔离。最终每个 leaf 只保留 `base.pt`、`ttrtc.pt` 和可复现加载所需的 log/JSON。

最终上传位置：

`DiscreteRTC/dRTC/NewModel/V7Full`

## 最终结果（2026-09-24）

| Task | 表示 | 阶段 | 选中 epoch | 完整 validation action MSE | W&B ID |
|---|---|---|---:|---:|---|
| hanging_mug | raw | base | 100 | 0.05006838 | `ey82cx7f` |
| hanging_mug | raw | ttRTC | 5 | 0.04991525 | `a9loddl2` |
| hanging_mug | B-spline | base | 100 | 0.04653973 | `6yn4b220` |
| hanging_mug | B-spline | ttRTC | 5 | 0.04652055 | `4768yan2` |
| stacking_cup | raw | base | 100 | 0.05624402 | `8gqr77ky` |
| stacking_cup | raw | ttRTC | 5 | 0.05647647 | `je27amjd` |
| stacking_cup | B-spline | base | 70 | 0.05345627 | `a6wbwnij` |
| stacking_cup | B-spline | ttRTC | 5 | 0.05361407 | `kx3qow0j` |
| classify_blocks | raw | base | 90 | 0.05737160 | `pkalnt69` |
| classify_blocks | raw | ttRTC | 5 | 0.05701298 | `bbp2e1c9` |
| classify_blocks | B-spline | base | 70 | 0.05542328 | `klf0hh7z` |
| classify_blocks | B-spline | ttRTC | 4 | 0.05543821 | `oheabq6u` |

每个 base 都完成了 100 epoch，只是最终发布的是 validation winner。base 每 10
epoch 对完整 validation split 评估；ttRTC 每个 epoch 完整评估。样本数分别为
hanging `2,723`、stacking `4,212`、classify `9,469`。12 个 run 都没有
optimizer skip 或非有限梯度。

## 最终审计与发布

- 本地恰好保留 6 个 `base.pt` 和 6 个 `ttrtc.pt`，没有 `.resume`、step、
  best-weight 或 best-epoch checkpoint 残留；
- 12/12 模型均通过独立 CPU strict load。参数量均为 `87,333,831`；raw head
  直接预测 `30x7` action，B-spline head 预测 `18x7` cubic uniform-left control
  points，再解码为 `30x7` action；
- 6/6 ttRTC 的 parent 路径和 base SHA-256 一致；
- 模拟下载后本机绝对 prepared path 不存在的环境，6/6 task/representation
  均能从 `State_EE/sidecars/{raw,bspline}` 自动解析 encoder/normalization；
- W&B project `robot-policy-bsp-unet-v7-ee` 恰好有 12 个本轮 run，全部为
  `finished`，用户/模型 artifact 为 0。W&B 后端自动为 `8gqr77ky` 建立了一个
  system-managed `wandb-history` parquet；API 明确禁止删除，它不是模型上传；
- Hugging Face 最终发布包含 12 models、24 server/training JSON 和 12 portable
  sidecars，共 48 个受审计文件。模型 LFS SHA-256、sidecar blob hash 和远端存在性
  全部一致，最终 revision：
  `9e334da2f2a8aaca002f410b18cc38ae9dc3cc18`；
- classify epoch-30 的两个协作者临时 snapshot 仍保留并明确标记为非最终模型；
- 完整 repository tests 全部通过，另有 1 个历史 optional test skip。

权威审计文件位于 V7 output 根目录：`V7_STATUS.json`、`V7_AUDIT.json`、
`V7_HF_UPLOAD.json`、`V7_INDEPENDENT_AUDIT.json`。本目标已完成；
LastCommand 和 State+LastCommand 版本尚未启动，等待单独决策。
