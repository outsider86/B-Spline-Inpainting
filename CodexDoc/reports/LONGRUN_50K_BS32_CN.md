# 50k Basic / 5k ttRTC、Batch-32 训练报告

**报告日期：** 2026-09-16  
**状态：** 已完成。

## 已交付范围

- 6 个 Basic checkpoint：三种架构 × Raw/B-spline，每个严格训练 **50,000 个 optimizer updates**。
- 6 个 ttRTC checkpoint：分别从对应的 50k parent 微调，严格训练 **5,000 个 optimizer updates**。
- Micro-batch 与 effective batch size 均为 **32**，没有隐藏的梯度累积。
- 仅使用两个新 W&B 项目，每个项目严格包含 6 个生产 run。
- 每个 run 都记录 `train/action_mse`、`validation/action_mse`，并包含模型 artifact。

## W&B 项目

- Basic，6/6 finished：[robot-policy-50k-bs32-basic](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic)
- ttRTC，6/6 finished：[robot-policy-5k-bs32-ttRTC](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC)

### Basic runs

| 动作表示 | 架构 | Run ID | Updates | 最终验证 `action_mse` |
|---|---|---|---:|---:|
| Raw | FM | [`y7cijqp5`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/y7cijqp5) | 50,000 | 0.029513 |
| Raw | Layerwise DD | [`jnwtiwdf`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/jnwtiwdf) | 50,000 | 0.003621 |
| Raw | Joint DD | [`p3i9bwzy`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/p3i9bwzy) | 50,000 | 0.011324 |
| B-spline | FM | [`dtwgmcv5`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/dtwgmcv5) | 50,000 | 0.026211 |
| B-spline | Layerwise DD | [`jwco3ma9`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/jwco3ma9) | 50,000 | 0.002321 |
| B-spline | Joint DD | [`m6nqhqyl`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-50k-bs32-basic/runs/m6nqhqyl) | 50,000 | 0.009827 |

### ttRTC runs

| 动作表示 | 架构 | Run ID | Updates | 最终验证 `action_mse` |
|---|---|---|---:|---:|
| Raw | FM | [`mlaw9wev`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/mlaw9wev) | 5,000 | 0.030458 |
| Raw | Layerwise DD | [`orcbvl0u`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/orcbvl0u) | 5,000 | 0.004734 |
| Raw | Joint DD | [`zrscqaje`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/zrscqaje) | 5,000 | 0.007063 |
| B-spline | FM | [`5qxw6xgv`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/5qxw6xgv) | 5,000 | 0.025992 |
| B-spline | Layerwise DD | [`kjfy1n9h`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/kjfy1n9h) | 5,000 | 0.001907 |
| B-spline | Joint DD | [`oz24h02u`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-5k-bs32-ttRTC/runs/oz24h02u) | 5,000 | 0.013061 |

以上数值是归一化、带 corruption 条件的验证诊断指标，并非 held-out 物理空间 rollout 结果。本轮训练交付没有要求新增 open-loop 评估。

## Checkpoint 产物

- Raw checkpoints 与 manifest：`robot_policy/outputs/longrun_50k_bs32/raw/checkpoints/`
- B-spline checkpoints 与 manifest：`robot_policy/outputs/longrun_50k_bs32/bspline/checkpoints/`
- Raw 配置：`robot_policy/configs/longrun_50k_bs32_raw.yaml`
- B-spline 配置：`robot_policy/configs/longrun_50k_bs32_bspline.yaml`

Raw ttRTC 文件使用明确的 `_ttrtc.pt` 后缀。B-spline 文件保留仓库已有的 `_rtc.pt` 后缀；它们就是本轮要求的 ttRTC checkpoint，W&B 名称统一使用 `ttrtc`。

## 完整性验证

- 本地 checkpoint 记录：**12/12 passed**。
- 通过策略加载器独立 CPU 重载：**12/12 passed**。
- 更新步数：所有 Basic 为 `50,000`，所有 ttRTC 为 `5,000`。
- 每个 checkpoint 的样本数：Basic `1,600,000`，ttRTC `160,000`。
- 每个 payload 与 W&B 配置中的 batch size 和 effective batch size 均为 `32`。
- Parent lineage：每个 ttRTC checkpoint 均指向同一动作表示和架构的 50k parent，manifest SHA-256 完全匹配。
- Joint parent cache：Raw 与 B-spline 均为 52 episodes / 31,706 frames，parent hash 匹配，且为精确整数 token。
- W&B：严格为 6 + 6 个 run，全部 `finished`，全部包含 action-MSE 指标键和模型 artifact。
- 项目测试：**14 passed**。

## 下一步

1. 对这 12 个长训练 checkpoint 运行同一套 held-out open-loop、delay、replay 与 latency 评估。
2. 使用解码后的物理 action MSE，将 50k/5k 结果与此前 2k/800 实验进行对比。
3. 如果需要统计方差而不仅是确定性单种子训练，则增加多个随机种子。
