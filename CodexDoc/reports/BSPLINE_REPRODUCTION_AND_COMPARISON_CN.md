# B-spline 复现与 Raw-Action 对比报告

**报告日期：** 2026-09-16  
**状态：** B-spline 确定性训练、ttRTC 微调、评估、回放、延迟测试、可视化及 W&B 对比均已完成。

## 结果概览

B-spline 实验已按照 raw-action 实验的受控协议重新运行：使用相同数据集与 episode 划分、相同三类策略架构、优化器与更新步数、确定性随机种子 `7`、相同评估窗口及同等级 GPU。共生成 6 个生产 checkpoint，保存在 `robot_policy/outputs/bspline_reproduction/`，没有覆盖旧的历史结果。

W&B 对比入口：

- [Raw 与 B-spline 对比项目](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-action-representation-comparison)
- [权威对比运行 `eqwedr12`](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-action-representation-comparison/runs/eqwedr12)
- [B-spline Basic 项目](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic)
- [B-spline ttRTC 项目](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC)

专用对比运行记录了 12 行 W&B 表格、物理动作空间 action MSE 图、ttRTC 效果图，并上传了 JSON/CSV 机器可读 artifact。

## 可比的 `action_mse` 定义

目前 `action_mse` 能感知动作表示方式，同时保证 raw 与 B-spline 之间可比：

- 指标在解码后的归一化 `30 × 7` raw-action 空间计算。
- 只评估受 corruption-supervised 控制点影响的原始动作位置。
- 对 B-spline，解码前用真值填充未被监督的控制点，避免无关样条支撑区污染指标。
- 对 raw identity 表示，该定义严格退化为原始 raw-action 指标；自动化测试验证二者完全一致。
- W&B 键名为 `train/action_mse` 与 `validation/action_mse`；各架构自身优化目标单独记录。

评估阶段另有主要指标：在全部 3,149 个 held-out 窗口上计算解码后物理动作 MSE。它不与训练时的归一化、带 corruption 条件的诊断指标混用。

## B-spline 训练结果

Basic 模型均训练 2,000 步，ttRTC 模型均微调 800 步；有效 batch size 为 128，使用 AdamW、学习率 `3e-4`、BF16 和随机种子 `7`。

| 架构 | 阶段 | 最终验证目标 | 最终归一化验证 `action_mse` | W&B run |
|---|---:|---:|---:|---|
| Flow matching | Basic | 0.058429 | **0.018705** | [`17qdogt2`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/17qdogt2) |
| Flow matching | ttRTC | 0.081424 | **0.021648** | [`ke2jdwfe`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/ke2jdwfe) |
| Layerwise discrete | Basic | 13.625690 | **0.014264** | [`8go0inek`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/8go0inek) |
| Layerwise discrete | ttRTC | 13.590894 | **0.010950** | [`0d87eo7m`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/0d87eo7m) |
| Joint discrete | Basic | 12.705885 | **0.013798** | [`h774y2hw`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-basic/runs/h774y2hw) |
| Joint discrete | ttRTC | 14.613434 | **0.014376** | [`ng3rdmdp`](https://wandb.ai/401910710-university-of-california-berkeley/bspline-actions-ttRTC/runs/ng3rdmdp) |

三个 Basic 精确重跑的 run ID 为 `2wt5pssh`、`bbpykwqj` 与 `wcblowty`。每个重跑都在 2,000 个训练点、20 个验证点及所有最终参数张量上实现最大差值为零。

## B-spline Held-out 评估

| 架构 | 阶段 | 物理动作 MSE ↓ | MAE ↓ | RMSE ↓ | Token accuracy |
|---|---:|---:|---:|---:|---:|
| Flow matching | Basic | **0.013835** | 0.053852 | 0.117623 | N/A |
| Flow matching | ttRTC | **0.013294** | 0.055207 | 0.115299 | N/A |
| Layerwise discrete | Basic | **0.032380** | 0.101878 | 0.179944 | 0.1425 |
| Layerwise discrete | ttRTC | **0.045018** | 0.131219 | 0.212174 | 0.1164 |
| Joint discrete | Basic | **0.023766** | 0.075348 | 0.154162 | 0.1708 |
| Joint discrete | ttRTC | **0.020955** | 0.073556 | 0.144758 | 0.1634 |

在 B-spline 表示内部，ttRTC 使完整测试集 MSE 在 flow matching 上下降 **3.91%**，在 joint discrete 上下降 **11.83%**，但在 layerwise discrete 上上升 **39.03%**。因此 ttRTC 对前两者有效，但对本次复现的 layerwise 模型产生了明显退化。

## Raw 与 B-spline 对比

下表统一使用解码后的物理动作 MSE，因此两种表示处于相同输出坐标系。

| 架构 | 阶段 | Raw MSE ↓ | B-spline MSE ↓ | B-spline 相对 Raw |
|---|---:|---:|---:|---:|
| Flow matching | Basic | **0.013315** | 0.013835 | 差 3.91% |
| Flow matching | ttRTC | **0.012975** | 0.013294 | 差 2.45% |
| Layerwise discrete | Basic | 0.058202 | **0.032380** | 好 44.37% |
| Layerwise discrete | ttRTC | **0.044474** | 0.045018 | 差 1.22% |
| Joint discrete | Basic | 0.044148 | **0.023766** | 好 46.17% |
| Joint discrete | ttRTC | 0.033632 | **0.020955** | 好 37.69% |

核心结论：B-spline 对 joint-discrete 以及 Basic layerwise 模型提升显著，而 raw action 在 flow matching 上略优。Layerwise B-spline ttRTC 是最清楚的退化项，在进一步分析前不能把 ttRTC 视为普遍有效。

## Delay、回放与延迟

B-spline delay sweep 对 `d=0…10` 的每个延迟使用相同的 128 个 held-out 样本。偶数延迟与训练过的 spline span 对齐，奇数延迟是刻意保留的未见 within-span 情况。所有设置下，已提交前缀的保持误差均为零。

| 架构 | Basic 平均 delay MSE | ttRTC 平均 delay MSE | ttRTC 效果 |
|---|---:|---:|---:|
| Flow matching | 0.008309 | 0.008310 | 差 0.01% |
| Layerwise discrete | 0.026190 | 0.038691 | 差 47.74% |
| Joint discrete | 0.031480 | 0.019862 | 好 36.91% |

三个 B-spline ttRTC 回放均执行 120/120 条命令，并在 `D=3` spans 设置下发生 20 次 plan switch。它们属于数据集回放，不代表闭环机器人任务成功率。

默认 batch-one 采样 p50：Basic/ttRTC flow matching 为 15.33/15.28 ms，layerwise discrete 为 12.58/11.87 ms，带缓存的 joint discrete 为 84.90/85.43 ms。B-spline joint 只有 126 个标量 token，相比 210-token raw joint，在默认设置下约快一倍。

## 验证与产物

- 项目测试：**14 passed**。
- B-spline checkpoint 独立重载：**6/6 passed**。
- 同随机种子 Basic 精确复现：**3/3 passed**。
- W&B 状态：6 个生产 run、3 个重跑 run 及 1 个对比 run 均已核验为 `finished`。
- B-spline 拟合误差：归一化 MAE `0.001384`，RMSE `0.008136`。
- 额外 256-bin 量化误差：归一化 MAE `0.001716`，RMSE `0.002247`。

主要本地产物：

- 配置：`robot_policy/configs/bspline_reproduction.yaml`
- Checkpoint：`robot_policy/outputs/bspline_reproduction/checkpoints/`
- 评估：`robot_policy/outputs/bspline_reproduction/evaluation/`
- 精确复现：`robot_policy/outputs/bspline_reproduction/reproducibility/`
- 可视化：`robot_policy/outputs/bspline_reproduction/visualizations/`
- 对比 JSON/CSV/图像：`robot_policy/outputs/action_representation_comparison/`

W&B 项目中仍保留采用旧指标语义的早期 exploratory B-spline runs。只有上表列出的生产 run ID 与 checkpoint manifest 是权威结果。

## 下一步

1. 每种表示至少运行三个不同的确定性随机种子，从而测量统计方差，而不只是验证相同种子的精确可重复性。
2. 针对 layerwise B-spline ttRTC 退化，增加按 delay、预测时间段、joint-only 与 gripper-only 划分的 MSE 分析。
3. 为 raw 与 B-spline 的比较增加逐窗口配对置信区间和显著性检验。
4. 只有在获得运行时集成、标定与安全授权后，才增加闭环仿真或硬件评估。
