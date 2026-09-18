# 全视觉 Token FM 与 Joint-DD 结果

## 实验协议

- 共 16 个检查点：DiT-S/B × raw/B-spline × FM/joint DD × base/ttRTC。
- 每个观测保留两个完整的 16×16 patch 网格，即 **512 个视觉 token**，并追加 1 个状态 token。
- 基础训练 50,000 次更新，ttRTC 训练 5,000 次更新，有效 batch size 为 32。
- `validation/action_mse` 来自从零开始的完整生成。采样器只接收视觉与状态，不接收目标动作、被扰动的真值或教师辅助信息。
- 后续评测包含完整测试集开环生成、延迟 RTC、oracle-prefix inference RTC 和 batch-1 同步延迟。
- 在每个 DiT 尺寸内，raw 与 B-spline 的 RTCEVAL 指标图共享同一组对数物理 MSE 纵轴范围；全部轨迹图按动作维度共享完整数据集的物理最小值/最大值。

## 主要结果

- 最佳物理 MSE 为 **0.008187**（DiT-B / raw / fm / ttrtc）。
- 在严格配对中，B-spline 有 5/8 组更优；物理 MSE 降幅中位数为 13.33%。
- ttRTC 在零延迟开环对比中有 6/8 组更优；降幅中位数为 0.55%。
- Joint cache 融合解码在 8/8 个检查点上逐 token 一致。
- FM 在 1,000 次 warm-up 更新之后共拒绝了 5625 次异常优化器更新；每次被拒绝的更新及其裁剪前梯度范数都完整保留在记录中。
- DiT-S、DiT-B 的部署验证均通过全部 8 个运行时用例；每份审计都清点全部 16 个检查点，并从两路各 256-token 的相机输入生成有限的 30×7 动作。

## 证据

- `completion_audit.json`：检查点身份、精确更新数、哈希、父模型谱系、有限梯度、从零验证、四类评测以及本地/远端 W&B 状态。
- `summary/deployment_validation_dit_{s,b}.json`：对全部 raw/B-spline × FM/joint-DD × base/ttRTC 组合执行真实 GPU 加载与推理。
- `summary/checkpoint_metrics.csv`：完整 16 行指标；`ranking_by_physical_mse.csv` 与 `pairwise_effects.csv`：排名和严格配对效应。
- 这些结果属于开环动作生成诊断，不代表闭环任务成功率。

## 全部检查点

| size | repr. | policy | stage | val gen MSE | test physical MSE | p50 ms | W&B |
|---|---|---|---|---|---|---|---|
| DiT-S | raw | fm | base | 0.007611 | 0.010827 | 25.61 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/70newtq7) |
| DiT-S | raw | fm | ttrtc | 0.008254 | 0.008333 | 15.39 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zt8fra2l) |
| DiT-S | raw | discrete_joint | base | 0.026301 | 0.023261 | 150.83 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/x1vtxm13) |
| DiT-S | raw | discrete_joint | ttrtc | 0.025776 | 0.022973 | 156.25 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/6cyontjw) |
| DiT-S | bspline | fm | base | 0.008366 | 0.008506 | 24.01 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/k2plilzb) |
| DiT-S | bspline | fm | ttrtc | 0.008092 | 0.008409 | 24.31 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zxp7uffq) |
| DiT-S | bspline | discrete_joint | base | 0.024129 | 0.019029 | 80.88 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/ivclz1i6) |
| DiT-S | bspline | discrete_joint | ttrtc | 0.025541 | 0.019057 | 84.87 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/efg4vkqh) |
| DiT-B | raw | fm | base | 0.006808 | 0.008203 | 68.15 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/mz36ylrg) |
| DiT-B | raw | fm | ttrtc | 0.007001 | 0.008187 | 68.32 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zip5w0in) |
| DiT-B | raw | discrete_joint | base | 0.015395 | 0.019526 | 298.37 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/c7mgy7f0) |
| DiT-B | raw | discrete_joint | ttrtc | 0.017608 | 0.020072 | 295.84 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/26qllqsn) |
| DiT-B | bspline | fm | base | 0.007076 | 0.008330 | 37.31 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/f581nhyj) |
| DiT-B | bspline | fm | ttrtc | 0.006578 | 0.008320 | 37.38 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/wo8veppb) |
| DiT-B | bspline | discrete_joint | base | 0.025449 | 0.017233 | 167.97 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/lwoiqslq) |
| DiT-B | bspline | discrete_joint | ttrtc | 0.026583 | 0.017078 | 163.03 | [run](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/680zgije) |
