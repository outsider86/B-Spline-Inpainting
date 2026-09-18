# DiT-S 全视觉实验结果（8 个检查点）

## 状态

原始 DiT-S 重跑目标已经完成，并通过独立审计。全部 8 个检查点均使用
双相机完整 16×16 token 网格，即每个相机 256 个、总计 256×2 = 512 个
视觉 token。基础策略训练 50,000 次更新；随后先按基础检查点精确哈希
缓存父模型预测，再进行 5,000 次 ttRTC 更新。

`validation/action_mse` 是从零开始的生成性能：策略只接收视觉和机器人
状态，FM 从噪声开始生成，联合离散扩散从全 MASK token 开始生成。目标
动作仅在生成完成后用于评分，不参与去噪条件。

| 动作表示 | 策略 | 阶段 | 最终验证 MSE | 最佳验证 MSE | 测试集物理 MSE | 采样 p50（ms） | W&B |
|---|---|---:|---:|---:|---:|---:|---|
| raw | FM | base | 0.007611 | 0.005651 | 0.010827 | 25.61 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/70newtq7) |
| raw | FM | ttRTC | 0.008254 | 0.006683 | 0.008333 | 15.39 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zt8fra2l) |
| raw | 联合 DD | base | 0.026301 | 0.013556 | 0.023261 | 150.83 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/x1vtxm13) |
| raw | 联合 DD | ttRTC | 0.025776 | 0.020801 | 0.022973 | 156.25 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/6cyontjw) |
| B-spline | FM | base | 0.008366 | 0.005903 | 0.008506 | 24.01 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/k2plilzb) |
| B-spline | FM | ttRTC | 0.008092 | 0.007014 | 0.008409 | 24.31 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/zxp7uffq) |
| B-spline | 联合 DD | base | 0.024129 | 0.017170 | 0.019029 | 80.88 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-basic/runs/ivclz1i6) |
| B-spline | 联合 DD | ttRTC | 0.025541 | 0.020515 | 0.019057 | 84.87 | [运行](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-full-vision-512-ttRTC/runs/efg4vkqh) |

## 验证结果

- 检查点清单：8/8 个原子写入的最终检查点。
- 每个检查点均完成：测试集全部 3,149 个片段的开环生成、batch-1
  延迟、每种延迟 128 个样本的 RTC，以及 64 个样本、前缀
  `{2,4,6,8,10}` 的 oracle-prefix 推理 RTCEVAL。
- 真实部署验证：8/8 策略均可加载并生成有限的 `[1,30,7]` 动作；视觉
  编码输出有限且使用 512 个 token。
- 本地审计与远程 W&B 审计均无错误通过。
- 测试套件：44/44 通过。

DiT-B/DiT-L 的扩展复现实验仍在运行，完成后将生成独立的 24 检查点
综合对比报告。
