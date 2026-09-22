# FM V5：state + last command 训练状态

更新时间：2026-09-22 UTC

## 目标与输入契约

V5 在 V4 BSP Flow-Matching policy 的基础上，只改变低维输入：

- `observation.state[0:7]`：当前测量 state；
- `observation.state[7:14]`：上一条 command；
- h2 仍使用前后两个时刻，每个时刻两张图，因此每个 sample 共四张图；
- 图像编码器仍是每个 camera 独立、从零训练的 ResNet-18；
- policy 仍只训练 raw FM 与 B-spline FM 两种。

每个时刻的 observation feature 从 `2*64+7=135` 维变为
`2*64+14=142` 维，h2 global condition 从 270 维变为 284 维。

## 数据与 split

| 任务 | 数据集 | train/val episode | train/val window | updates/epoch |
|---|---|---:|---:|---:|
| classify blocks | `Data/Processed/classify_blocks_30hz_cleanup` | 45 / 5 | 86,561 / 9,493 | 1,352 |
| hanging mug | `Data/Processed/hanging_mug_30hz_cleanup` | 55 / 6 | 24,150 / 2,738 | 377 |
| stacking cup | `Data/Processed/stacking_cup_30hz_cleanup` | 55 / 6 | 37,780 / 4,245 | 590 |

三套数据都不使用 test split。所有 action sidecar、14D state normalization 和
RGB84 cache 均已完整生成并核验。

## 训练与 checkpoint 规则

- 100 loader epochs；batch size = effective batch size = 64；
- 每个 epoch 在完整 validation split 上做 generation-from-scratch
  `action_mse`；
- 内存中持续保留逐 epoch 的真实 best EMA；
- 每 10 epoch 将“截至该时刻的真实 best”保存为可独立部署的
  `base.best_epoch_010.pt`、...、`base.best_epoch_100.pt`；
- 最终 `base.pt` 也使用全程最佳 validation 权重；
- W&B 只记录 metrics/summary，不上传 model artifacts。

## 当前六个权威 run

共同 W&B project：`robot-policy-bsp-unet-v5-state-command-100e`

| GPU | 任务 | 表示 | W&B run ID |
|---:|---|---|---|
| 0 | classify blocks | raw | `to20cz26` |
| 1 | classify blocks | B-spline | `8348qdwj` |
| 2 | hanging mug | raw | `rwnzg4hv` |
| 3 | hanging mug | B-spline | `gb0et8i7` |
| 4 | stacking cup | raw | `v3s5a0an` |
| 5 | stacking cup | B-spline | `6fjdzy3p` |

运行状态文件：
`robot_policy/outputs/BSP_UNET_V5_STATUS.json`。

此前被中断的本地训练 checkpoints/logs/W&B runtime 已按要求删除，verified
prepared/RGB cache 被保留；表中六个 run 均从 update 0 统一重启。

## 已完成验证

- 六份 config 的 dataset audit 与 14D full-size forward/loss 均通过；
- 51 项相关 regression tests 通过；
- 完成一次 full-size GPU 端到端 smoke，验证训练、epoch-10 best 保存、独立
  checkpoint load、14D deployment input 和 `[B,30,7]` action output；
- smoke 临时文件已删除。

## 下一步

1. 完成六个 100-epoch run；
2. 对每个 run 核验 10 个 epoch-numbered best 和最终 `base.pt`；
3. 汇总六个模型的最终/最佳 validation generation `action_mse` 与 selected epoch；
4. 删除完成后的 optimizer recovery 文件，只保留模型与审计元数据。

## Hanging mug 完成与清理

- Raw：100/100 epochs，最终选择 epoch 100，validation action MSE
  `0.0135608550`。
- B-spline：100/100 epochs，最终选择 epoch 68，validation action MSE
  `0.0130568118`。
- 两个模型的 epoch-10...100 与 rolling-best checkpoint 在核验后已按要求
  删除；各目录只保留最佳 `base.pt`，以及很小的 W&B/manifest 审计元数据。

## Stacking cup 完成与清理

- Raw：100/100 epochs，最终选择 epoch 98，validation action MSE
  `0.0185145810`。
- B-spline：100/100 epochs，最终选择 epoch 49，validation action MSE
  `0.0188139177`。
- 两个模型的 epoch-10...100 与 rolling-best checkpoint 在核验后已按要求
  删除；各目录只保留最佳 `base.pt`，以及很小的 W&B/manifest 审计元数据。
