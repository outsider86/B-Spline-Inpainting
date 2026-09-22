# V4 Flow-Matching 最终本地模型

日期：2026-09-22 UTC

所有训练进程均已停止，全部 GPU 空闲。W&B 只记录指标，不上传模型 artifact。每个变体的
权威模型都是本地 `base.pt`，其用于推理的 `model` 字段为观察到的 validation-best EMA
权重。

| 数据集 | 变体 | 选择点 | Validation action MSE | Checkpoint 类型 |
|---|---|---:|---:|---|
| stacking_cup_30hz | raw | epoch 41 | 0.0229889 | best 模型 + epoch-100 resume 状态 |
| stacking_cup_30hz | B-spline | epoch 50 | 0.0214426 | best 模型 + epoch-100 resume 状态 |
| classify_blocks_30hz | raw | epoch 64 | 0.0311488 | inference-only best 模型 |
| classify_blocks_30hz | B-spline | epoch 16 | 0.0304322 | inference-only best 模型 |
| hanging_mug_30hz | raw | update 33,360 | 0.0150083 | best 模型 + epoch-100 resume 状态 |
| hanging_mug_30hz | B-spline | update 41,283 | 0.0152258 | best 模型 + epoch-100 resume 状态 |

六份 checkpoint 均通过 clean policy load。两份 classify checkpoint 也通过 deployment
inspect/load，并明确标记 `training_resume_available=false`；这不影响推理与评估。

每组输出目录中的 `EARLY_STOP_VALIDATION_BEST.json` 保存了详细、机器可读的停止记录。
