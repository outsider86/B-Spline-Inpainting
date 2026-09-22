# V4 Flow-Matching 多数据集训练状态

日期：2026-09-22 UTC

六条 batch-64、h2 Flow-Matching 训练正在运行，每个 loader epoch 都会完整验证。
每个数据集分别训练 raw 与 B-spline；每个样本使用两个时刻、两个相机，以及从头联合训练
的 ResNet-18 编码器。

| 数据集 | Raw GPU / W&B | B-spline GPU / W&B | 每 epoch updates | 全量 validation samples |
|---|---|---|---:|---:|
| stacking_cup_30hz | 2 / `zjesf7u7` | 3 / `9zt4k9ha` | 636 | 4,537 |
| classify_blocks_30hz | 0 / `twqqp4w4` | 1 / `rniugssl` | 1,379 | 9,673 |
| hanging_mug_30hz | 4 / `erubna5k` | 5 / `aqxr7k6b` | 417 | 2,979 |

旧的 stacking 10-epoch validation run 已停止并保留。Fresh run 每个 epoch 验证，
每 100 epochs 保留精确快照。GPU 6 保持空闲。

hanging-mug 两条 run 均已在 update 41,700（epoch 100）完成。Raw 选择 update 33,360，
validation action MSE 为 0.0150083；B-spline 选择 update 41,283，指标为 0.0152258。
两份 validation-best `base.pt` 与 epoch-100 snapshot 均已通过 clean-load；GPU 4/5 已释放。

模型 checkpoint 仅保存在 scratch 本地。W&B 只记录指标与 summary，不再 staging 或上传
任何模型 artifact。

classify B-spline 的低吞吐来自 CPU 输入流水线，不是 GPU 降频：其压缩 episode target
cache 平均为 3.32 MiB，而 raw 仅 0.256 MiB。随机 window shuffle 配合每个 worker 只缓存
三个 episode，导致这些文件被反复解压；八个 DataLoader worker 忙于 CPU 解压时，GPU 1
长时间空闲。

下一步：核验 hanging 自动截停与最终 checkpoint，然后继续观察密集 validation 曲线，
根据用户判断决定其他数据集是否也在 epoch 100 提前停止。
