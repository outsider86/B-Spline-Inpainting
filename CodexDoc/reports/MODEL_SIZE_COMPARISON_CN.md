# 当前策略与 StarVLA DiT 模型规模对比

**报告日期：** 2026-09-16

## 对比边界

当前策略参数包括完整的可训练策略：observation/state projector、动作 Transformer、embedding 和输出头；不包括使用缓存特征的冻结 DINOv2 与 SigLIP 视觉编码器。

下述 StarVLA 数值只统计动作 DiT，不包括 Qwen VLM，因此是最接近的架构对架构比较。本地 StarVLA 源码固定于 `2f17402a5ccaa09907516ae5e542b0fa6ee5d155`。

## 当前策略

| 动作表示 | 架构 | Observation/state | Action backbone | Embedding/output | 总可训练参数 |
|---|---|---:|---:|---:|---:|
| Raw | FM | 0.502M | 3.414M | 0.084M | **4.000M** |
| Raw | Layerwise DD | 0.502M | 3.414M | 0.139M | **4.055M** |
| Raw | Joint DD | 0.502M | 3.117M | 0.139M | **3.757M** |
| B-spline | FM | 0.502M | 3.414M | 0.082M | **3.997M** |
| B-spline | Layerwise DD | 0.502M | 3.414M | 0.123M | **4.039M** |
| B-spline | Joint DD | 0.502M | 3.117M | 0.123M | **3.741M** |

所有当前策略均使用 6 层 Transformer、hidden width 192 和 6 个 attention heads。

若进行严格的 action-head-only 对比，需要去掉 0.502M 的 observation/state projector。六个当前 action head 的参数量如下：

| 动作表示 | 架构 | 当前 action head |
|---|---|---:|
| Raw | FM | **3.498M** |
| Raw | Layerwise DD | **3.553M** |
| Raw | Joint DD | **3.256M** |
| B-spline | FM | **3.496M** |
| B-spline | Layerwise DD | **3.537M** |
| B-spline | Joint DD | **3.239M** |

## StarVLA DiT-S/B/L 预设

StarVLA 旧版 `DiTActionHeader` 明确定义了三个预设。下表精确参数量来自固定版本的构造器，并使用与当前实验相同的 7 维动作和 30 步 action horizon。diffusion scheduler 没有可学习参数，因此这里的数字也是该实现完整可训练 action head 的参数量。

| StarVLA 预设 | 结构 | Action head 参数 | 相对当前 3.239–3.553M heads | BF16 权重 |
|---|---|---:|---:|---:|
| DiT-S | 6 层、width 384、4 heads | **11.101M** | **大 3.12–3.43 倍** | 21.2 MiB |
| DiT-B | 12 层、width 768、12 heads | **86.535M** | **大 24.36–26.71 倍** | 165.1 MiB |
| DiT-L | 24 层、width 1024、16 heads | **304.759M** | **大 85.77–94.08 倍** | 581.3 MiB |

因此，当前策略甚至比 DiT-S 更小。从结构上看，它更接近一个自定义的 extra-small 预设：层数与 DiT-S 同为 6 层，但宽度只有一半（192 对 384）。

## StarVLA 参考规模

下面较新的 QwenPI 系列并不直接使用旧版 S/B/L 结构，应视为 StarVLA 的另一套 action-head 家族。

| StarVLA 变体 | 结构 | 仅 DiT 参数 | 相对当前 3.74–4.05M 策略 |
|---|---|---:|---:|
| QwenPI_v3 压缩动作 DiT | 36 层、width 1024、16 heads | **532.326M** | **大 131–142 倍** |
| QwenPI_v3 完整 action model | DiT 加动作编码器/解码器 | **538.678M** | **大 133–144 倍** |
| QwenPI / QwenDiscrete 动作 DiT | 36 层、width 2048、32 heads | **2.128B** | **大 525–569 倍** |

QwenPI_v3 源码还报告了 94.593M 的逐层 VLM-to-DiT projector，以及 5.071B 的完整 Qwen3-VL-4B 系统参数；这些不属于 DiT-only 对比。

## 权重内存

- 当前策略权重：FP32 约 **14.3–15.5 MiB**，BF16 约 **7.1–7.7 MiB**。
- StarVLA 压缩 DiT：FP32 约 **1.98 GiB**，BF16 约 **1.00 GiB**。
- StarVLA 2048-wide DiT：FP32 约 **7.93 GiB**，BF16 约 **3.96 GiB**。

以上均为纯权重估计，不包括 optimizer state、gradient、activation、VLM/视觉编码器和训练 trace。当前 45–55 MB checkpoint 文件还包含 AdamW 状态与元数据，因此大于推理权重本身。

## 结论

当前代码库实现的是 compact StarVLA-style policy，而不是与 StarVLA DiT 参数规模匹配的模型。当前完整可训练策略不到压缩 QwenPI_v3 action model 的 1%。因此现有实验适合比较架构和动作表示，但不能视为与完整 StarVLA 的等容量对比。
