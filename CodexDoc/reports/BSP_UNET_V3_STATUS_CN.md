# BSP 从头训练视觉 U-Net v3 状态

更新时间：2026-09-19 UTC

## 当前状态

新架构实现、数据缓存、部署接口接入和训练预检均已完成，现开始使用六张 GPU
训练 16 个 checkpoint。

## 已确认的参考架构

源码确认：参考 BSP policy 的图像编码器会与策略一起从头训练。每个相机使用一套
互不共享的 `pretrained=False` ResNet-18，并将 BatchNorm 替换为 GroupNorm；随后用
可学习的 1×1 映射和 32-keypoint SpatialSoftmax 得到每相机 64 个数，再经过
`Linear(64,64)+ReLU`。机器人状态不经过额外 MLP，直接拼接。连续观测时刻最终
flatten 成一个全局 FiLM condition，送入 `[256,512,1024]` temporal U-Net。

v3 沿用了上述图像编码器和 U-Net 规模。在当前双相机、7 维状态数据上，每个时刻
的 condition 是 `2×64+7=135` 维，因此单帧/双帧分别为 135/270 维。本实验完全
不使用 DINOv2 或 SigLIP。

## 实验矩阵

- 策略：连续 FM 或 256-bin 离散 diffusion；
- 动作：raw 30×7 或 cubic B-spline 18×7 控制点；
- 观测：当前单帧或连续“上一帧+当前帧”；
- 阶段：50k base 或 5k RTC 子模型；
- 总计：16 个 checkpoint。

离散 diffusion 使用 monotonic block corruption 学习离散动作 token。推理从所有
可变 token 全部为 MASK 开始，默认经过 8 轮逐步 unmask。连续与离散 RTC 使用同一
份 hard mask；B-spline RTC 只固定受影响 spans 所需的精确控制点支撑集。

## 已通过的预检

- 完整 FM：89,254,855 参数；
- 完整离散 DD：90,055,360 参数；
- BF16 全尺寸前向、反向和从零生成通过；
- 真实 batch 64 训练、EMA、验证、checkpoint 保存/重载和 manifest 均通过；
- 项目测试 62/62 通过；
- RGB 缓存：52 episodes / 31,706 frames / 双相机 / 84×84 uint8 CHW。

## 下一步

保持六条 GPU 队列连续运行，监控梯度和 validation/action-MSE 收敛情况；随后对全部
16 个 checkpoint 做从零生成、延迟、train/test RTC 轨迹和部署接口审计，生成中英
文总结，并发布到 `NewModel/v3`。
