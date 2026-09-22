# BSP 从头训练视觉 U-Net v3 状态

更新时间：2026-09-19 UTC

## 当前状态

原始 16-checkpoint 矩阵已经全部完成。FM checkpoint 已定版；首轮 DD 暴露出训练分布
和 checkpoint 选择问题，现完整保存在
`outputs/BSP_UNET_V3/diagnostics/legacy_d2f_partialval_50k`。修订后的四组 DD
base+RTC 正在 GPU 0--3 并行重训；8 个已冻结 FM checkpoint 正在上传到
`DiscreteRTC/dRTC/NewModel/v3`。

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
- 阶段：FM 为 50k base；修订 DD 为 10k base，并用完整 validation 选择最佳权重；
  两者 RTC 均为 5k；
- 总计：16 个 checkpoint。

修订后的离散 diffusion 用 50% 全 MASK 样本和 50% 原始 D2F block corruption：
既保留部分 inpainting/RTC 能力，也直接覆盖从全 MASK 生成的初始状态。推理默认采用
确定性的 8 轮 MaskGIT unmask。连续与离散 RTC 使用同一份 hard mask；B-spline RTC
只固定受影响 spans 所需的精确控制点支撑集。

## DD 问题与验证证据

旧训练每次只用 validation 的 256/3077 个样本选择 checkpoint，早期的噪声最优点与
完整 test 不一致；同时 teacher-corruption loss 持续降低，但从零 rollout 已饱和甚至
退化。完整 validation 的对照实验选择了 50% full-MASK。10k 诊断模型的 held-out
test normalized action MSE 为：raw h1/h2 `0.04773/0.04977`，B-spline h1/h2
`0.06547/0.06460`；归档的 50k B-spline 旧基线为 `0.08192/0.07738`。

同时测试了 DD-OpenVLA 的 categorical sampling 和退火 Gumbel remask，但不同表示上
收益很小且不一致，因此保留由完整 validation 选出的确定性 argmax/confidence 解码。
BSP-UNet 是卷积 U-Net，不存在 transformer KV cache；评估元数据已正确标记
`joint_kv_cache=false`。

## 已通过的预检

- 完整 FM：89,254,855 参数；
- 完整离散 DD：90,055,360 参数；
- BF16 全尺寸前向、反向和从零生成通过；
- 真实 batch 64 训练、EMA、验证、checkpoint 保存/重载和 manifest 均通过；
- 项目测试 65/65 通过；
- RGB 缓存：52 episodes / 31,706 frames / 双相机 / 84×84 uint8 CHW。

## 下一步

完成四组修订 DD base/RTC；随后对最终 16 个 checkpoint 做从零生成、延迟、
train/test RTC 轨迹和真实部署接口审计，完成中英文总结与 `NewModel/v3` 上传。
