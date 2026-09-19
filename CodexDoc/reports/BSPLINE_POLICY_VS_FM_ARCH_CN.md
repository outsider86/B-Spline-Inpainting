# `RefCode/bspline-policy` 与当前 FM Policy 的架构对比

更新时间：2026-09-19 UTC

## 范围与结论

本文只比较**可学习的策略架构**：观测如何进入策略、动作序列主干、时间/噪声条件、训练目标与生成过程。本文有意不比较数据预处理、归一化细节、B-spline 拟合算法或动作表示优劣。

核心结论：`RefCode/bspline-policy` 不是一个全新的策略主干；它用很薄的适配器继承了 `diffusion_policy` 的两种已有策略。仓库文档和默认训练配置使用 **Conditional 1D U-Net + DDIM epsilon diffusion**；另有一个可选的 **encoder-decoder Transformer + DDIM**。当前项目的 FM 则使用 **StarVLA 风格的交替 cross/self-attention DiT + flow matching**。因此二者最重要的差别是生成主干和生成目标，而不是 B-spline 表示本身。

## 一页对比

| 维度 | 参考默认 B-spline Policy | 参考可选 Transformer | 当前 FM Policy |
|---|---|---|---|
| 生成主干 | 1D temporal U-Net | Transformer decoder | 交替 cross/self-attention DiT |
| 动作序列交互 | 局部卷积、多尺度下采样/上采样、skip connection | causal self-attention | 全序列 self-attention；每隔一层对观测做 cross-attention |
| 观测注入 | 两个观测时刻压成一个全局向量，在所有残差块内用 FiLM 注入 | 时间 token 与观测 token 构成 decoder memory | 保留空间视觉 token；动作/latent token 直接 cross-attend 观测 token |
| 时间条件 | diffusion timestep sinusoidal embedding + MLP | 独立 timestep condition token | continuous time sinusoidal embedding + MLP，经 AdaNorm 调制 attention 分支 |
| 学习目标 | 预测高斯噪声 `epsilon` 的 MSE | 预测高斯噪声 `epsilon` 的 MSE | 预测 rectified-flow velocity `target - noise` 的 MSE |
| 训练噪声路径 | 100 个离散 diffusion timesteps | 100 个离散 diffusion timesteps | 连续线性路径，时间按 Beta 分布采样 |
| 默认生成 | 16 步 DDIM | 10 步 DDIM | 12 步显式 Euler ODE 积分 |
| 因果性 | 卷积本身不是 causal mask | 配置使用 causal attention | self-attention 不使用 causal mask，一次联合生成完整 chunk |
| 当前容量 | 固定 `[256,512,1024]` U-Net | 256 宽、8 层、4 heads | DiT-S：384/6/4；DiT-B：768/12/12 |
| RTC 硬条件 | 参考代码没有当前项目的 span-support RTC 契约 | 同左 | FM 在每个 Euler step 前后重新覆盖硬条件 |

## 参考仓库：默认策略实际上是什么

### 1. B-spline wrapper 很薄

默认类 `DiffusionUnetBSplineImagePolicy` 直接继承 `DiffusionUnetHybridImagePolicy`。它没有引入新的 U-Net block、attention block 或新的 diffusion objective；策略结构仍由上游 `diffusion_policy` 实现。B-spline wrapper 对模型输入/输出做少量适配，并让 rollout 返回完整预测 horizon。

因此，讨论参考策略架构时，应审计父类及其 `ConditionalUnet1D`，而不能只看 wrapper 文件。

关键源文件：

- `RefCode/bspline-policy/bspline_policy/bspline_policy/policy/diffusion_unet_bspline_image_policy.py`
- `RefCode/bspline-policy/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`
- `RefCode/bspline-policy/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py`
- `RefCode/bspline-policy/bspline_policy/bspline_policy/config/train_diffusion_unet_real_hybrid_bspline_workspace.yaml`

### 2. Conditional 1D U-Net 主干

默认 action generator 接收形状为 `[B, horizon, action_dim]` 的 noisy action sequence，在进入网络后转成 channel-first 的 `[B, action_dim, horizon]`。配置中的主干宽度是 `256 -> 512 -> 1024`，kernel size 为 5。

每个分辨率层包含两个 conditional residual blocks：

1. `Conv1d -> GroupNorm -> Mish`；
2. 从 diffusion-time embedding 与全局 observation condition 生成逐通道调制参数；
3. 配置 `cond_predict_scale=True`，因此使用 scale + bias FiLM；
4. 再经过一个 `Conv1d -> GroupNorm -> Mish`；
5. 与 identity/1x1 residual branch 相加。

编码侧逐层下采样，最深处有两个 1024-channel residual blocks；解码侧拼接 skip feature 后逐层上采样，最后用卷积投影回 action dimension。这种设计的归纳偏置是**局部时间卷积 + 多尺度聚合**，不是 token-to-token attention。

### 3. 观测条件

默认配置 `n_obs_steps=2` 且 `obs_as_global_cond=True`。它的输入不是 patch-token sequence，而是一个包含多路 RGB 与低维机器人状态的字典。以 batch size `B` 为例，每个 key 的原始策略输入是：

- RGB：`[B, 2, 3, H, W]`；
- position/quaternion/gripper/joint state：`[B, 2, d_key]`；
- action target 或推理噪声：`[B, 16, action_dim]`。

这里的 `2` 是连续两个 observation steps，`16` 是生成 horizon。参考实现没有 language input，也没有把 previous action history 作为 policy input；源码还显式拒绝 `past_action`。

#### 3.1 每路图像如何编码

dataset 先把 HWC `uint8` 图像变为 CHW `float32`，除以 255 得到 `[0,1]`；policy normalizer 再线性映射到 `[-1,1]`。随后每张图像按以下路径处理：

```text
单路 RGB [3,H,W]
  -> 训练时随机 / 推理时中心 crop 到 76x76
  -> ResNet18Conv（去掉 avgpool 与 FC，pretrained=False）
  -> GroupNorm 版本的 ResNet feature map
  -> 1x1 conv 产生 32 张 spatial-attention maps
  -> SpatialSoftmax：每张 map 输出期望坐标 (x,y)
  -> 32 x 2 = 64
  -> Linear(64,64) + ReLU
  -> 本相机的 64-D feature
```

重要细节：

- 每个 camera key 都会创建一套独立的 `VisualCore/ResNet18Conv`；代码没有设置 `share_net_from`，所以多相机之间默认不共享 CNN 参数。
- ResNet-18 不是 ImageNet pretrained，而是随 policy 从头训练。
- 配置把 ResNet 中的 `BatchNorm2d` 换为 `GroupNorm(num_channels/16 groups)`。
- `SpatialSoftmax` 将整张 feature map 压成 32 个二维关键点，因此每路相机最终只有 64 个标量，而不是 64 个 spatial tokens。
- crop 的 `num_crops=1`：训练随机取一块，eval 取中心块，不会在推理时做多 crop ensemble。

#### 3.2 低维状态如何加入

low-dimensional keys 没有 MLP encoder；归一化后直接 flatten。所有相机的 64-D features 与所有 low-dimensional values 按 `shape_meta` 中的 key 顺序直接 concatenation：

```text
obs_feature(t) = concat(camera_1_64D, ..., camera_N_64D,
                        all_low_dim_state)
```

三份任务配置对应的单时刻和两时刻维度如下：

| 任务配置 | 图像输入 | 低维状态 | 单时刻 `Do` | U-Net global condition `2*Do` |
|---|---:|---:|---:|---:|
| `square_image_abs_bspline` / YAM | 1 camera × 64 | 3+4+1 = 8 | 72 | 144 |
| `clean_bspline_policy_haoyu_left` | 2 cameras × 64 | 3+4+1 = 8 | 136 | 272 |
| `stack_cube_teleop` | 3 cameras × 64 | 双臂各 6+3+4+1，共 28 | 220 | 440 |

#### 3.3 最终送给默认 U-Net 的整体输入

实现先把所有 observation keys 的前两个时刻从 `[B,2,...]` reshape 为 `[B*2,...]`，让 observation encoder 逐时刻处理；编码结果再恢复为 `[B,2,Do]`，最后 flatten 成：

`global_cond: [B, 2*Do]`

默认 U-Net 的一次 forward 实际接收三个部分：

1. `noisy_action`: `[B,16,Da]`，进入 U-Net 前转成 `[B,Da,16]`；
2. `diffusion_timestep`: `[B]`，经 128-D sinusoidal embedding 和 MLP；
3. `global_cond`: `[B,2*Do]`。

time embedding 与 `global_cond` 拼接后，在每个 U-Net residual block 中通过 FiLM 产生 channel-wise scale 与 bias。也就是说，**图像 feature 不与 noisy action 在输入维直接拼接，也不成为 attention tokens**；它是广播到整个 action horizon 的全局条件。

按现有 task/config，完整生成器输入形态是：

| 任务 | `noisy_action` | `global_cond` | timestep embedding |
|---|---:|---:|---:|
| YAM / Haoyu-left | `[B,16,11]` | `[B,144]` / `[B,272]` | `[B,128]` 后再经 MLP |
| Stack-cube | `[B,16,21]` | `[B,440]` | `[B,128]` 后再经 MLP |

上表的 action channel 包含 reference wrapper 使用的完整模型通道；这里只用于说明网络张量，不评价动作表示。

从策略结构角度看，这意味着参考默认模型在进入生成主干前已将观测压成全局向量；生成主干不能直接 cross-attend 某个空间 patch/token。

#### 3.4 可选 Transformer 的不同点

可选 Transformer 复用完全相同的图像/状态 encoder，但不把两个时刻 flatten 成一个向量。它保留 `[B,2,Do]`，分别线性投影成两个 condition tokens；再在前面加入一个 diffusion-time token，形成 3 个 decoder-memory tokens。即使在这条路径里，也不存在 image patch tokens：每个时刻的全部相机和机器人状态已经被压成一个 `Do`-dim vector。

### 4. 目标和采样

默认配置使用：

- 100 个训练 diffusion timesteps；
- squared-cosine beta schedule；
- `prediction_type=epsilon`；
- noisy sample 上的 epsilon MSE；
- 16 个 DDIM inference steps；
- EMA 权重用于评估/rollout。

它学习的是离散 diffusion scheduler 定义的去噪场，不是连续 flow velocity。

### 5. 默认 U-Net 的参数规模应如何理解

不含 observation encoder 时，配置 `[256,512,1024]` 的 generator 已约为 **63M 参数**；全局 observation condition 还会增加各 FiLM projection 的参数，因此准确总数依赖任务的 observation feature dimension。参考仓库没有提供一个跨任务恒定的完整 policy 参数数，不能把某一个任务的视觉 encoder 和当前项目离线视觉特征方案直接混为同一口径。

## 参考仓库：可选 Transformer

`train_dp_transformer.yaml` 提供另一个非默认方案：

- embedding width 256；
- 8 个 Transformer decoder layers；
- 4 attention heads；
- FFN expansion 4x；
- attention dropout 0.3；
- action token 使用 learned positional embedding；
- diffusion time 与 observation features 作为 decoder memory；
- action self-attention 和 action-to-condition cross-attention 都使用 causal mask；
- 目标仍是 epsilon prediction；
- 配置为 10 步 inference。

`n_cond_layers=0` 不等于完全没有 condition encoder：代码会用一个 `Linear -> Mish -> Linear` MLP 处理 time/observation condition。该 generator 在代表性维度下约为 **9M 参数**，明显小于参考默认 U-Net。

这一路径与当前 FM 都使用 attention，但仍有三个本质区别：它是 causal encoder-decoder Transformer、学习 epsilon diffusion，并把压缩后的 observation feature 当作 memory；当前 FM 是非 causal 的交替 cross/self-attention DiT、学习连续 velocity，并保留密集 observation tokens。

## 当前 FM Policy

关键源文件：

- `robot_policy/src/robot_policy/policies/fm.py`
- `robot_policy/src/robot_policy/policies/common.py`
- `robot_policy/src/robot_policy/encoders/observation.py`
- `robot_policy/src/robot_policy/config.py`

### 1. Token 构成

当前 FM 将每个连续 action row 线性投影到 hidden dimension，并加 learned action-position embedding。它还在序列前拼接 32 个 learned `future_tokens`。主干因此同时处理 latent/query tokens 和 action tokens，但输出头只读取最后的 action-token 部分。

观测不会先压成单个 global vector。策略保留每个相机的空间 token，并附加一个 state token；这些 token 经 learned projector 后作为 cross-attention 的 key/value。当前 FULL_VISION_512 配置对应 512 个视觉 token加 1 个 state token。

### 2. 交替 cross/self-attention DiT

`LayerwiseDiT` 的 block 按层交替：

- 偶数 block：action/latent queries 对 observation tokens 做 cross-attention；
- 奇数 block：action/latent tokens 之间做 self-attention；
- 每个 block 后都有 4x MLP residual branch；
- continuous time embedding 通过 AdaNorm 的 shift/scale 调制 attention 输入；
- 不使用 causal mask，因此整个 chunk 可双向联合建模。

DiT-S 为 384 hidden / 6 blocks / 4 heads，即 3 个 cross-attention blocks + 3 个 self-attention blocks。DiT-B 为 768 / 12 / 12，即各 6 个。当前研究已将 DiT-L 设为 legacy/load-only。

### 3. Flow-matching 目标与生成

训练时从高斯噪声 `x0` 与真实动作 `x1` 构造线性插值：

`x(t) = (1 - t) * noise + t * target`

网络预测常速度目标：

`v* = target - noise`

时间 `t` 不是均匀采样，而是由代码中的 Beta 分布变换得到。推理从纯高斯噪声开始，默认用 12 步显式 Euler：

`x <- x + (1 / steps) * v_theta(x, obs, t)`

这比 DDIM scheduler 更直接：没有 beta schedule、alpha cumulative product 或 epsilon-to-sample 变换。

### 4. 参数规模

以现有 FULL_VISION_512 B-spline FM checkpoints 为准，并且只统计 checkpoint 内可学习 policy 参数，不包含离线冻结的视觉 backbone：

| 模型 | 总参数 | observation projector/state | action backbone | embedding/output |
|---|---:|---:|---:|---:|
| DiT-S FM | 15,160,455 | 1,238,528 | 13,610,880 | 311,047 |
| DiT-B FM | 108,233,223 | 3,062,528 | 103,958,784 | 1,211,911 |

因此，当前 DiT-S FM 明显小于参考默认 U-Net generator；DiT-B FM 则更大。这个比较只说明生成策略容量，不等同于端到端系统总参数比较，因为两边 observation backbone 的训练/冻结和输入接口不同。

## RTC 对架构的影响

当前 continuousRTC 不增加另一套网络。它复用完全相同的 FM velocity model，只在输入状态与 loss mask 上加入 hard condition：

- 对受 delay 影响的 B-spline spans，固定 mask 精确等于这些 spans 的 basis-support control rows 并集；
- degree 为 3 时，`D` 个左端 spans 固定 control rows `[0, D+3)`；
- mask 外的 control rows 不从 previous plan 泄漏到模型；
- 每个 Euler step 前后都重新覆盖固定值，保证其数值严格不漂移。

joint discreteRTC 复用同一个几何 mask，只是固定的是离散 tokens，并在每个 unmasking round 保持 immutable。也就是说，continuousRTC 与 discreteRTC 的条件范围现在是同一份数学契约，而不是两个近似实现。

## 架构层面的判断

1. **参考默认 U-Net 的优势**是很强的局部时间平滑和多尺度卷积归纳偏置；短 horizon 上不需要让模型自己学习所有局部结构。
2. **当前 FM 的优势**是 observation token 保真度更高，cross-attention 能按 action query 选择空间证据，且连续 ODE 目标简洁，适合从头生成和硬条件 inpainting 使用同一个 velocity field。
3. **当前 FM 的风险**是 DiT-B 容量远大于参考默认 generator，并且 attention + AdaNorm 的数值稳定性更依赖优化设置；已观察到的超大梯度问题与这一点一致。
4. **参考可选 Transformer 不是当前 FM 的同构基线**。虽然都使用 attention，但它的 causal decoder 与 epsilon diffusion 改变了信息流和训练目标。
5. 若要做严格架构消融，最干净的实验不是比较两种 B-spline 编码，而是在相同 observation/action interface 下分别替换 `ConditionalUnet1D + DDIM` 与 `LayerwiseDiT + FM`，并对齐参数量、训练更新数和生成步数。

## 源码审计索引

- 参考 B-spline U-Net adapter：`RefCode/bspline-policy/bspline_policy/bspline_policy/policy/diffusion_unet_bspline_image_policy.py`
- 参考 B-spline Transformer adapter：`RefCode/bspline-policy/bspline_policy/bspline_policy/policy/diffusion_transformer_bspline_image_policy.py`
- 参考默认训练配置：`RefCode/bspline-policy/bspline_policy/bspline_policy/config/train_diffusion_unet_real_hybrid_bspline_workspace.yaml`
- 参考可选 Transformer 配置：`RefCode/bspline-policy/bspline_policy/bspline_policy/config/train_dp_transformer.yaml`
- 参考 U-Net implementation：`RefCode/bspline-policy/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py`
- 参考 Transformer implementation：`RefCode/bspline-policy/diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py`
- 当前 FM：`robot_policy/src/robot_policy/policies/fm.py`
- 当前 DiT blocks：`robot_policy/src/robot_policy/policies/common.py`
- 当前 observation-token interface：`robot_policy/src/robot_policy/encoders/observation.py`
