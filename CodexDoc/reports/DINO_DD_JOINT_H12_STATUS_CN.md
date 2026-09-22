# DINOv2 Joint Discrete Diffusion h1/h2 当前状态

日期：2026-09-21

## 实验范围

当前矩阵共有 8 个 checkpoint：

| 动作表示 | 观测历史 | Base | ttRTC |
|---|---:|---:|---:|
| raw action | h1 | 50k updates | 5k updates |
| raw action | h2 | 50k updates | 5k updates |
| cubic B-spline token | h1 | 50k updates | 5k updates |
| cubic B-spline token | h2 | 50k updates | 5k updates |

所有变体都使用 joint categorical discrete diffusion，不再包含 layer-wise
DD。训练 corruption 为 50% 全 MASK 序列与 50% 原有 block corruption，因而
训练过程会直接看到从零生成时使用的全 MASK 初始状态。推理使用 8 轮基于
置信度的 block unmasking，block size 为 21。

## 图像观测架构

冻结的视觉 backbone 是 DINOv2 ViT-L/14
（`vit_large_patch14_reg4_dinov2`），读取倒数第二层特征。本实验族不使用
SigLIP。每个相机、每个时刻的处理如下：

1. DINOv2 输出 256 个、维度为 1,024 的 patch token；
2. 每个相机分别使用一个可训练的 32-query cross-attention resampler；
   global 与 hand camera 的 resampler 参数不共享；
3. 两组 32 个视觉 token 后面再接一个投影后的机器人 state token。

因此 h1 共有 65 个 observation token，h2 共有 130 个。h2 会完整保留时间
轴与相机轴，并加入确定性的时间位置编码；两个时刻不会被平均或压成一个
global vector。

## 数据与缓存

- 数据集：`Data/stacking_cups_action_30hz`
- 划分：42 个 train / 5 个 validation / 5 个 test episode
- 共享缓存：`robot_policy/outputs/DINO_DD_JOINT_H12/cache/dinov2_patch16`
- 覆盖范围：52 个 episode、31,706 帧
- 每帧保存形状：2 cameras x 256 tokens x 1,024 channels，FP16
- 缓存大小：约 31 GiB

DINOv2 特征、模型权重缓存、W&B 文件、checkpoint 与日志都保存在 scratch
工作区，没有向 home 目录移动实验数据。

## 已完成验证

- 通过 74 个选定的 repository tests；唯一排除的旧测试依赖已经按要求删除
  的 36-checkpoint SWEEP 目录。
- raw/B-spline x h1/h2 共 4 个 base GPU smoke training 全部通过。
- 4 个 RTC GPU smoke training 全部通过，包括历史观测索引，以及 raw 与
  B-spline hard-mask 路径。
- 所有 smoke checkpoint 均通过保存、重新加载与 generation-from-scratch
  validation。
- 静态 diff 检查通过。

## 正在运行的训练

当前只占用 GPU 0--3；GPU 4--6 保持空闲，供其他 agent 的评估使用。

| GPU | 变体 | W&B run |
|---:|---|---|
| 0 | raw h1 base | [84mf4xkm](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/84mf4xkm) |
| 1 | raw h2 base | [e8e5u39c](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/e8e5u39c) |
| 2 | B-spline h1 base | [44ah3k06](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/44ah3k06) |
| 3 | B-spline h2 base | [gr8i2pn6](https://wandb.ai/401910710-university-of-california-berkeley/robot-policy-dino-dd-joint-h12-basic/runs/gr8i2pn6) |

训练初期所有数值均有限，没有 optimizer step 被跳过。日志中的 gradient
norm 是裁剪前的值，当前已启用 `grad_clip=1.0`。每个 base 完成后，launcher
会为其建立按 checkpoint hash 隔离的 parent-prediction cache，然后自动开始
对应的 ttRTC finetune。

运行状态记录在
`robot_policy/outputs/DINO_DD_JOINT_H12/status.json`，各变体日志位于
`robot_policy/outputs/DINO_DD_JOINT_H12/logs`。

四个任务目前均使用 batch 64、accumulation 1，保持 effective batch size 64
不变，同时提高单卡利用率。每 500 updates 会完整评估 3,077 个 validation
windows。早期只跑短 validation 的 W&B run 作为历史记录保留；上表列出的是
用于权威完整验证曲线的 continuation runs。最近一次检查时，raw h1/h2 已超过
12.5k updates，B-spline h1/h2 已超过 15.5k updates；所有 optimizer update
均为有限值，没有 step 被跳过。

已再次审计本地 DD-OpenVLA 的 D2F 实现：训练 mask 概率随 block 单调不减，
每个目标 block 的 attention 只能读取 observation、前序 block 与本 block。
当前 joint-DD 具有相同的 block-causal 信息边界，并把完成的 block 写入 KV
cache。这里有意不复制 D2F 的 teacher-distillation loss：本实验使用 hard
categorical target，并加入 50% 全 MASK exposure 来匹配从零生成的部署状态。

## 下一步

1. 持续监控 50k base 训练的完整 validation generation-from-scratch
   action MSE 与梯度稳定性；
2. 完成 4 个 exact-parent cache 和 4 个 5k ttRTC finetune；
3. 对 8 个 checkpoint 完成 train/test open-loop generation、prefix
   inpainting RTC、latency 与真实部署接口测试；
4. 发布最终中英文对比报告与 checkpoint bundle。
