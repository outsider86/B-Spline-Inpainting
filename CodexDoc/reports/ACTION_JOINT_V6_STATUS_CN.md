# ActionJoint V6 训练状态

## 实验矩阵

本轮输出统一放在：

`/scratch/wangpc/B-Spline-Inpainting/robot_policy/output/NEW`

每个 task 同时使用 6 张 GPU 训练 6 个 BSP-UNet Flow-Matching 模型：

- observation state：`LastCommand_Joint`（7D）、`State_Joint`（7D）、
  `State_LastCommand_Joint`（14D）；
- action representation：raw 与 uniform left-clamped B-spline；
- observation horizon：h1，即同一时刻的 global/hand 两张图像；
- action：7D 实际 joint angle。

task 的严格执行顺序为 `hanging_mug` → `stacking_cup` →
`classify_blocks`。

## 训练与验证协议

- base：100 epochs，batch size 64；
- base validation：只在 epoch 10、20、…、100 对完整 validation split 做
  generation-from-scratch action MSE；
- best base：从上述 10 个完整 validation 点中选择；
- ttRTC：从 best base 权重开始，使用全新 optimizer 训练 5 epochs；
- 最终每个 leaf 只保留 `base.pt`、`ttrtc.pt`，以及对应 log、training JSON、
  server JSON 和 manifest；所有 resume、epoch snapshot、rolling-best sidecar
  在验证后删除；
- 不使用 test split，也不向 W&B 上传 artifact。

## W&B

每个 task 使用一个独立 project，base 与 ttRTC 位于同一 project，通过
stage/group 和 run name 区分：

- `robot-policy-bsp-unet-v6-hanging-mug`
- `robot-policy-bsp-unet-v6-stacking-cup`
- `robot-policy-bsp-unet-v6-classify-blocks`

## Cache 共享原则

每个 task 的三种 observation-state 数据拥有相同的视频和 action 时序。
launcher 会对全部 MP4 及 action/timestamp/frame/episode 数组做 SHA-256
一致性验证，通过后仅共享一份 RGB84 cache。action cache 中包含各版本独立的
state normalization，因此仍按 state variant × representation 分开生成。

当前权威进度见：

`robot_policy/output/NEW/V6_STATUS.json`

`hanging_mug` 的 6 个 base 与 6 个 ttRTC 已全部完成。launcher 已将完整
validation 上的最优权重发布为最终 `base.pt`/`ttrtc.pt`，并清理全部 resume、
periodic 与 best-epoch 中间文件；12 个 run 均在线记录于 hanging-mug 独立
W&B project，且 optimizer skip 均为 0。

按完成即上传的要求，`hanging_mug` 子树已提前上传到
`DiscreteRTC/dRTC/NewModel/V6/hanging_mug`。task 级严格审计确认 12/12 模型
可加载、配套 JSON 完整且没有训练中间 checkpoint；远端共 12 个模型和 24 个
JSON，模型 LFS SHA-256 为 12/12 一致，无缺失文件。对应 dataset revision 为
`c827f2b04799be020897b60159279d6c1825a1b2`。全部 task 完成后仍会再次执行
36 模型全量审计与整树远端校验。

当前 `stacking_cup` 的 6 个 base run 正在 GPU 0--5 并行训练，且全部写入
`robot-policy-bsp-unet-v6-stacking-cup`。epoch 10/20/30/40/50 的完整
validation 均覆盖全部 4,211 个 validation windows。六路在 epoch 50 均刷新
各自 best，generation action MSE 范围为 0.0180482--0.0198387；训练期间没有
非有限 loss/gradient 或 optimizer skip，现继续推进至 epoch 60。

`stacking_cup` 后续已完成全部 6 个 base 和 6 个 ttRTC。base 最终选择 epoch
依次为：LastCommand raw/B-spline = 70/100，State raw/B-spline = 100/70，
State+LastCommand raw/B-spline = 60/100；ttRTC 均从对应 best base 开始，并从
5 次完整 validation 中选择最终权重。全部 12 个 run 的 optimizer skip 为 0，
且训练中间 checkpoint 已清零。

`stacking_cup` 子树也已提前上传至
`DiscreteRTC/dRTC/NewModel/V6/stacking_cup`。task 级审计为 12/12 模型通过，
远端 12 个模型与 24 个 JSON 均存在，模型 LFS SHA-256 为 12/12 一致。对应
revision 为 `b59cfae040213ed1028bc32dc954c10f1efaf4e3`。

最后一个 task `classify_blocks` 的 6 个 base 与 6 个 ttRTC 也已全部完成。
base 最终选择 epoch 依次为：LastCommand raw/B-spline = 90/70，State
raw/B-spline = 70/80，State+LastCommand raw/B-spline = 60/20；ttRTC 选择
epoch 依次为 5/5、5/5、5/4。全部 validation 均覆盖 9,467 个 validation
samples，所有 run 的 optimizer skip 均为 0。

## 最终 validation-best 结果

表中数值为完整 validation split 上的 generation-from-scratch action MSE；
`e` 表示被发布 checkpoint 对应的 epoch。

| Task | Observation state | Representation | Base | ttRTC |
|---|---|---:|---:|---:|
| hanging_mug | LastCommand | raw | e100 / 0.0115253636 | e5 / 0.0115644013 |
| hanging_mug | LastCommand | B-spline | e90 / 0.0117124693 | e5 / 0.0116665477 |
| hanging_mug | State | raw | e90 / 0.0125336456 | e5 / 0.0126129311 |
| hanging_mug | State | B-spline | e80 / 0.0126252889 | e5 / 0.0124728527 |
| hanging_mug | State+LastCommand | raw | e70 / 0.0116072344 | e5 / 0.0115960392 |
| hanging_mug | State+LastCommand | B-spline | e100 / 0.0114557928 | e1 / 0.0115313689 |
| stacking_cup | LastCommand | raw | e70 / 0.0187636974 | e4 / 0.0185712790 |
| stacking_cup | LastCommand | B-spline | e100 / 0.0175009139 | e5 / 0.0175499370 |
| stacking_cup | State | raw | e100 / 0.0189064756 | e5 / 0.0190488806 |
| stacking_cup | State | B-spline | e70 / 0.0194656161 | e1 / 0.0199390450 |
| stacking_cup | State+LastCommand | raw | e60 / 0.0186401576 | e5 / 0.0187613057 |
| stacking_cup | State+LastCommand | B-spline | e100 / 0.0176345984 | e5 / 0.0176392303 |
| classify_blocks | LastCommand | raw | e90 / 0.0233113019 | e5 / 0.0230967354 |
| classify_blocks | LastCommand | B-spline | e70 / 0.0225643763 | e5 / 0.0224542118 |
| classify_blocks | State | raw | e70 / 0.0255008774 | e5 / 0.0255440672 |
| classify_blocks | State | B-spline | e80 / 0.0250842212 | e5 / 0.0251315308 |
| classify_blocks | State+LastCommand | raw | e60 / 0.0235361312 | e5 / 0.0235573525 |
| classify_blocks | State+LastCommand | B-spline | e20 / 0.0229856817 | e4 / 0.0226660629 |

## 最终审计与上传

- 本地最终 inventory：18 个 `base.pt` + 18 个 `ttrtc.pt`；没有 resume、
  periodic、epoch/step snapshot 或 rolling-best weights 残留。
- 独立 CPU load 与配置审计：36/36 通过。确认 action 输出为 7D joint
  angle，state 为对应的 7D/14D，h1 每样本恰好使用同一时刻 global/hand 两张
  图，B-spline 为 cubic uniform-left，且每个 ttRTC 的 parent 指向对应 base。
- 完整 validation 样本数为 hanging 2,722、stacking 4,211、classify 9,467；
  base 有 10 次完整 validation，ttRTC 有 5 次完整 validation。
- Hugging Face 目录 `DiscreteRTC/dRTC/NewModel/V6` 已包含 36 个模型与 72 个
  server/training JSON，共 108 个文件；远端没有缺失，36 个模型的 LFS
  SHA-256 与本地全部一致。
- 最终 dataset revision：
  `08a54d5c34cda652cf29f8451afd94ec0228c5a6`。

本轮 V6 ActionJoint 训练与发布目标已完成。后续 open-loop、deployment 和
GT-prefix RTC 评估属于下一阶段，不混入本轮训练完成条件。
