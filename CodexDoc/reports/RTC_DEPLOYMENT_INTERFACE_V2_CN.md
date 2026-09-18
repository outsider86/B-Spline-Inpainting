# RTC 部署接口验证

日期：2026-09-18

## 结论

当前 msgpack/WebSocket 部署接口可以正确支持两种主动动作表示的 RTC 推理。
验证使用真实策略服务器、公开 `PolicyClient`、GPU 上的真实 DiT-S ttRTC
checkpoint 以及 localhost 网络序列化，而不只是直接调用 Python wrapper。

| 动作表示 | 测试策略 | 必需的历史输入 | delay=3 结果 |
| --- | --- | --- | --- |
| raw | FM ttRTC | 物理坐标 `prev_action_chunk`，`[1,30,7]` | 有限 `[1,30,7]`；固定 3 个动作行 |
| B-spline | joint-DD ttRTC | 归一化 `prev_control_rows`，`[1,18,7]` | 有限 `[1,30,7]`；影响 2 个 span，固定 5 个三次样条支撑控制行 |

服务器在握手元数据中声明接口约定。`PolicyClient` 读取
`rtc_requires_previous_field`，并通过 `infer_realtime` 发送对应字段。延迟单位是
30 Hz raw action step，严格限制为 1–10。Base checkpoint 会拒绝 realtime
请求。服务器不会在客户端之间共享计划缓存；每个请求都显式携带上一段计划。

## B-spline 的关键限制

B-spline RTC 必须接收上一请求返回的归一化参数行，并会主动拒绝解码后的
30×7 动作。重新拟合和三次样条支撑扩展会把 raw delay 3 映射为 2 个受影响
span 和 5 个保留控制行，从而避免把 raw action 索引误当作样条控制行索引。

## 本次加固

真实网络测试发现 msgpack 解码的 NumPy buffer 可能是只读的。部署边界现在会先
创建自有的 float32 副本，再把历史动作或控制行转换成 torch tensor，避免传输
buffer 别名并消除 PyTorch 的只读数组警告。版本化 Hugging Face 目录现在也能
自动发现 `vN/sidecars/{raw,bspline}`。

## 证据

- `outputs/FULL_VISION_512/summary/deployment/rtc_websocket_validation.json`
- `outputs/FULL_VISION_512/summary/deployment_validation_dit_{s,b}.json`
- 完整项目测试，包括只读传输数组与版本化 release sidecar 自动发现。

该验证仅证明策略服务器推理接口可用，不等同于闭环机器人安全验证或任务成功率。
