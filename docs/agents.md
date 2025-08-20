# Agent 适配器

`hypoarena` 的发现循环不绑定任何模型：循环只依赖 `DiscoveryAgent` 协议
（`propose` / `critique` / `revise`，底层统一为 `respond(request)`）。
仓库内置三种适配器，**全部可离线运行**。

| 适配器 | 来源 | 用途 |
| --- | --- | --- |
| `ScriptedAgent` | 由请求内容与 `quality` 档位决定回复 | 默认测试与示例；用档位植入技能顺序 |
| `ReplayAgent` | JSONL fixture 中记录的回复 | 回归测试：把一次真实对话固定下来 |
| `HttpAgent` | OpenAI 兼容的 chat-completions 接口 | 接入真实服务；测试只连本地 loopback mock |

## 请求与响应

`AgentRequest(task, prompt, context, temperature, max_tokens)` 带一个由内容派生的
`request_id`；`AgentResponse` 必须回显该 id，否则 `BaseAgent` 抛 `AdapterError`
（"adapter replied to a different request"），防止适配器串号。

`AgentResponse` 同时携带 `prompt_tokens` / `completion_tokens`。离线适配器用
空白分词数作为 token 代理（`count_words`），因此 `Usage` 记账在任何适配器下都非零且可复现。
**token 数只被记录，从不计价**：包内没有任何价格表，也不会把成本写进产物。

## ScriptedAgent 的质量档位

`quality ∈ [0, 1]` 映射到三个档位，回复的具体程度随档位单调增加：

| 档位 | 区间 | propose | critique | revise |
| --- | --- | --- | --- | --- |
| vague | `[0, 1/3)` | 忽略上下文，给泛化陈述 | "需要更多支持" | 原样返回（加 `(unchanged)`） |
| focused | `[1/3, 2/3)` | 用上下文实体造句 | 指出缺少可检验的 assay | 收窄 scope |
| mechanistic | `[2/3, 1]` | 加机制与测量方式 | 同时指出机制、assay、scope 三处缺口 | 收窄 scope 并给出测量方式 |

锦标赛测试正是用这个阶梯植入技能顺序：同一请求下，高档位回复严格更具体、更长，
因此 Elo 能否恢复顺序是一个可判定的性质，与任何模型能力无关。

## ReplayAgent

两种模式：

- `keyed`：按 `(task, prompt, context)` 精确匹配；未命中时抛 `ReplayExhaustedError`，
  或返回构造时给定的 `fallback` 文本。重复 key 在构造时就被拒绝。
- `sequence`：按 fixture 顺序发牌，发完即抛 `ReplayExhaustedError`（带 `served` 计数），
  task 不匹配抛 `AdapterError`；`reset()` 可倒回开头。

fixture 是 JSONL：首行 `meta`（`counts.entries` + 内容签名），随后每行
`{"record": "replay", "entry": {...}}`。`RecordingAgent` 可以包住任意适配器，
把一次真实对话录成 fixture，从而"跑一次、之后永远离线复现"。

## HttpAgent

- 传输层是可注入的 `Transport` 协议；默认 `UrllibTransport` 只用标准库。
- 重试策略来自 `HttpConfig`：`retry_statuses`（默认 408/429/500/502/503/504）、
  `max_retries`、线性退避 `retry_backoff × attempt`（无 jitter，保证可复现）。
  连接级失败同样计入重试预算；预算耗尽后抛 `TransportError(status=..., attempts=...)`。
- **默认只允许 loopback**：`check_endpoint` 在非 `127.0.0.1` / `localhost` / `::1`
  的 base URL 上直接抛错，除非显式设置 `allow_remote=True`。仓库内的测试与示例
  从不设置它。
- 凭据卫生：`api_key` 只出现在 `Authorization` 头里；`redacted()` 用 `***` 掩码，
  `fingerprint()` 基于掩码视图计算，序列化（`http_config_to_dict`）写出的也是掩码视图，
  反序列化永远不恢复凭据。

## 测试方式

`tests/loopback.py` 在 `127.0.0.1` 的临时端口上起一个 `ThreadingHTTPServer`，
按队列返回 `(status, body)`，可注入延迟以触发超时。测试覆盖：正常回复、
503/429 后恢复、500 耗尽预算、404 不重试、非法 JSON、缺少 choices、超时。
这些是**适配器协议测试**，不是模型评测：mock 返回的文本是写死的。
