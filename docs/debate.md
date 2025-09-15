# 辩论循环（generate → critique → revise）

`hypoarena.debate` 实现发现循环中的辩论阶段。它只做编排，不判断内容好坏：
排序交给锦标赛，引用是否成立交给 grounding 校验器。

## 角色与协议

一个 `DebateLoop` 由三类角色组成，全部只需满足 `DiscoveryAgent` 协议：

- proposer：产出初始假设；
- critics（1..N 个）：对当前陈述给出批评；
- reviser：根据批评改写陈述（默认由 proposer 兼任）。

构造时会校验角色是否满足协议、critic 数量是否够用；不满足直接抛 `ValidationError`，
不会在运行中途才发现。

## 适配器约定

| 任务 | prompt | context |
| --- | --- | --- |
| `propose` | `config.proposal_prompt` | 语料/已有 claim 的上下文行 |
| `critique` | `config.critique_prompt.format(statement=...)` | 同上 |
| `revise` | **当前陈述本身** | 各 critic 的批评文本 |

`revise` 之所以把陈述放在 prompt、把批评放在 context，是为了让"没有改动"这件事可判定：
适配器只要原样返回 prompt，循环就知道收敛了。`replay_transcript` 与录制的 fixture
都遵循同一约定。

## 收敛

`DebateTurn.changed` 比较归一化后的陈述。若某一轮没有变化：

- `stop_on_unchanged=True`（默认）：立即停止，`converged=True`；
- `stop_on_unchanged=False`：跑满 `rounds`，`converged` 仍反映是否出现过不动点。

用脚本化适配器实测（`ScriptedAgent` 三个质量档，`rounds=5, critics=2`）：
所有档位都在**第 2 轮**收敛——第 1 轮改写生效，第 2 轮命中不动点。
档位之间的差别体现在最终陈述的具体程度（vague 只加 `(unchanged)`，
focused 收窄 scope，mechanistic 追加测量方式），长度严格递增；
这正是锦标赛恢复技能顺序所依赖的可测差异。

## 记账

`DebateResult.usage` 形如 `{"agents": {name: Usage.as_dict()}, "total": ...}`，
由 `merge_usage` 合并（同一对象兼任 proposer 与 reviser 时只记一次）。
调用次数满足 `1 + rounds_run × (critics + 1)`，测试逐配置核对该等式。
token 数只被记录，不计价。

## 上下文

`corpus_context(corpus, limit, max_chars)` 按文档 ID 顺序取每篇的第一句（超长截断加
省略号），因此同一语料总是给出同一上下文；`claim_context(claims, limit)` 取已有 claim
的陈述，用于让新一轮辩论看到当前假设集合。

## 产物

`hypoarena.serialize` 提供 `debate_result_to_dict` / `debate_result_from_dict`
（含 turns，适合嵌入运行元数据）与 `debate_to_lines` / `debate_from_lines`
（JSONL：`meta → turn* → result`）。头部计数为 turn 数，签名为整份编码的摘要；
缺少 result 行、计数或签名不符都会抛 `SchemaError`。

## 与图的衔接

`revised_claim(claim, result)` 用辩论结果生成新 claim：保留 ID 与 citation，
provenance 记为 `origin="agent"`、`generation+1`、`parents=(原 claim,)`、
`notes="debate:<rounds>"`。`apply_debate(graph, claim_id, result)` 就地替换并
调用 `graph.validate()`，因此 link 与边全部保留，图有效性由测试覆盖。
