# 流水线编排

`hypoarena.runner` 把各个机制串成一条可复现的离线流水线。它本身不判断内容
好坏：grounding、去重、锦标赛、演化、信念累积都各自负责一段，runner 只负责
**按固定顺序执行、落盘、记账、断点续跑**。

## 阶段顺序

`STAGES` 定义了唯一合法的执行顺序：

```
corpus → generate → verify → dedup → debate → rank → evolve → accumulate → report
```

`RunConfig.stages` 可以是它的一个子序列（顺序必须保持一致），用于只跑一部分
流水线。`RunConfig` 收集全部可调项：`seed`、`run_id`、`corpus`、`grounding`、
`dedup`、`debate`、`tournament`、`belief`、`evolution`。`fingerprint()` 对每一个
会影响产物的设置做哈希，因此两次运行只要指纹相同，产物就逐字节相同。

## 产物目录

一次运行写入一个 `ArtifactStore(root, run_id)` 目录：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 运行元数据（版本、配置指纹、注入的时间戳、seed） |
| `corpus.jsonl` / `truth.jsonl` / `graph.jsonl` | 合成语料、植入真值、初始假设图 |
| `candidates.jsonl` | generate 阶段提出的候选 claim |
| `grounding.jsonl` | 每条 claim 的 grounding 报告 |
| `dedup.jsonl` | 去重簇报告 |
| `debates.jsonl` | 辩论转录 |
| `tournament.jsonl` | 锦标赛对局与评分审计轨迹 |
| `evolution.jsonl` | 每一步演化算子的接受/拒绝记录 |
| `beliefs.jsonl` | 每条 claim 的后验信念 |
| `summary.json` / `cost.json` | 运行摘要与 token 记账 |
| `report.json` | 最终报告载荷（含 limitations） |
| `checkpoints/<stage>.done` | 断点续跑标记 |

所有写入都是原子的（临时文件 + rename），JSONL 首行是带计数与签名的 `meta`
头。序列化路径不含任何密钥。

## 断点续跑

`Pipeline.run()` 逐阶段执行；每个阶段完成后调用 `store.mark_stage`。再次以同一
`run_id` 运行时，已完成阶段被跳过（`StageResult.skipped = True`），对应的
`restore_*` 处理器从产物重新载入状态。**续跑产生的 `report.json` 与一次跑完
逐字节相同**——包括 token 记账：ledger 会先从 `cost.json` 恢复历史用量，再累加
本次进程内的开销，因此报告不会因为分两次跑而少记。

运行元数据会在第一个阶段开始前写入，因此阶段中途失败后仍能校验原配置。续跑会
先检查包版本、schema、配置指纹和 checkpoint 前缀；允许用相同配置从较短阶段前缀
继续到更长前缀，但拒绝复用由不同 seed 或其他设置产生的产物。不带 `--resume` 的
新运行会先清除旧 checkpoint，避免较短的新运行误用先前留下的下游完成标记。

## token 记账

`CostLedger` 记录每个阶段的 prompt/completion token 数，只做**计数**：没有单价、
没有计费、不向任何外部服务发送数据。`totals()` 与 `by_stage()` 进入
`cost.json` 和 `report.json`，用于诚实标注"这次运行消耗了多少 token"。

## 报告与诚实性

`run_report(pipeline)` 汇总计数、grounding 摘要、去重、排名前若干、信念前若干、
演化统计、植入链恢复率与 token 用量，并**内嵌 `REPORT_LIMITATIONS`**：声明语料
全部合成、排名只反映评审 rubric 而非科学真伪、grounding 是词面启发式、信念是
约定似然下的记账、token 只计数不计费。任何由 runner 产出的报告都携带这段限制
说明。
