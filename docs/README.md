# 文档索引

从这里进入 `hypoarena` 的各个文档。内容为中文，技术名词保留英文。第一次使用请先读
[honesty.md](honesty.md) 与 [usage.md](usage.md)。

## 总览

- [architecture.md](architecture.md) — 分层、流水线阶段与模块职责
- [api.md](api.md) — 模块索引与公开接口约定
- [honesty.md](honesty.md) — 诚实性说明：度量了什么、没有度量什么
- [references.md](references.md) — 机制来源的上游参考与引用约定

## 数据与 schema

- [schema.md](schema.md) — Claim / Evidence / Citation / Scope / Provenance 记录与校验
- [graph.md](graph.md) — 假设—证据图的存储、遍历与有效性
- [corpus.md](corpus.md) — 文档、语料容器与 span 解析
- [synthetic.md](synthetic.md) — 植入真值的合成文献工厂

## 机制

- [grounding.md](grounding.md) — span 级 grounding 校验与分级 flag
- [agents.md](agents.md) — agent 协议与离线适配器
- [debate.md](debate.md) — generate → critique → revise 辩论循环
- [tournament.md](tournament.md) — rubric、Elo / Bradley–Terry 与审计轨迹
- [dedup.md](dedup.md) — 去重与新颖度（Jaccard / TF-IDF / MinHash LSH）
- [evolve.md](evolve.md) — 演化算子与新颖度闸门
- [belief.md](belief.md) — 贝叶斯证据累积与矛盾策略
- [ranker.md](ranker.md) — 可选可训练 ranker（torch extra）

## 编排与接口

- [runner.md](runner.md) — 流水线编排、产物、断点续跑与 token 记账
- [reports.md](reports.md) — 报告渲染为 Markdown 与自包含 HTML
- [cli.md](cli.md) — 命令行子命令、公共参数与退出码
- [usage.md](usage.md) — 安装、快速开始与产物约定
- [examples.md](examples.md) — 三个离线示例总览
- [testing.md](testing.md) — 测试策略、标记与离线保证
