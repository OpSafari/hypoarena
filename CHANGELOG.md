# Changelog

本项目遵循语义化版本；0.x 阶段的破坏性变更会在 minor 版本说明中显式列出。

## 0.1.3

- 新增 `hypoarena.debate`：角色化辩论循环（propose → critique → revise）、`DebateConfig`、`DebateResult`/`DebateTurn` 转录、确定性上下文提取与 `apply_debate`。
- 新增 `hypoarena.tournament`：rubric 打分、Bradley–Terry/Elo 模型（平局、K 因子调度）、确定性赛程、`PlantedJudge`/`FeatureJudge` 与完整对局审计轨迹。
- 新增 `hypoarena.dedup`：规范化文本签名、字符/词 n-gram Jaccard、TF-IDF 余弦、MinHash LSH、聚类与 `NoveltyGuard`，并在合成改写簇上给出实测 recall/precision。
- 新增 `hypoarena.evolve`：保图演化算子（scope 收窄、变量替换、关系翻转、crossover、decomposition）与新颖度闸门 `EvolutionEngine`。
- 新增 `hypoarena.belief`：odds 空间的贝叶斯证据累积、分级似然模型、矛盾处理策略、先验敏感性与单调性性质。
- 新增编排层 `hypoarena.config`/`artifacts`/`cost`/`stages`/`runner`：`RunConfig` 指纹、原子产物存储与断点、token 记账（只计数不计费）、`StageResult`/`RunSummary` 与可续跑且逐字节一致的 `Pipeline`。
- `hypoarena.serialize` 扩展：锦标赛/去重/辩论/演化/信念/配置与运行摘要的编解码。
- 文档：debate、tournament、dedup、evolve、belief 与 runner 编排。
- 测试：确定性与收敛扫描、Elo 对植入技能序的恢复、去重实测指标、信念单调性与 golden、对抗性 grounding、续跑逐字节一致。

## 0.1.2

- 新增 `hypoarena.graph`：claim/evidence/link/edge 存储、遍历、子图与合并、`validate()` 不变量扫描、`GraphStats` 与内容签名。
- 新增 `hypoarena.corpus`：不可变文档、精确 span 解析（`resolve` / `verify_quote` / `search_quote`）、统计与签名、子语料与冲突检测合并、按 seed 的 span 采样。
- 新增 `hypoarena.synthetic`：植入因果链、竞争假设、矛盾陈述、改写簇与干扰文档的合成语料工厂，输出带精确 citation 的 gold claim / evidence 与 `SyntheticBundle.graph()`。
- 新增 `hypoarena.grounding`：span 解析、实体重合、极性与数字一致性检查，分级 flag（grounded / weakly_grounded / ungrounded / fabricated）与打分。
- 新增 `hypoarena.agents`：`DiscoveryAgent` 协议、`BaseAgent` 记账模板、`ScriptedAgent` 质量档位、`ReplayAgent`（keyed/sequence）、`RecordingAgent` 与 `replay_transcript`。
- 新增 `hypoarena.http_agent`：OpenAI 兼容 chat-completions 客户端、可注入 transport、线性退避重试、超时与严格响应解析；默认仅允许 loopback，凭据全程掩码。
- `hypoarena.serialize` 扩展：图/语料/植入真值/grounding 报告/HTTP 配置的编解码，统一的 JSONL `meta` 头（计数 + 签名）与原子文件写入。
- 文档：架构、schema、语料与序列化、grounding、合成语料、agent 适配器。
- 测试：结构性质扫描（ seeded 随机图与语料）、对抗性 grounding（编造引用/位移 span/漂移数字/极性翻转全部被拦）、合成语料上的实测 grounding 率、loopback mock 集成。

## 0.1.1

- 新增 `hypoarena.errors`：带稳定 `code` 与 `exit_code` 的异常层级，以及 `ERROR_CODES` 注册表。
- 新增 `hypoarena.ids`：规范 JSON、SHA-256 内容哈希、多段哈希与 ID 构造/校验。
- 新增 `hypoarena.text`：归一化链、分词、停用词、字符/词 n-gram、数字抽取、句子切分与否定线索。
- 新增 `hypoarena.codec`：严格解码（未知键、类型、范围、schema 版本）与规范 JSONL 行编解码。
- 新增 `hypoarena.schema`：`Scope`、`Citation`、`Provenance`、`Claim`、`Evidence` 与关系/极性枚举。
- 新增 `hypoarena.serialize`：记录与 dict/JSONL 的编解码层。
- 文档：架构总览、使用指南、API 约定、schema 参考、诚实性说明、参考文献、贡献与安全策略。
- CI：3.11/3.12/3.13 矩阵（build / test-all / format-check / lint / typecheck）、wheel 冒烟任务与发布工作流。
- 测试：golden 摘要、golden 配置、golden 公开 API、golden 文本归一化、golden schema JSONL 与标记卫生检查。

## 0.1.0

- 初始骨架：src 布局打包、Makefile 任务（build / test / test-all / format / lint / typecheck / release）、
  pytest 标记（`slow`、`model`）、ruff 与 mypy 配置。
- `hypoarena.__version__` 与 `hypoarena._version` 暴露版本号。
