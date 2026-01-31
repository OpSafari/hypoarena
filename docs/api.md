# API 约定

`hypoarena` 的公开接口遵循统一约定，便于组合与测试。

## 命名与类型

- 所有 schema 都是 `@dataclass`（多数为 `frozen=True` 或带显式校验方法），字段全部带类型标注。
- `from_dict` / `to_dict` 成对出现：`to_dict` 输出可直接 `json.dumps` 的纯 JSON 结构，
  `from_dict` 对未知键、缺失键与类型错误严格报错。
- 校验失败抛出 `hypoarena.errors` 中的具体异常类型，而不是 `ValueError` 裸异常。
- 任何随机行为都通过显式 `seed` 参数或注入的随机源控制，不读取全局状态。

## 稳定性

- 0.x 阶段公开 API 可能在 minor 版本间调整；调整会记录在 `CHANGELOG.md`。
- `hypoarena.__all__` 与 `docs/api.md` 的模块表是接口面的单一来源，
  由 golden 测试 `tests/test_golden_public_api.py` 守护。
- 以 `_` 开头的模块与名称是内部实现，不保证兼容。

## 模块索引

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| `hypoarena._version` | 已实现 | 版本号（`__version__`、`VERSION_TUPLE`） |
| `hypoarena.cli` | 已实现 | 参数解析与子命令入口（`main`、`build_parser`） |
| `hypoarena.errors` | 已实现 | 异常层级，每个错误带稳定 `code` 与 `exit_code`（`ERROR_CODES`、`error_for_code`） |
| `hypoarena.ids` | 已实现 | 确定性哈希与 ID（`canonical_json`、`stable_hash`、`content_hash`、`hash_parts`、`make_id`、`is_valid_id`） |
| `hypoarena.codec` | 已实现 | 严格解码与规范 JSONL 行（`require_*`、`reject_unknown_keys`、`check_schema_version`、`dumps_line`、`loads_line`） |
| `hypoarena.schema` | 已实现 | 记录与枚举（`Claim`、`Evidence`、`Citation`、`Scope`、`Provenance`、`PredictedRelation`、`EvidencePolarity`、`ClaimRelation`） |
| `hypoarena.graph` | 已实现 | 假设—证据图（`HypothesisGraph`、`ClaimEdge`、`GraphStats`），见 [graph.md](graph.md) |
| `hypoarena.serialize` | 已实现 | 记录 ↔ dict/JSONL 编解码（`claim_to_dict`、`graph_to_text`、`graph_from_lines` 等） |
| `hypoarena.corpus` | 已实现 | 文档与语料容器、span 解析与采样（`Document`、`Corpus`、`CorpusStats`、`sample_spans`），见 [corpus.md](corpus.md) |
| `hypoarena.synthetic` | 已实现 | 植入真值的合成文献工厂（`SyntheticConfig`、`generate`、`build_bundle`、`SyntheticBundle`、`PlantedTruth`），见 [synthetic.md](synthetic.md) |
| `hypoarena.grounding` | 已实现 | span 级 grounding 校验（`GroundingVerifier`、`VerifierConfig`、`GroundingReport`、`GroundingFlag`），见 [grounding.md](grounding.md) |
| `hypoarena.agents` | 已实现 | agent 协议与离线适配器（`ScriptedAgent`、`ReplayAgent`、`RecordingAgent`、`replay_transcript`、`Usage`），见 [agents.md](agents.md) |
| `hypoarena.http_agent` | 已实现 | OpenAI 兼容 HTTP 适配器（`HttpConfig`、`HttpAgent`、`ChatRequest`、`ChatResponse`），默认仅允许 loopback |
| `hypoarena.debate` | 已实现 | 辩论循环与记录（`DebateLoop`、`DebateConfig`、`DebateResult`、`DebateTurn`、`corpus_context`、`apply_debate`），见 [debate.md](debate.md) |
| `hypoarena.text` | 已实现 | 归一化与分词（`normalize`、`tokenize`、`content_tokens`、`char_ngrams`、`word_ngrams`、`extract_numbers`、`sentence_split`、`has_negation`） |

记录字段与校验规则详见 [schema.md](schema.md)。后续模块（graph、corpus、
grounding、agents、tournament、dedup、evolve、belief、runner、artifacts、reports）
落地后会补充到本表。
