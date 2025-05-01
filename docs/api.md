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
| `hypoarena.text` | 已实现 | 归一化与分词（`normalize`、`tokenize`、`content_tokens`、`char_ngrams`、`word_ngrams`、`extract_numbers`、`sentence_split`、`has_negation`） |

后续模块（schema、graph、corpus、grounding、agents、tournament、dedup、evolve、
belief、runner、artifacts、reports）落地后会补充到本表。
