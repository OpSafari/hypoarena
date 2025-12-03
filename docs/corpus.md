# 语料与序列化

`hypoarena.corpus` 提供 grounding 校验所依赖的“真值文本”，
`hypoarena.serialize` 提供记录、图与语料的编解码。

## Document

| 字段 | 说明 |
| --- | --- |
| `document_id` | `doc_` 前缀的确定性 ID |
| `title` | 非空白标题 |
| `text` | 全文，citation 的 offset 都指向它 |
| `source` | 来源标签，例如 `synthetic:v1`；用于报告与统计 |
| `attributes` | 有序的 `(key, value)` 元组，例如 `("chain", "c1")` |

文档不可变。修改文本会让所有指向它的 citation 失效，因此更新语料的正确做法是
构建新的 `Corpus`。常用方法：`span_text(start, end)`、`find_all(quote)`、
`locate(quote)`、`citation_spans(quote)`、`contains_span(citation)`。

## Corpus

- `add_document` 拒绝重复 ID（`DuplicateIdError`）与非 `Document` 对象；
- `document_ids` / `documents` / `__iter__` 全部按 ID 排序，遍历与插入顺序无关；
- `resolve(citation)` 是 grounding 的真值来源：文档存在、offset 合法、且该区间字符
  **逐字符等于** `quote`，否则抛 `SpanNotFoundError`，错误里同时带上 `quoted` 与 `found`，
  便于定位“数字漂移”“改写”“编造引用”这三类问题；
- `verify_quote(citation)` 是不抛异常的布尔版本，适合批量扫描；
- `search_quote(quote)` 返回全语料中所有匹配位置，按文档 ID 与 offset 排序；
- `stats()` 给出文档数、字符数、最长/最短与来源分布；
- `signature()` 是内容摘要，写入 provenance 的 `corpus_hash`，用于校验
  “这条 claim 是否真的在它声称的语料上被验证过”；
- `subcorpus(ids)` 与 `merged(other, on_conflict=...)`：同 ID 同内容视为共享，
  同 ID 不同内容默认报错（`CorpusError`），因为静默选边会让 citation 指向错误文本；
- `sample_spans(document, rng, count)`：按词边界采样不重叠的 citation，
  随机性全部来自调用方传入的 `rng`，因此固定 seed 可复现同一批 span。

## 序列化格式

所有产物都是 JSONL：一行一个对象，键按字典序，分隔符固定，因此相同内容总是产生
逐字节相同的文件。

| 文档类型 | 行顺序 | 头部计数 |
| --- | --- | --- |
| graph | `meta → claim → evidence → link → edge` | claims / evidence / links / edges |
| corpus | `meta → document` | documents / characters |

`meta` 行结构统一（`record`、`schema_version`、`counts`、`signature`），由
`meta_line()` 生成、`check_document_meta()` 校验。读取时若头部与真实内容不符
（文档被截断、被手工编辑、或来自另一个版本），立刻抛 `SchemaError`，
而不是静默重建出一个残缺对象。

文件读写使用 `write_graph` / `read_graph` / `write_corpus` / `read_corpus`：
先写同名 `.partial` 文件并 `fsync`，再原子重命名，因此中断不会留下半个产物。

## 安全约定

序列化只写记录自身的字段，未知键在解码时被拒绝；测试会递归扫描产物键名，
确保不会出现 `token`、`secret`、`api_key` 之类形状的字段。
