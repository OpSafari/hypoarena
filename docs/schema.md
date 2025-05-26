# Schema 参考（v1.0）

所有记录都是 frozen dataclass，构造即校验：非法记录无法存在，构造函数直接抛出
`hypoarena.errors` 中的具体异常。序列化在 `hypoarena.serialize` 中集中实现，
`hypoarena.codec` 提供严格解码（未知键、类型、范围与 schema 版本全部拒绝）。

## ID 约定

`<prefix>_<hex digest>`，前缀为 2-8 个小写字母，摘要为 8-64 位十六进制：

| 前缀 | 记录 | 生成方式 |
| --- | --- | --- |
| `doc` | 语料文档 | `make_id("doc", seed, index)` |
| `clm` | Claim | `make_id("clm", statement, subject, object, scope)` |
| `evd` | Evidence | `make_id("evd", document, start, end, statement)` |
| `agt` | Agent | `make_id("agt", agent 名称)` |

ID 由内容派生，因此同样的输入总是得到同样的 ID，重复插入会被图拒绝。

## 记录字段

### Claim

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `claim_id` | `str` | 必须是 `clm_` 前缀的合法 ID |
| `statement` | `str` | 非空白 |
| `subject` / `object` | `str` | 非空白，且归一化后不能相同 |
| `relation` | `PredictedRelation` | `increases`/`decreases`/`enables`/`inhibits`/`causes`/`associates` |
| `scope` | `Scope` | `population` 非空白，`conditions` 非空白且不重复 |
| `citations` | `tuple[Citation, ...]` | 可为空（由 grounding 校验分级），同一 span 不能重复引用 |
| `mechanism` | `str \| None` | 给定时非空白 |
| `provenance` | `Provenance` | 见下 |

### Evidence

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `evidence_id` | `str` | 必须是 `evd_` 前缀的合法 ID |
| `statement` | `str` | 非空白 |
| `polarity` | `EvidencePolarity` | `support`/`refute`/`neutral` |
| `strength` | `float` | 有限值且落在 `[0, 1]`；`neutral` 必须为 `0.0` |
| `citations` | `tuple[Citation, ...]` | 至少一条，且不重复 |
| `method` | `str` | 非空白，例如 `synthetic_finding` |
| `provenance` | `Provenance` | 见下 |
| `effect_size` | `float \| None` | 给定时必须有限 |
| `sample_size` | `int \| None` | 给定时 `>= 1` |

`Evidence.weighted_polarity` 返回 `[-1, 1]` 的带符号权重（support 为正、refute 为负、
neutral 为 0），是信念累积的直接输入。

### Scope / Citation / Provenance

- `Scope(population, conditions)`：`narrowed(condition)` 幂等地追加条件，
  `is_narrower_than(other)` 要求同 population 且条件是严格超集（refines 边的依据）。
- `Citation(document_id, start, end, quote)`：半开区间，`end > start >= 0`，
  `quote` 非空白且长度不超过 span；`overlaps(other)` 判断同文档区间相交。
  quote 是否真的出现在该 offset 由 grounding 校验对照语料判断，schema 只管结构。
- `Provenance(origin, agent_id, generation, parents, seed, corpus_hash, notes)`：
  `origin ∈ {synthetic, agent, user, evolved}`；`agent` 必须给 `agent_id`，
  `evolved` 必须给 `parents`；`corpus_hash` 为 8-64 位十六进制。
  `is_reproducible` 表示合成来源同时记录了 seed 与 corpus_hash。

## 序列化约定

- 一行一个 JSON 对象（JSONL），UTF-8，键按字典序，分隔符固定为 `,` 与 `:`，
  因此相同记录总是产生逐字节相同的行。
- 顶层记录（Claim / Evidence）带 `schema_version`；嵌套值对象不带，避免冗余。
- 可选字段即使为 `None` 也显式写出 `null`，保证键集合稳定。
- 解码时未知键、缺失键、错误类型与不匹配的 schema 版本都会抛 `SchemaError`。
- 产物中不允许出现凭据形状的键（测试会递归扫描 `token`、`secret`、`api_key` 等）。

示例（节选自 `tests/test_golden_schema.py` 的 golden 行，已按可读性换行说明）：

```python
from hypoarena.schema import PredictedRelation, Scope
from hypoarena.serialize import claim_to_line

claim = sample_claim(scope=Scope("hek293 cells", ("hypoxia",)))
line = claim_to_line(claim)
assert line.endswith("\n")
assert PredictedRelation.INCREASES.value in line
```

## 版本策略

schema 版本采用 `主.次` 字符串。新增可选字段是 minor 变更，读取旧产物时保持兼容；
重命名或删除字段、改变语义属于 major 变更，`check_schema_version` 会直接拒绝旧载荷，
需要在 `CHANGELOG.md` 中给出迁移说明。
