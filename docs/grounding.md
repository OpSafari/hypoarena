# Grounding 校验

`hypoarena.grounding` 判断一条 claim 是否**真的**由它引用的语料支持。
它只使用机械可判定的启发式，不判断科学真伪。

## 分级结果

| flag | 触发条件 |
| --- | --- |
| `grounded` | 至少一条引用，全部解析成功且没有任何 issue |
| `weakly_grounded` | 引用都能解析，但存在软性问题（overlap 过低、极性冲突、数字不符、引文过短） |
| `fabricated` | 任一引用无法解析：文档不存在、offset 越界、或该 offset 上的文本与 quote 不一致 |
| `ungrounded` | 一条引用都没有 |

一条 claim 只要有一条引用是伪造的，整体就判为 `fabricated`；
"部分引用正确"不会让伪造引用蒙混过关。

## 检查项

| 检查 | 实现 | 失败时的 issue |
| --- | --- | --- |
| span 解析 | 文档存在 + `0 <= start < end <= len(text)` + `text[start:end] == quote` 逐字符相等 | `missing_document` / `span_out_of_range` / `quote_mismatch` |
| 引文长度 | `len(quote.strip()) >= min_quote_length`（默认 12） | `short_quote` |
| 实体重合 | claim 的 `subject`/`object` 内容词与 quote 内容词的 Jaccard，阈值默认 0.34 | `low_entity_overlap` |
| 极性一致 | claim 与 quote 的否定线索必须同侧（`negation_flip`） | `polarity_conflict` |
| 数字一致 | claim 中出现的每个数字都要在 quote 中出现（容差默认 0） | `numeric_mismatch` |

阈值都在 `VerifierConfig` 中，且有 `fingerprint()` 写入运行元数据，
因此"某次运行用了什么阈值"是可追溯的。

## 打分

每条引用：解析成功且无 issue 记 1.0，解析成功但有软性问题记 0.5，未解析记 0.0；
claim 的 `score` 是其引用得分的平均值（保留 4 位小数）。分数只用于排序与报告，
判定仍由上表的分级规则给出。

## 实测数字（合成语料）

配置 `SyntheticConfig(seed=S, chains=2, chain_length=3)`：16 篇文档、
6 条 gold claim（4 条植入链接 + 2 条竞争假设）、12 条证据。
在 seed 131 / 977 / 270106 上分别运行：

| 输入 | grounded | weakly | ungrounded | fabricated | grounded_rate |
| --- | --- | --- | --- | --- | --- |
| gold claim（未篡改） | 6 | 0 | 0 | 0 | 1.0 |
| span 整体右移 1 字符 | 0 | 0 | 0 | 6 | 0.0 |
| quote 换成编造文本 | 0 | 0 | 0 | 6 | 0.0 |
| 去掉全部引用 | 0 | 0 | 6 | 0 | 0.0 |

这些数字由 `tests/test_grounding_measured.py` 固定，说明的是：
在这份**合成**语料上，校验器对"真实引用"零误杀、对三类篡改零漏放。
它们不是任何真实文献抽取任务的准确率。

## 已知限制

- 实体重合基于词袋 Jaccard，不理解同义改写；同义词替换会把 grounded 降为 weakly。
- 极性检查只识别否定线索，不识别"抑制/激活"这类方向词与关系标签的冲突。
- 数字比较忽略单位（`12%` 与 `12 fold` 视为同一个数），也不理解区间与显著性表述。
- 逐字符相等意味着标点或大小写差异会导致 `quote_mismatch`：这是刻意的严格，
  真实抽取系统需要在生成引用时保留原文，而不是事后重写。
