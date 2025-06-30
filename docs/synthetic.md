# 合成文献工厂

`hypoarena.synthetic` 生成**完全合成**的论文式语料，并在其中植入已知的因果结构。
它存在的唯一目的，是给 grounding 校验、去重、锦标赛与信念累积提供一个有 oracle 的
可度量目标。语料不含真实文献内容，任何在其上得到的数字都只描述机制在这份合成数据上的行为。

## 植入了什么

| 结构 | 说明 |
| --- | --- |
| 因果链 | `chains × (chain_length - 1)` 条链接，方向从 `CAUSAL_RELATIONS` 中随机抽取 |
| 改写簇 | 每条链接有 `paraphrases_per_link` 个不同表述，各自成为一篇文档 |
| 竞争假设 | 同一变量对上换一个关系（`competing`），是"另一种解释"而不是改写 |
| 矛盾陈述 | 保留植入关系但用否定句式（`contradiction`），只有极性检查能抓住 |
| 干扰文档 | 不含任何植入变量的方法学/批次说明文本 |

## 生成流程

```python
from hypoarena.synthetic import SyntheticConfig, build_bundle

config = SyntheticConfig(seed=270106, chains=3, chain_length=3)
bundle = build_bundle(config)
print(bundle.summary())
```

`generate(config)` 用一个 `Random(f"hypoarena:synthetic:{seed}")` 驱动全部抽样，
顺序固定，因此同一配置总是得到逐字节相同的语料。`SyntheticConfig.expected_documents()`
给出精确文档数，测试会核对实际生成数量，避免生成器悄悄缩水。

## 产物

`SyntheticBundle` 把一次生成的全部内容放在一起：

- `corpus`：`Corpus`，每篇文档带 `kind` / `chain` 元数据；
- `truth`：`PlantedTruth`，含链、竞争假设、矛盾链接、改写簇（文档 ID 分组）与干扰文档 ID；
- `claims`：gold claim，每条植入链接与其竞争假设各一条，citation 指向真实 span；
- `evidence`：每条 finding 一个证据项（否定句为 `refute`，其余为 `support`）；
- `findings`：`PlacedFinding`，记录句子、文档与其精确 offset；
- `graph()`：把上述记录组装成 `HypothesisGraph`，并给竞争假设加 `contradicts` 边。

因为 citation 是在拼装文档文本时按字符偏移记录的，`corpus.resolve(citation)`
对每条 gold citation 都成立——这正是 grounding 测试需要的"应当通过"样本；
把 quote 改一个字符、把 offset 挪一位，就得到"应当拒绝"的对抗样本。

## 强度与可复现性

证据强度取自固定表 `SUPPORT_STRENGTHS`（否定项统一为 `CONTRADICTION_STRENGTH`），
不走随机流，因此贝叶斯累积的测试可以精确预测后验。所有记录的 provenance 都写入
`seed` 与 `corpus_hash`，`is_reproducible` 为真。

## 序列化

`hypoarena.serialize` 提供 `truth_to_dict` / `truth_from_dict` 与
`truth_to_lines` / `truth_from_lines`（行类型 `meta → chain → competing →
contradiction → cluster`，头部带计数与签名）。注意 JSONL 形式不保存
`distractor_ids`：那是语料的属性，重建后的 truth 中该字段为空元组。

## 边界

- 词表有限（几十个实体、若干模板），改写簇超出模板组合空间时用"Replicate N"形式补足；
- 文本是英文模板句，不包含真实数据、真实作者或真实结论；
- 生成器不联网、不读取任何外部数据集。
