# 去重与新颖度

`hypoarena.dedup` 提供四层递进的重复检测，全部离线、确定性：

| 方法 | 度量 | 适用场景 |
| --- | --- | --- |
| `exact` | 归一化文本的 SHA-256 摘要 | 完全重复（大小写、标点、空白差异） |
| `jaccard` | 字符/词 n-gram 的 Jaccard | 小规模、需要精确分数 |
| `tfidf` | TF-IDF 向量的余弦相似度 | 词频重要性差异大的语料 |
| `minhash` | MinHash 估计的 Jaccard + LSH 分带 | 大规模候选集 |

`shingle_unit` 选择字符或词 n-gram；`content_only=True` 会先丢掉功能词与
"报告动词"（observed / show / assays / data 等，见 `hypoarena.text.REPORTING_WORDS`），
只保留实体与关系词。**这一项决定了改写簇能否被分开**（见下方实测）。

## MinHash 与 LSH 的误差

- 签名长度 `num_perm` 决定估计误差：标准误约为 `1 / sqrt(num_perm)`
  （64 → 0.125，256 → 0.0625）。阈值附近的判定应据此选择 `num_perm`。
- LSH 分带（`bands × rows = num_perm`）只是**候选生成器**：
  相似度为 `s` 的一对文本成为候选的概率是 `1 - (1 - s^rows)^bands`，
  拐点约为 `(1 / bands)^(1 / rows)`。
- 任何候选都会用真实度量再验证一次，只有达到 `threshold` 才并入簇。
  因此分带只影响召回与耗时，不会让"仅碰撞"的对进入结果。

## 聚类与新颖度

`DuplicateFinder.find(texts)` 用并查集把已验证的对合并成簇，簇内成员与簇本身都排序，
输出与输入字典顺序无关。`DuplicateFinder.report(texts)` 额外带上配置与总数，
产出 `DedupReport`（`duplicate_rate`、`cluster_of`、`metrics_against(truth)`、`signature`）。

`NoveltyGuard` 是演化循环用的增量版本：`check(text)` 给出最相似的已见文本与分数，
`admit(identifier, text)` 只在新颖时登记，因此 seen 集合不会被重复项撑大。

## 实测：能否找回植入的改写簇

语料：`SyntheticConfig(seed=270106, chains=2, chain_length=3,
paraphrases_per_link=3, distractor_documents=4)`，即 12 篇植入 finding（4 簇 × 3）
加 4 篇干扰文档；`num_perm=256, bands=64`。指标按**标识符对**统计
（`cluster_metrics`），因此"把一簇拆成两半"和"把两簇并成一簇"是两种不同的错误。

| 配置 | clusters | precision | recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| `exact` | 0 | 1.000 | 0.000 | 0.000 | 0 | 12 |
| minhash, char-3, 表面, t=0.5 | 4 | 0.400 | 0.500 | 0.444 | 9 | 6 |
| minhash, word-1, 表面, t=0.6 | 4 | 0.750 | 0.500 | 0.600 | 2 | 6 |
| minhash, word-1, **content-only**, t=0.6 | 4 | **1.000** | **1.000** | **1.000** | 0 | 0 |
| jaccard, word-1, content-only, t=0.6 | 4 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| minhash, word-1, content-only, t=0.7 | 4 | 1.000 | 0.833 | 0.909 | 0 | 2 |
| tfidf, word-1, 表面, t=0.6 | 4 | 0.364 | 0.667 | 0.471 | 14 | 4 |

把竞争假设与矛盾陈述也放进待去重的池子（它们**故意**使用同一组变量）后，
同样的 content-only 配置：t=0.6 → precision 0.500 / recall 1.000；
t=0.7 → precision 0.625 / recall 0.833。

## 这些数字说明什么，不说明什么

- 说明：在**合成**语料上，只要 shingle 忽略报告措辞，改写簇可以被完全找回；
  阈值升高会牺牲召回而不牺牲精度；`exact` 对改写完全无效。
- 不说明：真实文献中的"同一发现"往往跨术语体系、跨物种、跨指标，
  表面相似度不足以判定；此时需要语义向量或人工规则，本仓库不提供也不评测这类能力。
- 竞争假设与矛盾陈述被并入同一簇是**表面方法的固有局限**：它们描述同一对变量，
  词面高度重合。区分"改写"与"关于同一对象的不同主张"需要关系/极性信息，
  这正是 grounding 校验与图中 `contradicts` 边的职责，不该指望去重解决。

## 序列化

`hypoarena.serialize` 提供 `dedup_report_to_dict/from_dict` 与
`dedup_report_to_lines/from_lines`（行序 `meta → report`，头部计数为
clusters/members，签名为整份编码的摘要）。配置一并写入产物，
因此"这次去重用了什么阈值与方法"永远可追溯。
