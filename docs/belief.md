# 证据累积

`hypoarena.belief` 把分级证据折算成假设的后验概率。计算在 odds 空间进行：
每条证据贡献一个似然比，后验 = 归一化后的乘积。这样做有两个可直接测试的性质：
**证据顺序不影响结果**，**同极性证据总是朝同一方向推动信念**。

## 似然模型

`LikelihoodModel(support_ratio=3.0, refute_ratio=3.0, neutral_ratio=1.0)`：

- support（强度 `s`）：odds × `support_ratio ** s`
- refute（强度 `s`）：odds × `refute_ratio ** -s`
- neutral：odds × `neutral_ratio`（默认 1.0，即无影响）

按强度做指数插值是**约定**，不是任何真实实验系统的标定结果。
概率端点会被夹到 `[1e-9, 1-1e-9]`，因此信念永远可被新证据修正。

## 矛盾处理策略

`ContradictionPolicy` 决定"既有支持又有反驳"时怎么办：

| 策略 | 行为 |
| --- | --- |
| `ignore` | 反驳证据不贡献任何比率（基线） |
| `downweight` | 反驳比率取 `ratio ** downweight_factor`，仍向下推但更温和 |
| `discount`（默认） | 反驳数达到 `contradiction_threshold` 后，**所有**比率取 `** downweight_factor`，把后验拉回先验附近 |
| `reject` | 反驳数达到阈值后，后验直接落到下限 |

## 手算 golden

先验 0.5、`support_ratio = 3` 时：

| 证据 | 似然比 | 后验 |
| --- | --- | --- |
| 1 条满强度 support | 3 | 0.75 |
| 2 条满强度 support | 9 | 0.90 |
| 1 条满强度 refute | 1/3 | 0.25 |
| support + refute | 1 | 0.50 |
| 1 条 0.5 强度 support | √3 | 0.634 |
| 先验 0.25 + 1 条满强度 support | 1 | 0.50 |

这些值由 `tests/test_golden_belief.py` 固定，任何对更新规则的改动都会立刻暴露。

## 实测：策略如何改变排序

语料：`SyntheticConfig(seed=11, chains=2, chain_length=3)` → 4 条植入 claim
（其中 2 条各有 3 条证据：2 支持 + 1 反驳；另 2 条各 2 条：1 支持 + 1 反驳），
2 条竞争 claim（各 1 条支持证据）。各类策略下的平均后验：

| 策略 | 植入 claim 均值 | 竞争 claim 均值 | 排名第一的是植入 claim |
| --- | --- | --- | --- |
| `ignore` | 0.8523 | 0.7061 | 是 |
| `downweight` | 0.8274 | 0.7061 | 是 |
| `discount` | 0.7436 | 0.7061 | 是 |
| `reject` | 0.4330 | 0.7061 | 是（但均值低于竞争 claim） |

结论很直接：**矛盾处理策略会决定"证据多但被反驳过"的假设能否排在"证据少但没被反驳"
的假设前面**。`reject` 把被反驳的植入 claim 直接压到下限，于是竞争 claim 的平均后验
反而更高；即便如此，排名第一的仍是那条没有被反驳的植入 claim。
这不是"哪个策略正确"的答案，而是提醒：策略是建模选择，必须显式记录
（`BeliefConfig.fingerprint()` 会写入运行元数据）。

## 先验敏感性

`prior_sensitivity(evidence, priors)` 在固定证据下扫描先验网格
（默认 `(0.1, 0.25, 0.5, 0.75, 0.9)`），返回 `(prior, posterior)` 序列；
`sensitivity_spread` 给出最大后验差，`is_prior_robust` 判断结论是否在所有先验下同号。
上表语料中一条植入 claim 的 spread 实测为 **0.7843**：证据不足以让结论脱离先验，
这类数字应当写进报告，而不是只报一个后验。

## 与图的衔接

`accumulate_graph(graph, config)` 按 claim ID 顺序读取每条 claim 关联的证据
（证据按 ID 排序，保证与插入顺序无关），没有证据的 claim 保留先验并出现在结果里，
这样报告不会静默丢掉未被检验的假设。`rank_beliefs` / `belief_table` 给出排序与表格行，
`state_for` 按 ID 取单条状态。

## 局限

- 似然比是**人为设定**的，不是从数据估计的；换个领域就要重新设定，
  本仓库不提供也不声称任何标定。
- 证据被当作条件独立处理；同一实验重复发表、同一 span 被多条证据引用都会重复计权。
  去重（`hypoarena.dedup`）与图中的重复 link 检查是缓解手段，不是证明。
- 后验只表示"在这套记账规则下的相对支持度"，不是科学真理的概率。
