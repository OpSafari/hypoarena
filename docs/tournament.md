# 锦标赛排序

`hypoarena.tournament` 把"很多条看似合理的假设"变成一个有审计轨迹的排序。
机制是成对比较：judge 给双方打 rubric 分，分数决定胜/负/平，
Elo（Bradley–Terry 形式）更新双方评分。

## Rubric

四个维度，每个都在 `[0, 1]`：

| 维度 | 含义 |
| --- | --- |
| `novelty` | 陈述的信息量（相对于已有假设的新意） |
| `testability` | 是否可以被实验检验（方向性关系、机制、scope 条件） |
| `grounding` | 引用是否成立（来自 grounding 校验的分数） |
| `consistency` | 与图中其他假设是否冲突 |

`RubricWeights` 决定各维度权重（默认 0.25/0.30/0.30/0.15），只要求非负且总和为正；
`normalized()` 会归一化，因此只差一个倍数的两套权重行为完全相同。
加权总分 `weighted_total` 始终落在 `[0, 1]`。

## 评分模型

`EloModel` 参数与默认值：`initial=1500`、`k_factor=32`、`k_decay=0.97`、
`k_floor=6`、`scale=400`、`draw_margin=0.05`。

- 期望胜率：`E(left) = 1 / (1 + 10^((right - left) / scale))`，两侧期望互补；
- 更新：`Δ = K × (score − E)`，`score ∈ {0, 0.5, 1}`，平局由 `draw_margin` 判定；
- K 调度：`K(n) = max(k_floor, k_factor × k_decay^n)`，`n` 是该主体已进行的场次，
  因此早期比赛移动大、后期只做微调（收敛性由测试固定：
  连胜时步长严格递减，后半场平均步长/前半场 < 1）；
- 双方场次相同时更新是零和的；场次不同（K 不同）时会有小额残余漂移，
  golden 测试记录了这个事实而不是假装精确守恒。

## 赛程

`round_robin_pairings(subjects, repeats, seed)`：每轮覆盖全部两两组合，
轮内顺序由 `Random(f"hypoarena:tournament:{seed}:{round}")` 打乱，
奇数轮交换左右两侧以抵消位置偏好。相同 seed 得到完全相同的赛程。

## Judge

- `PlantedJudge(qualities, noise, seed)`：按 claim id 给出植入质量，
  加入**确定性**的成对噪声（`Random(f"{seed}:{claim}:{opponent}")`）。
  它是排序恢复测试的 oracle，不代表任何模型判断。
- `FeatureJudge(reports, contradictions, novelty_scale)`：从可检查的特征打分，
  grounding 取校验分数（无报告时退化为"是否有引用"），testability 由
  方向性关系 / 机制 / scope 条件各加 0.2（基线 0.4），novelty 取陈述内容词数
  相对 `novelty_scale` 的比例，consistency 每条 `contradicts` 边扣 0.25。
  每个维度都能追溯到具体输入，因此排序可以逐条解释。

## 实测：能否恢复植入的技能顺序

配置：4 条 claim，植入质量 `0.95 / 0.70 / 0.45 / 0.20`，`repeats=3`，
seed 取 0..4，`order_recovery` 是 Kendall τ 归一化到 `[0, 1]`。

| judge 噪声 | 5 个 seed 的 recovery | 均值 |
| --- | --- | --- |
| 0.00 | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 |
| 0.10 | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 |
| 0.20 | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 |
| 0.35 | 1.000, 1.000, 1.000, 1.000, 0.833 | 0.967 |

同一组配置下（`repeats=6`，噪声 0.1）：`transitivity_rate = 1.000`，
`convergence_ratio = 0.557`（后半场平均步长是前半场的 55.7%），
最终评分极差 `319.2`。

这些数字由 `tests/test_tournament_recovery.py` 与
`tests/test_golden_tournament.py` 固定，描述的是**评分机制在合成质量信号上的行为**：
质量差异越大、噪声越小、轮次越多，恢复越准。它们不是任何真实模型或真实任务的评测结果。

## 审计轨迹

每场比赛保存为 `MatchResult`：双方 ID、四维 rubric 分、加权总分、outcome、
judge 名称、轮次与场次索引、seed。`replay_ratings(matches, model)` 只凭这份轨迹
就能重算出完全相同的榜单——测试逐 seed 核对，因此产物本身就是可验证的证据。

`hypoarena.serialize` 提供 `tournament_to_dict/from_dict` 与
`tournament_to_lines/from_lines`（行序 `meta → config → rating* → match*`，
头部计数为 ratings/matches，签名为整份编码的摘要）。

## 局限

- rubric 分数由 judge 决定：`FeatureJudge` 只反映"可检查特征"，
  不理解科学内容；换成真实模型 judge 时，排序质量取决于该模型，本仓库不为其背书。
- Elo 是顺序敏感的：同一批比赛换个顺序会得到略有不同的评分（K 随场次衰减所致）。
  赛程由 seed 固定，因此结果可复现，但不应把单次评分差值当作显著性结论。
- 平局判定基于加权总分差，`draw_margin` 需要根据 judge 的分辨率调整。
