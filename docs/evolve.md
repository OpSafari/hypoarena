# 演化算子

`hypoarena.evolve` 从已有 claim 派生新候选。所有算子都返回**新的不可变记录**，
从不原地修改父 claim；每个子 claim 的 provenance 记录算子名、父 ID 与代数。

## 算子

| 算子 | 做什么 | 图中的边 |
| --- | --- | --- |
| `narrow_scope` | 追加一个 scope 条件，陈述后缀 `under <condition>` | 父 `refines` 子 |
| `substitute_variable` | 替换 subject 或 object；陈述中若逐字出现该变量则同步改写 | 无边（不声称蕴含） |
| `flip_relation` | 换成对立关系（increases↔decreases、enables↔inhibits） | 父 `contradicts` 子 |
| `crossover` | 取第一父本的变量/关系/群体，第二父本的机制，条件与引用取并集 | 无边 |
| `decompose` | 按 `and` / `;` / `as well as` 拆分复合陈述，每个 conjunct 一个子 claim | 父 `entails` 子 |

拒绝规则（返回 `None` 或空元组，而不是抛错）：

- `flip_relation`：`causes` 与 `associates` 没有对立项；
- `crossover`：两亲本既无共享变量、或群体不同、或组合会让变量指向自身；
- `decompose`：拆不出两个可用 conjunct（含"拆分结果与父陈述等价"或"conjunct 重复"）；
- `narrow_scope`：条件已存在时直接返回父 claim（幂等，不产生冗余子代）。

## 新颖度闸门

`NoveltyGate` 包一个 `NoveltyGuard`（默认 content-only 词一元组 Jaccard，阈值 0.9）
在**归一化陈述**上判重。被接受的候选返回 `EvolutionRecord`（含可读的 `rationale`），
被拒绝的返回 `Rejection`（含 `reason`、最近邻 ID 与相似度），因此"为什么没收这个候选"
永远可查。`preload(claims)` 用图中已有 claim 预热，避免把既有假设当成新发现。

## 引擎

`EvolutionEngine(seed, operators, per_operator, novelty_config)`：

- `candidate(operator, claims, rng)` 用 `Random(f"hypoarena:evolve:{seed}:{generation}")`
  选择亲本与参数（条件来自 `CONDITION_POOL`，替换变量来自 `VARIABLE_POOL`）；
- `insert(graph, record)` 加入子 claim，并按上表补边，随后立刻 `graph.validate()`；
- `step(graph, generation)` 跑一代，返回 `EvolutionStep(accepted, rejected)`；
- `run(graph, generations)` 从图中最深的 provenance 代数继续编号。

## 被测试固定的性质

- **有效性保持**：任何算子、任何代之后 `graph.validate()` 都通过；边只连已存在的 claim；
  ID 不冲突；证据 link 数不变（演化不动证据）。
- **可追溯**：子 claim 的 `provenance.origin == "evolved"`、`parents` 全部在图中、
  代数严格大于每个父本。
- **增长有界**：每代最多新增 `len(operators) × per_operator` 条 claim。
- **确定性**：同 seed 同图 → 逐代 `signature()` 相同、最终图签名相同；不同 seed → 不同谱系。
- **新颖度饱和**：长时间只跑 `narrow_scope` 时，拒绝数 > 0，且拒绝原因只可能是
  `not novel` / `operator produced no child` / `child already present`。

## 局限

- `substitute_variable` 只在陈述**逐字**包含该变量时改写文本，否则只改字段；
  因此子 claim 的陈述可能与变量不完全一致，需要重新跑 grounding 校验。
- `decompose` 是句法切分，不理解从属结构；子 claim 继承父本的全部引用，
  这意味着"拆分后的每一半是否仍被同一 span 支持"必须由校验器重新判定，
  引擎不会替它背书。
- `crossover` 只是结构化组合，不产生新机制；机制文本直接来自某个亲本。
- 演化不评估真假：排序与筛选属于锦标赛与证据累积的职责。
