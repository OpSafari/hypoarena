# 假设—证据图

`hypoarena.graph.HypothesisGraph` 是一次发现运行的共享状态：claim 是节点，
evidence 挂在它所支持的 claim 上，claim 之间用带类型的有向边连接。

## 记录与索引

| 结构 | 说明 |
| --- | --- |
| `_claims` | `claim_id -> Claim`，ID 唯一 |
| `_evidence` | `evidence_id -> Evidence`，ID 唯一 |
| `_links` / `_backlinks` | claim ↔ evidence 的双向索引，必须对称 |
| `_edges` / `_incoming` | 出边与入边索引，必须对称 |

对外只暴露只读视图：`claims`、`evidence_items`、`edges`、`link_pairs()` 都按 ID
或 `(source, target, relation)` 排序，因此遍历顺序与插入顺序无关。

## 边类型

| 关系 | 含义 | 对称性 |
| --- | --- | --- |
| `entails` | 源 claim 蕴含目标 claim | 有向 |
| `contradicts` | 两者不能同时成立 | 视为对称：反向重复添加会被拒绝 |
| `refines` | 目标是源的细化（scope 更窄或机制更具体） | 有向 |

## 不变量

`validate()` 一次性检查以下全部条件，任何一条被破坏都会抛
`GraphInvariantError` 并带上定位信息：

1. 索引键与记录自身的 ID 一致；
2. 每条 link 的两个端点都存在，且双向索引对称、无重复；
3. 每条边的 source/target 都是图中已有的 claim，边不能是自环；
4. 同一条 `(source, target, relation)` 边只出现一次，出入索引对称；
5. 边必须挂在正确的 source 分组下。

所有变更方法都在写入前检查前置条件（重复 ID → `DuplicateIdError`，缺失端点 →
`UnknownReferenceError`），`subgraph()`、`merge()` 与反序列化在返回前再调用一次
`validate()`。因此"任何算子都保持图有效性"不是约定，而是被测试覆盖的性质
（见 `tests/test_graph_properties.py`）。

## 组合算子

- `subgraph(claim_ids, include_evidence=True)`：只保留选中的 claim，跨出选择范围的边被丢弃；
  evidence 记录不可变，因此按引用复制。
- `merge(other) -> int`：把另一张图并入当前图，返回新增记录数。ID 相同且内容相同的记录跳过；
  ID 相同但内容不同视为冲突并抛错——静默选边会伪造 provenance。
- `merged(other)`：返回新图，不修改任何输入。
- `replace_claim(claim)`：保留 ID 不变地替换内容，link 与边全部保留（演化算子依赖这一点）。
- `remove_claim(claim_id)`：级联删除所有相关的边与 link，但保留 evidence 记录本身，
  因为同一条观测可以支持其他 claim。

## 序列化

`hypoarena.serialize` 提供两种形式：

- `graph_to_dict` / `graph_from_dict`：单个 JSON 对象，适合嵌入运行元数据；
- `graph_to_lines` / `graph_from_lines`：JSONL 文档，行顺序为
  `meta → claim → evidence → link → edge`，即依赖先于引用者，单次前向扫描即可重建。

`meta` 行记录 `schema_version`、各类记录计数与 `graph.signature()`。读取时默认校验计数与签名，
文档被截断或篡改会立刻报 `SchemaError`，而不是得到一张静默残缺的图。

## 复杂度

| 操作 | 复杂度 |
| --- | --- |
| `add_claim` / `add_evidence` | O(1) |
| `link_evidence` | O(k)，k 为该 claim 已有的 link 数 |
| `edges` / `claims` 等排序视图 | O(n log n) |
| `has_path` | O(V + E)，确定性 BFS |
| `validate` | O(V + E + L) |
| `subgraph` / `merge` | O(V + E + L) 加上排序开销 |
