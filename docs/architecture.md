# 架构总览

`hypoarena` 把 generate–debate–evolve 式发现流水线拆成可独立测试的阶段。
每个阶段只依赖上一阶段的**数据产物**（dataclass + JSONL），因此可以单独替换实现，
也可以从任意检查点续跑。

## 流水线阶段

```text
corpus ──> claims/evidence ──> graph ──> grounding ──> debate ──> tournament
                                     │                              │
                                     └──── evolve <── dedup <───────┘
                                                │
                                          accumulate ──> report
```

| 阶段 | 输入 | 输出 | 关键不变量 |
| --- | --- | --- | --- |
| corpus | 配置 + seed | 合成文档与 span | 相同 seed 得到逐字节相同的语料 |
| graph | claim / evidence | 有向图 | 无重复 ID；边端点必须存在 |
| grounding | claim + corpus | 分级 flag | 编造的引用永远不会判为 grounded |
| debate | agent 协议 | 修订后的 claim | 相同脚本响应得到相同轨迹 |
| tournament | 成对比较 | Elo/Bradley–Terry 评分 | 评分更新有界；审计轨迹完整 |
| dedup | 文本集合 | 簇与新颖度 | 指标可复现；误报/漏报行为有文档 |
| evolve | 图 + 算子 | 新 claim | 每个算子保持图有效性 |
| accumulate | 证据 + 先验 | 后验信念 | 单调性；矛盾处理策略显式 |
| report | 全部产物 | Markdown / HTML | 不可信文本被转义；含限制说明 |

## 分层

- **数据层**：dataclass schema、校验规则、JSONL 序列化与安全过滤。
- **机制层**：grounding、去重、评分、演化、信念累积等纯函数式实现，
  全部接受显式随机源（seed），不读取全局状态。
- **适配层**：agent 协议与脚本化 / 回放 / HTTP 适配器。默认使用离线实现，
  HTTP 适配器只在本地 loopback mock 上测试。
- **编排层**：runner、artifacts、cost 记账与检查点。
- **接口层**：CLI 与报告渲染。

## 确定性

所有随机性来自显式 seed；排序使用稳定键（ID 或字典序）。
因此“一次跑完”和“分段检查点续跑”必须产生逐字节相同的产物，这一性质由测试保证。

## 当前模块

| 模块 | 职责 |
| --- | --- |
| `hypoarena._version` | 版本号单一来源 |
| `hypoarena.cli` | 命令行入口与参数解析 |

模块表随实现推进补充；接口约定见 [api.md](api.md)。
