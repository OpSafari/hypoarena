# 示例总览

`examples/` 下的每个示例都是**完全离线、确定性**的脚本：不联网、不调用真实模型、
不下载数据集、不写产物文件。它们各自演示一条机制，并打印由脚本**真实运行**得到的度量。
`tests/test_examples.py` 会在测试套件里逐个运行它们，断言退出码为 0、输出包含承诺的
度量标记，并带有"合成数据"声明。

运行方式（仓库根目录）：

```bash
.venv/bin/python examples/<name>/<script>.py
```

## 三个示例

### planted_corpus/verify_grounding.py

用 `synthetic.build_bundle` 生成带植入因果链的合成语料与 gold claim，用
`grounding.GroundingVerifier` 校验整张图并打印 grounding 摘要；再把一条 gold claim 的
引用改指向语料中不存在的文档，演示校验器把它标成 `fabricated`。对应模块：
[synthetic.md](synthetic.md)、[grounding.md](grounding.md)。

### tournament/elo_recovery.py

从合成 bundle 取真实 gold claim 作为选手，**由脚本植入**一组递减质量分，交给带确定噪声的
`PlantedJudge` 两两比较，用 `Tournament` 跑 Elo，报告 `order_recovery` / `kendall_tau` /
`transitivity_rate`。对应模块：[tournament.md](tournament.md)。

### evolution/novelty_cycle.py

从合成 bundle 构建假设图，用 `evolve` 的演化算子（受新颖度闸门约束）扩充若干代，并用
`dedup.DuplicateFinder` 在演化前后度量重复簇与重复率，报告图增长与被闸门拦下的数量。
对应模块：[evolve.md](evolve.md)、[dedup.md](dedup.md)。

## 诚实边界

三个示例里的所有数字都来自**合成植入**数据：

- grounding 摘要描述词面启发式在合成语料上的判定，不是对真实文献的核验；
- Elo 恢复率衡量评分更新规则在植入质量分上的性质，**不**代表任何真实语言模型的排序能力；
- 演化增长与新颖度刻画算子在合成植入图上的行为，不代表真实科学发现。

想要一条命令跑完整条流水线，用 CLI 的 `demo`（见 [cli.md](cli.md)）：它在小的植入语料上
端到端运行并打印"恢复 vs 植入"的因果链。
