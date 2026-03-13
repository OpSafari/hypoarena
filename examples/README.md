# examples

每个示例都是**完全离线**的脚本：不联网、不调用真实模型、不下载数据集、不写产物文件。
示例使用合成语料（synthetic corpus），其中人工植入了因果链、竞争假设与改写簇，因此
输出里的"恢复率 / 精度 / 增长率"等数字只描述机制在这份合成数据上的表现，绝非任何真实
模型的基准成绩。

运行方式（在仓库根目录）：

```bash
.venv/bin/python examples/<name>/<script>.py
```

## 示例清单

| 目录 | 脚本 | 展示 |
| --- | --- | --- |
| `planted_corpus/` | `verify_grounding.py` | 生成植入语料并用 grounding 校验器打分；演示编造引用被判为 `fabricated` |
| `tournament/` | `elo_recovery.py` | 用带植入质量分的 `PlantedJudge` 跑 Elo 锦标赛，度量对植入技能序的恢复 |
| `evolution/` | `novelty_cycle.py` | 用演化算子扩充假设图，并以去重度量新颖度增长 |

每个示例目录都附带自己的 README，内含由该脚本**真实运行**得到的输出与诚实边界说明。
`tests/test_examples.py` 会在测试套件里逐个运行这些示例，断言它们离线跑通、退出码为 0，
并打印出各自承诺的度量标记与"合成数据"声明。

想要一键端到端演示，直接用命令行：

```bash
hypoarena demo --chains 2 --chain-length 2 --out /tmp/hya
```
