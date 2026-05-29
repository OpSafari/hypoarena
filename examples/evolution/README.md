# evolution — 演化算子扩充假设图并用去重度量新颖度

完全离线、确定性。运行（仓库根目录）：

```bash
.venv/bin/python examples/evolution/novelty_cycle.py
```

脚本从合成 bundle 构建假设图，先用 `dedup.DuplicateFinder` 度量初始的重复情况，再用
`evolve.EvolutionConfig().engine()` 跑若干代演化（scope 收窄、变量替换、crossover、
decomposition，均受新颖度闸门约束），最后再次度量重复，报告图的增长与新颖度变化。

实际输出（seed=5，3 代，由本脚本真实运行得到）：

```text
evolution + dedup novelty cycle (synthetic, offline, seed=5)
  claims before=9  after=18  grew=9
  generations=3  accepted=9  rejected(novelty/invalid)=6
  duplicate clusters before=0  after=1
  duplicate_rate before=0.000  after=0.111
note: numbers describe the operators on a synthetic planted graph only; no claim about real discovery.
```

**诚实边界**：`accepted` 是通过新颖度与有效性闸门、真正加入图的算子产物；`rejected`
是被闸门拦下的（重复或非法）。这些数字只刻画演化算子在**合成植入图**上的行为，不代表
任何真实科学发现过程。
