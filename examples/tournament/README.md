# tournament — Elo 锦标赛恢复植入的技能序

完全离线、确定性。运行（仓库根目录）：

```bash
.venv/bin/python examples/tournament/elo_recovery.py
```

脚本从合成 bundle 取若干真实 gold claim 作为选手，**由脚本自己植入**一组递减的质量分
（0.95 → 0.20），交给带确定噪声的 `PlantedJudge` 两两比较，用 `Tournament` 跑 Elo，
再报告恢复出的排名与植入排名的吻合程度（`order_recovery`、`kendall_tau`、
`transitivity_rate`）。

实际输出（seed=11，噪声 0.1，repeats=4，由本脚本真实运行得到）：

```text
Elo tournament recovery (synthetic, planted qualities, offline)
  subjects=5  matches=40  repeats=4  noise=0.1
  planted qualities (best->worst): [0.95, 0.8, 0.6, 0.4, 0.2]
  recovered order equals planted order: True
  order_recovery=1.000  kendall_tau=1.000  transitivity=1.000
  top subject elo=1653.5357
note: qualities were planted by this script; the numbers measure the rating mechanism, not any real model.
```

**诚实边界**：质量分是本脚本植入的、不是学出来的；恢复率衡量的是 Elo 更新规则在合成
输入上的性质，**不**代表任何真实语言模型给假设排序的能力，也不是基准成绩。
