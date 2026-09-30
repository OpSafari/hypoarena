# 命令行接口

安装后提供 `hypoarena` 命令（等价于 `python -m hypoarena`）。每个子命令都在
**合成语料**上驱动同一条离线流水线（见 [runner.md](runner.md)），跑到对应阶段为止，
再打印一小段人类可读的摘要。全程不联网、不需要 torch、不产生任何费用。

## 子命令

子命令按流水线顺序排列，每个都执行到该阶段为止的前缀：

| 子命令 | 作用 | 关键产物 |
| --- | --- | --- |
| `corpus` | 生成带植入真值的合成语料 | `corpus.jsonl`、`truth.jsonl`、`graph.jsonl` |
| `generate` | 由 agent 提出候选 claim | `candidates.jsonl` |
| `verify` | 用 grounding 校验器给每条 claim 打分 | `grounding.jsonl` |
| `dedup` | 找出近重复 claim 簇 | `dedup.jsonl` |
| `debate` | 运行 propose→critique→revise 辩论 | `debates.jsonl` |
| `rank` | 用 Elo 锦标赛排序，打印前几名 | `tournament.jsonl` |
| `evolve` | 用演化算子扩充假设图 | `evolution.jsonl` |
| `accumulate` | 由分级证据更新信念 | `beliefs.jsonl` |
| `report` | 跑完整条流水线并写出报告 | `report.json`、`report.md`、`report.html` |
| `demo` | 小规模端到端演示，打印"恢复 vs 植入" | 同上 |

`demo` 是最省事的一键离线演示：它在一个小的植入语料上跑完整条流水线，打印每条
植入因果链是否被恢复，并附带一句诚实声明——这只是合成演示，不构成对真实科学
发现能力的任何主张。

## 公共参数

每个子命令都接受相同的运行配置参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--seed` | 270106 | 合成运行的随机种子（决定语料与流水线） |
| `--out` | `runs` | 产物根目录 |
| `--run-id` | `run` | 运行标识，必须是单个路径段 |
| `--chains` | 3 | 植入的因果链数量 |
| `--chain-length` | 3 | 每条链的变量数（links = length − 1） |
| `--resume` | 关（flag） | 配置兼容时复用已有检查点，跳过已完成的阶段 |

示例：

```console
$ hypoarena demo --seed 7 --chains 2 --chain-length 2 --out /tmp/hya
$ hypoarena rank --out /tmp/hya --run-id ranked
```

## 退出码

失败会以 `hypoarena.errors.HypoArenaError` 抛出，`main` 把它的 `exit_code`
作为进程退出码，并在 stderr 打印一行说明，便于脚本按类别分支而无需解析文本：

| 退出码 | 类别 |
| --- | --- |
| 0 | 成功 |
| 1 | 其它未分类错误 |
| 2 | 校验错误（schema / 配置 / 重复 ID / 非法引用） |
| 3 | 语料错误（如 span 未找到） |
| 4 | 适配器错误（replay 耗尽 / 传输失败） |
| 5 | 配置错误 |
| 6 | 产物错误（如密钥泄漏拦截） |

argparse 层面的用法错误（未知子命令、未知 flag）由 argparse 直接以退出码 2 结束。
