# 使用指南

## 安装

需要 Python 3.11+。运行时依赖只有 NumPy。

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

可选：安装 CPU 版 PyTorch 以启用可训练 ranker 演示（`torch` extra）。

```bash
.venv/bin/pip install -e ".[dev,torch]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

一条命令跑完整条离线流水线，并打印"恢复 vs 植入"的因果链：

```bash
.venv/bin/hypoarena demo --chains 2 --chain-length 2 --out /tmp/hya
```

它在合成语料上运行 corpus → generate → verify → dedup → debate → rank →
evolve → accumulate → report，并把 `report.json` / `report.md` / `report.html`
写到 `/tmp/hya/run/`。

## 命令行

子命令按流水线顺序排列，每个都执行到对应阶段为止并打印一小段摘要：

| 子命令 | 作用 |
| --- | --- |
| `corpus` | 生成带植入真值的合成语料 |
| `generate` | 由脚本化 agent 提出候选 claim |
| `verify` | 用 grounding 校验器给 claim 打分 |
| `dedup` | 找出近重复 claim 簇 |
| `debate` | 运行 propose→critique→revise 辩论 |
| `rank` | 用 Elo 锦标赛排序并打印前几名 |
| `evolve` | 用演化算子扩充假设图 |
| `accumulate` | 由分级证据更新信念 |
| `report` | 跑完整条流水线并写出报告 |
| `demo` | 小规模端到端演示 |

公共参数：`--seed`、`--out`、`--run-id`、`--chains`、`--chain-length`。
失败按错误类别返回稳定退出码（校验=2、语料=3、适配器=4、配置=5、产物=6）。
完整说明见 [cli.md](cli.md)。

## 阅读报告

`report` 与 `demo` 会写出三种报告：`report.json`（机器可读）、`report.md`
（Markdown）与 `report.html`（自包含 HTML，内联样式、离线可开）。渲染文档对不可信
的假设陈述做转义，并**始终内嵌 limitations 段**，说明语料是合成的、排名只反映评审
rubric、token 只计数不计费。细节见 [reports.md](reports.md)。

## 示例

`examples/` 下有三个完全离线、可运行的示例（植入语料与 grounding 校验、Elo 排名恢复、
演化 + 去重的新颖度度量），每个都附带 README 与真实运行输出。总览见
[examples.md](examples.md)。

## 产物约定

- 每个阶段的输出都是 JSONL：一行一个 JSON 对象，UTF-8，键按字典序排列，便于 diff；
  JSONL 首行是带计数与签名的 `meta` 头。
- 一次运行写在一个 run 目录下：`run.json`（元数据：版本、配置指纹、seed、注入时间戳）、
  各阶段 `*.jsonl`、`summary.json`、`cost.json`、`report.*`，以及 `checkpoints/<stage>.done`
  续跑标记。目录布局与续跑语义见 [runner.md](runner.md)。
- 产物中不写入密钥；序列化前会做敏感字段过滤。

## 离线保证

默认命令与示例不联网、不下载权重、不调用真实模型。唯一会发出 HTTP 请求的是显式配置
base URL 的适配器，其测试只连接本地 loopback mock server，`HttpConfig` 默认拒绝非
loopback 端点。
