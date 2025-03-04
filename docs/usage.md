# 使用指南

## 安装

需要 Python 3.11+。运行时依赖只有 NumPy。

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

可选：安装 CPU 版 PyTorch 以启用可训练 ranker 演示。

```bash
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

## 命令行

```bash
.venv/bin/hypoarena --version
.venv/bin/hypoarena --help
python -m hypoarena --help
```

子命令（`corpus`、`generate`、`verify`、`dedup`、`debate`、`rank`、`evolve`、
`accumulate`、`report`、`demo`）随对应机制落地逐步接入，`--help` 始终反映当前可用集合。

## 产物约定

- 每个阶段的输出都是 JSONL：一行一个 JSON 对象，UTF-8，键按字典序排列，便于 diff。
- 运行元数据（版本、配置哈希、seed）与阶段产物写在同一 run 目录下。
- 产物中不写入密钥；序列化前会做敏感字段过滤。

## 离线保证

默认命令与示例不联网。唯一会发出 HTTP 请求的是显式配置 base URL 的适配器，
其测试只连接本地 loopback mock server。
