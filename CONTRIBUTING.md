# 贡献指南

欢迎提交 issue 与 PR。本项目关注**机制层**的可复现实现：图、grounding 校验、
辩论循环、锦标赛评分、去重、演化算子与证据累积，而不是任何具体模型的表现。

## 开发环境

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

`torch` 只用于可选的可训练 ranker 演示；未安装时相关测试会自动跳过。
`requirements.lock.txt` 记录了验证过的固定版本组合，改动依赖后请同步更新。

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `make build` | 构建 wheel（`--no-isolation`，使用本地 venv） |
| `make test` | 快速测试，排除 `slow` 标记 |
| `make test-all` | 全量测试，含 `slow` 与 `model` |
| `make format` | ruff 格式化 + 自动修复 |
| `make format-check` / `make lint` | 格式与规则检查 |
| `make typecheck` | mypy（不固定 `python_version`，跟随当前解释器） |
| `make release` | 清理、构建 sdist+wheel，并在临时 venv 中冒烟安装 |

## 提交规范

- 使用 Conventional Commits：`feat:`、`fix:`、`test:`、`docs:`、`ci:`、`build:`、`chore:`、`refactor:`。
- 一个提交只做一件事，主题行简短；不使用空提交。
- 版本号只做 patch 递增（0.1.0 → 0.1.1），且 `pyproject.toml`、`src/hypoarena/_version.py`、
  `CHANGELOG.md` 必须在同一个提交内一起更新。

## 测试策略

- 默认测试必须**完全离线**：不联网、不调用真实模型、不下载权重或数据集。
- 量化实验使用合成语料，其中人工植入（planted）因果链、竞争假设与改写簇，
  度量的是“机制能否恢复植入结构”，而不是模型能力。
- 需要 `torch` 的用例标记 `@pytest.mark.model`；耗时训练或集成用例标记 `@pytest.mark.slow`。
  两者都被 `make test-all` 与 CI 覆盖。
- 涉及公开 API、配置 schema 或序列化格式的改动，请同步更新对应的 golden 测试，
  让格式漂移成为显式决定。
- 测试应覆盖边界与两个方向（接受与拒绝），不要复制实现逻辑。

## PR 流程

1. 从 `main` 拉分支，保持改动聚焦。
2. 本地跑通 `make test-all`、`make lint`、`make typecheck`。
3. 填写 PR 模板中的检查项；涉及示例时说明实际运行输出。
