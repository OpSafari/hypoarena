# 测试策略

## 运行

| 命令 | 作用 |
| --- | --- |
| `make test` | 快速套件，排除 `slow` 标记 |
| `make test-all` | 全量套件，含 `slow` 与 `model` |
| `make format-check` / `make lint` | ruff 格式与规则检查 |
| `make typecheck` | mypy（不锁定 `python_version`） |

CI 在 ubuntu 的 Python 3.11 / 3.12 / 3.13 矩阵上运行 build、test-all、format-check、
lint 与 typecheck，并对构建出的 wheel 做一次安装冒烟测试。

## 标记

- `slow` — 训练或较重的集成检查，从 `make test` 排除以保持逐提交快速；仍纳入 `test-all`。
- `model` — 需要可选 `torch` extra；未安装时用 `pytest.importorskip` 干净跳过，安装后
  纳入 `test-all`。torch 相关测试只用 CPU、固定 seed，不下载任何权重。

## 三类保证

1. **golden / 穷举**：公开 API 表面、schema、配置指纹与序列化输出用 `content_hash` 或
   精确文本钉住（`tests/test_golden_*.py`），任何漂移都必须是有意的改动。
2. **性质 / 扫描**：带 seed 的随机图与语料扫描结构不变量——图有效性、信念单调性、
   Elo 更新有界、去重指标可复现、续跑与一次跑完逐字节一致。
3. **对抗 / 边界**：grounding 校验对编造引用、位移 span、漂移数字与否定翻转全部拦截；
   报告渲染对 hostile text 转义；schema 校验拒绝越界字段与重复 ID。测试覆盖被触碰操作的
   两个方向（如加入/移除、支持/反驳、恢复/缺失）。

## 离线与确定性

默认测试与示例全部离线、确定性：随机性来自显式 seed，排序用稳定键，agent 是脚本化 /
回放式实现，HTTP 适配器只连本地 loopback mock server。因此"一次跑完"与"分段续跑"
逐字节一致，可由测试直接断言，文档里的每个数字都由仓库内脚本真实跑出。
