# Changelog

本项目遵循语义化版本；0.x 阶段的破坏性变更会在 minor 版本说明中显式列出。

## 0.1.1

- 新增 `hypoarena.errors`：带稳定 `code` 与 `exit_code` 的异常层级，以及 `ERROR_CODES` 注册表。
- 新增 `hypoarena.ids`：规范 JSON、SHA-256 内容哈希、多段哈希与 ID 构造/校验。
- 新增 `hypoarena.text`：归一化链、分词、停用词、字符/词 n-gram、数字抽取、句子切分与否定线索。
- 新增 `hypoarena.codec`：严格解码（未知键、类型、范围、schema 版本）与规范 JSONL 行编解码。
- 新增 `hypoarena.schema`：`Scope`、`Citation`、`Provenance`、`Claim`、`Evidence` 与关系/极性枚举。
- 新增 `hypoarena.serialize`：记录与 dict/JSONL 的编解码层。
- 文档：架构总览、使用指南、API 约定、schema 参考、诚实性说明、参考文献、贡献与安全策略。
- CI：3.11/3.12/3.13 矩阵（build / test-all / format-check / lint / typecheck）、wheel 冒烟任务与发布工作流。
- 测试：golden 摘要、golden 配置、golden 公开 API、golden 文本归一化、golden schema JSONL 与标记卫生检查。

## 0.1.0

- 初始骨架：src 布局打包、Makefile 任务（build / test / test-all / format / lint / typecheck / release）、
  pytest 标记（`slow`、`model`）、ruff 与 mypy 配置。
- `hypoarena.__version__` 与 `hypoarena._version` 暴露版本号。
