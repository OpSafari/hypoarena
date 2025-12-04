# 报告渲染

`hypoarena.reports` 把 `run_report` 产出的报告载荷渲染成两种文档：**Markdown**
（`render_markdown`）与**自包含 HTML**（`render_html`）。渲染是纯函数：同一份
载荷永远得到逐字节相同的输出，因此断点续跑重写出的报告与一次跑完完全一致。

## 两个方向的安全保证

报告里的假设陈述来自 agent，属于**不可信文本**。渲染器对它做转义，保证一段
恶意 statement 既不能破坏表格结构，也不能注入标记：

- **Markdown**：`escape_markdown_cell` 折叠换行（单元格不能跨行），并反斜杠转义
  `\ | \` < >`，因此 `gene A | gene B` 不会把一行拆成两列，`<script>` 也不会被
  当作内联 HTML。这是防御性子集，不是完整的 CommonMark 转义器。
- **HTML**：`escape_html` 使用 `html.escape(..., quote=True)`，`< > & " '` 全部转义。
  文档只用**内联 CSS**，没有任何外部样式表、脚本、字体或图片，离线打开即可，
  渲染结果在任何环境一致。

`test_reports_hostile.py` 对这些保证做单向断言：危险形式必须**缺席**，转义形式
必须**在场**。

## 段落与顺序

两种渲染共享同一组表格抽取器（`counts_table`、`grounding_table`、`ranking_table`、
`dedup_table`、`beliefs_table`、`evolution_table`、`recovery_table`、`cost_table`），
按固定顺序输出，因此 Markdown 与 HTML 不会各说各话。抽取器返回 `None` 的段落
（例如未做去重时的 dedup）会被**省略**而非留空，让文档如实反映这次运行做了什么。

## 诚实性块

`## Limitations` 段落**永远输出**，即使为空也会保留标题；HTML 里它带高亮样式。
限制说明直接来自 runner 的 `REPORT_LIMITATIONS`：语料全部合成、排名只反映评审
rubric、grounding 是词面启发式、信念是约定似然下的记账、token 只计数不计费。
任何渲染出的报告都携带这段说明，不可能被误读成对科学真伪的判定。

## 落盘

`write_reports(store, report)` 原子写入 `report.md` 与 `report.html`，返回两个产物
名。runner 的 report 阶段在写 `report.json` 之后调用它，因此一次运行同时留下
JSON（机器可读）与两种渲染文档（人类可读）。文本产物经由 `ArtifactStore.write_text`
写入，与 JSONL 产物同样走临时文件 + `fsync` + 原子替换。

## golden

`test_golden_reports.py` 用固定载荷钉住整篇渲染的 `content_hash`，任何版面改动
（段落重排、表头改名、单元格格式变化）都会让 golden 失败——漂移必须是有意的。
