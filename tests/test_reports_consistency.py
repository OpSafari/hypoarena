"""The Markdown and HTML renderers must present the same data.

Both draw on the shared table extractors, so a section, row count or cell value
that appears in one format must appear in the other. This guards against the two
renderers drifting apart after a change to one of them.
"""

from __future__ import annotations

from hypoarena.reports import (
    html_table,
    markdown_table,
    render_html,
    render_markdown,
)
from report_fixture import sample_report

SECTIONS = (
    "Counts",
    "Grounding",
    "Ranking",
    "Dedup",
    "Beliefs",
    "Evolution",
    "Recovery",
    "Cost",
)


def test_both_renderers_emit_the_same_sections() -> None:
    markdown = render_markdown(sample_report())
    html = render_html(sample_report())
    for name in SECTIONS:
        # match the heading prefix: the Cost section carries a longer title
        assert f"## {name}" in markdown
        assert f"<h2>{name}" in html


def test_table_row_counts_agree_between_renderers() -> None:
    headers = ["a", "b"]
    rows = [["1", "2"], ["3", "4"], ["5", "6"]]
    markdown_rows = len(markdown_table(headers, rows).splitlines()) - 2
    html_rows = html_table(headers, rows).count("<tr>") - 1
    assert markdown_rows == html_rows == len(rows)


def test_both_renderers_carry_the_same_cell_values() -> None:
    report = sample_report()
    markdown = render_markdown(report)
    html = render_html(report)
    for value in ("documents", "agt_a", "clm_a", "grounded_rate", "1560.0"):
        assert value in markdown
        assert value in html


def test_both_renderers_omit_dedup_together() -> None:
    report = sample_report(dedup=None)
    assert "## Dedup" not in render_markdown(report)
    assert "<h2>Dedup</h2>" not in render_html(report)
