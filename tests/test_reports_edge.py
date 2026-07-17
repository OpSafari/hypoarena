"""Reports must render safely for degenerate and partial payloads.

A report assembled from a run that skipped stages, produced no rankings or
carries null sections must still render - and still carry its honesty block.
These cover the boundaries the golden fixture does not.
"""

from __future__ import annotations

from hypoarena.reports import render_html, render_markdown


def test_a_minimal_payload_renders_both_formats() -> None:
    markdown = render_markdown({"run_id": "m"})
    html = render_html({"run_id": "m"})
    assert markdown.startswith("# Run report: m")
    assert "<h1>Run report: m</h1>" in html
    assert "## Limitations" in markdown
    assert "Limitations" in html


def test_empty_sections_render_their_headings_without_rows() -> None:
    report = {
        "run_id": "e",
        "counts": {},
        "grounding": {},
        "ranking": [],
        "beliefs": [],
        "evolution": {},
        "recovered": {"planted": 0, "recovered": 0, "rate": 0.0, "links": []},
        "cost": {},
        "limitations": [],
    }
    markdown = render_markdown(report)
    assert "## Ranking" in markdown
    assert "## Recovered planted links" in markdown
    assert render_html(report).count("<tr>") >= 1


def test_none_and_missing_fields_do_not_raise() -> None:
    for report in (
        {"run_id": None},
        {"run_id": "x", "counts": None, "ranking": None, "limitations": None},
        {"run_id": "x", "dedup": None, "recovered": None, "cost": None},
        {},
    ):
        assert isinstance(render_markdown(report), str)
        assert isinstance(render_html(report), str)


def test_dedup_section_only_appears_when_present() -> None:
    assert "## Dedup" not in render_markdown({"run_id": "x"})
    assert "## Dedup" in render_markdown({"run_id": "x", "dedup": {"total": 0}})


def test_a_non_string_run_id_is_rendered_not_crashed() -> None:
    assert "123" in render_markdown({"run_id": 123})
    assert "123" in render_html({"run_id": 123})
