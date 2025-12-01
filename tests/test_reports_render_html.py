"""The HTML document assembler: self-containment, escaping and honesty block."""

from __future__ import annotations

from hypoarena.reports import render_html
from report_fixture import sample_report


def test_html_is_a_self_contained_document() -> None:
    text = render_html(sample_report())
    assert text.startswith("<!doctype html>")
    assert text.rstrip().endswith("</html>")
    assert "<style>" in text
    assert "<script" not in text
    assert "http://" not in text and "https://" not in text


def test_html_contains_every_section_heading() -> None:
    text = render_html(sample_report())
    for heading in (
        "<h2>Counts</h2>",
        "<h2>Grounding</h2>",
        "<h2>Ranking</h2>",
        "<h2>Dedup</h2>",
        "<h2>Beliefs</h2>",
        "<h2>Limitations</h2>",
    ):
        assert heading in text


def test_html_escapes_hostile_statement_text() -> None:
    report = sample_report(
        recovered={
            "planted": 1,
            "recovered": 1,
            "rate": 1.0,
            "links": [{"statement": '<script>alert("x")</script>', "recovered": True}],
        }
    )
    text = render_html(report)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&quot;" in text


def test_html_always_emits_the_limitations_block() -> None:
    text = render_html(sample_report(limitations=[]))
    assert '<ul class="limitations">' in text


def test_html_omits_dedup_when_absent() -> None:
    assert "<h2>Dedup</h2>" not in render_html(sample_report(dedup=None))


def test_html_is_deterministic() -> None:
    assert render_html(sample_report()) == render_html(sample_report())
