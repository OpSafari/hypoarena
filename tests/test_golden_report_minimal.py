"""Exact golden for the smallest report rendering.

A minimal payload (a run id and one limitation) exercises every section with
zero or empty data. Pinning both digests plus a few exact lines keeps the
empty-report layout honest, readable and free of accidental drift.
"""

from __future__ import annotations

from hypoarena.ids import content_hash
from hypoarena.reports import render_html, render_markdown

MINIMAL = {"run_id": "tiny", "limitations": ["synthetic only"]}
MARKDOWN_DIGEST = "59d429d05db84839"
HTML_DIGEST = "f3b453220a532cbe"


def test_minimal_markdown_matches_the_golden_digest() -> None:
    assert content_hash(render_markdown(MINIMAL)) == MARKDOWN_DIGEST


def test_minimal_html_matches_the_golden_digest() -> None:
    assert content_hash(render_html(MINIMAL)) == HTML_DIGEST


def test_minimal_markdown_keeps_zero_rows_and_the_limitation() -> None:
    text = render_markdown(MINIMAL)
    assert text.startswith("# Run report: tiny\n")
    assert "| documents | 0 |" in text
    assert text.rstrip().endswith("- synthetic only")


def test_minimal_html_is_self_contained() -> None:
    html = render_html(MINIMAL)
    assert html.startswith("<!doctype html>")
    assert "<script" not in html
    assert "synthetic only" in html
