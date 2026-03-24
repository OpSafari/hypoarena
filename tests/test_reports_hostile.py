"""Adversarial inputs must never break the table structure or inject markup.

The report renderers sit at the boundary between untrusted statement text and a
document a human opens, so these checks are one-directional on purpose: they
assert the dangerous form is *absent* and the escaped form is *present*.
"""

from __future__ import annotations

import pytest

from hypoarena.reports import render_html, render_markdown
from report_fixture import sample_report

HOSTILE = '<script>alert("pwn")</script>'
PIPEY = "gene A | gene B"
UNICODE = "café 文 🎉"


def hostile_report(statement: str) -> dict[str, object]:
    return sample_report(
        recovered={
            "planted": 1,
            "recovered": 1,
            "rate": 1.0,
            "links": [{"statement": statement, "recovered": True}],
        },
        limitations=[statement],
    )


def test_html_never_emits_a_raw_script_tag() -> None:
    text = render_html(hostile_report(HOSTILE))
    assert "<script" not in text
    assert "&lt;script&gt;" in text
    assert "&quot;" in text


def test_html_escapes_quotes_in_both_link_and_limitation() -> None:
    text = render_html(hostile_report(HOSTILE))
    # the raw double-quote from the payload cannot survive into the document
    assert 'alert("pwn")' not in text


def test_markdown_pipe_cannot_split_a_cell() -> None:
    text = render_markdown(hostile_report(PIPEY))
    link_line = next(line for line in text.splitlines() if "gene A" in line)
    assert "|" not in link_line.replace("\\|", "")  # only escaped pipes remain
    assert "gene A \\| gene B" in link_line


def test_markdown_newline_in_statement_does_not_add_a_row() -> None:
    text = render_markdown(hostile_report("line one\n- fake bullet"))
    # the injected newline is collapsed, so it never becomes a standalone,
    # attacker-authored list item; the statement stays inline on one line
    assert not any(line.strip() == "- fake bullet" for line in text.splitlines())
    assert "- [recovered] line one - fake bullet" in text


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_unicode_survives_both_renderers(render) -> None:
    text = render(hostile_report(UNICODE))
    assert "café 文 🎉" in text


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_renderers_tolerate_a_minimal_report(render) -> None:
    # no sections at all: still a valid document, still honest
    text = render({"run_id": "empty"})
    assert "Limitations" in text


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_rendering_is_pure_with_respect_to_the_input(render) -> None:
    report = hostile_report(HOSTILE)
    before = repr(report)
    render(report)
    assert repr(report) == before  # renderer must not mutate its input
