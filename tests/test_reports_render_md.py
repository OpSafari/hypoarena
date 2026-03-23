"""The Markdown document assembler: ordering, omission and honesty block."""

from __future__ import annotations

from hypoarena.reports import render_markdown
from report_fixture import sample_report


def test_markdown_has_a_title_and_fingerprint_header() -> None:
    text = render_markdown(sample_report())
    assert text.startswith("# Run report: demo\n")
    assert "- config fingerprint: `abc123`" in text
    assert "- corpus hash: `0123456789abcdef`" in text


def test_markdown_contains_every_expected_section_in_order() -> None:
    text = render_markdown(sample_report())
    order = [
        "## Counts",
        "## Grounding",
        "## Ranking",
        "## Dedup",
        "## Beliefs",
        "## Evolution",
        "## Recovery",
        "## Cost",
        "## Recovered planted links",
        "## Limitations",
    ]
    positions = [text.index(heading) for heading in order]
    assert positions == sorted(positions)


def test_markdown_lists_recovered_links_with_flags() -> None:
    text = render_markdown(sample_report())
    assert "- [recovered] A causes B" in text
    assert "- [recovered] B causes C" in text


def test_markdown_marks_a_missing_link() -> None:
    report = sample_report(
        recovered={
            "planted": 1,
            "recovered": 0,
            "rate": 0.0,
            "links": [{"statement": "X causes Y", "recovered": False}],
        }
    )
    assert "- [missing] X causes Y" in render_markdown(report)


def test_markdown_always_emits_the_limitations_section() -> None:
    assert "## Limitations" in render_markdown(sample_report(limitations=[]))


def test_markdown_omits_dedup_when_the_run_had_none() -> None:
    assert "## Dedup" not in render_markdown(sample_report(dedup=None))
    assert "## Dedup" in render_markdown(sample_report())


def test_markdown_is_deterministic() -> None:
    assert render_markdown(sample_report()) == render_markdown(sample_report())
