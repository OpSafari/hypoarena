"""Golden renderings: any drift in report layout must be a deliberate change.

The pinned digests cover the whole rendered document, so a reordered section, a
renamed header or a changed cell format all fail here on purpose. A few exact
fragments are asserted alongside the digests to keep the failure readable.
"""

from __future__ import annotations

from hypoarena.ids import content_hash
from hypoarena.reports import render_html, render_markdown
from report_fixture import sample_report

MARKDOWN_GOLDEN = "fc448fa5fb49551e"
HTML_GOLDEN = "7d1bc6b721ab5c9f"


def test_markdown_rendering_matches_the_golden_digest() -> None:
    assert content_hash(render_markdown(sample_report())) == MARKDOWN_GOLDEN


def test_html_rendering_matches_the_golden_digest() -> None:
    assert content_hash(render_html(sample_report())) == HTML_GOLDEN


def test_markdown_golden_fragments_are_present() -> None:
    text = render_markdown(sample_report())
    assert "| documents | 6 |" in text
    assert "## Cost (tokens counted, not billed)" in text
    assert "| 1 | agt_a | 1560.0 | 3 | 3 | 0 | 0 | 1.0 |" in text


def test_html_golden_fragments_are_present() -> None:
    text = render_html(sample_report())
    assert "<td>agt_a</td>" in text
    assert "<th>subject</th>" in text
    assert '<ul class="limitations">' in text


def test_golden_renderings_are_stable_across_calls() -> None:
    for _ in range(3):
        assert content_hash(render_markdown(sample_report())) == MARKDOWN_GOLDEN
        assert content_hash(render_html(sample_report())) == HTML_GOLDEN
