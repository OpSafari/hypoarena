"""HTML table primitives: escaping and self-contained rendering."""

from __future__ import annotations

import pytest

from hypoarena.reports import escape_html, html_table


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (42, "42"),
        ("plain", "plain"),
        ("<script>", "&lt;script&gt;"),
        ("a & b", "a &amp; b"),
        ('say "hi"', "say &quot;hi&quot;"),
        ("it's", "it&#x27;s"),
    ],
)
def test_escape_html_neutralises_markup(value: object, expected: str) -> None:
    assert escape_html(value) == expected


def test_html_table_renders_header_and_body() -> None:
    table = html_table(["a", "b"], [[1, 2]])
    assert table == (
        "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    )


def test_html_table_with_no_rows_has_an_empty_tbody() -> None:
    assert html_table(["x"], []) == (
        "<table><thead><tr><th>x</th></tr></thead><tbody></tbody></table>"
    )


def test_html_table_escapes_hostile_cells() -> None:
    table = html_table(["h"], [['<img src=x onerror="a">']])
    assert "<img" not in table
    assert "&lt;img" in table
    assert "&quot;" in table
