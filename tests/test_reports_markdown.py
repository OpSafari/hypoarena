"""Markdown table primitives: escaping and rendering at the boundaries."""

from __future__ import annotations

import pytest

from hypoarena.reports import escape_markdown_cell, markdown_table


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (42, "42"),
        (3.5, "3.5"),
        ("plain", "plain"),
        ("a|b", "a\\|b"),
        ("`code`", "\\`code\\`"),
        ("<script>", "\\<script\\>"),
        ("back\\slash", "back\\\\slash"),
    ],
)
def test_escape_markdown_cell_neutralises_specials(
    value: object, expected: str
) -> None:
    assert escape_markdown_cell(value) == expected


def test_escape_collapses_newlines_and_whitespace_runs() -> None:
    assert escape_markdown_cell("line1\nline2") == "line1 line2"
    assert escape_markdown_cell("  a \t b  ") == "a b"


def test_escape_backslash_before_pipe_does_not_double_the_pipe_escape() -> None:
    # "\|" -> backslash doubled, then pipe escaped: literal backslash + cell-safe pipe
    assert escape_markdown_cell("\\|") == "\\\\\\|"


def test_markdown_table_renders_header_divider_and_rows() -> None:
    table = markdown_table(["a", "b"], [[1, 2], [3, 4]])
    assert table == "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"


def test_markdown_table_with_no_rows_is_header_and_divider_only() -> None:
    assert markdown_table(["x"], []) == "| x |\n| --- |"


def test_markdown_table_escapes_hostile_cells() -> None:
    table = markdown_table(["h"], [["a|b"]])
    assert table.splitlines()[-1] == "| a\\|b |"
