"""Human-readable run reports rendered to Markdown and self-contained HTML.

Rendering is pure and offline: the same report mapping always produces the same
bytes, hostile text is escaped, and every document embeds the run's limitations
so a rendered report can never be mistaken for a claim about scientific truth.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

# A defensive subset of characters that could break a Markdown table cell or
# smuggle in inline HTML. This is not a full CommonMark escaper; the contract is
# only that untrusted statement text can never split a cell or open a raw tag.
# Backslash must be replaced first so later escapes are not double-escaped.
_MARKDOWN_ESCAPES = (
    ("\\", "\\\\"),
    ("|", "\\|"),
    ("`", "\\`"),
    ("<", "\\<"),
    (">", "\\>"),
)


def escape_markdown_cell(text: object) -> str:
    """Escape one value so it is safe inside a Markdown table cell.

    All whitespace runs (including newlines) collapse to a single space because
    a table cell cannot span lines, then the characters in
    :data:`_MARKDOWN_ESCAPES` are backslash-escaped.
    """
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    for needle, replacement in _MARKDOWN_ESCAPES:
        value = value.replace(needle, replacement)
    return value


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table with escaped cells.

    An empty ``rows`` yields just the header and divider, which is still a valid
    (empty) table rather than an error: a report section with no data must say
    so explicitly instead of vanishing.
    """
    head = "| " + " | ".join(escape_markdown_cell(h) for h in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [head, divider]
    for row in rows:
        lines.append(
            "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |"
        )
    return "\n".join(lines)


def escape_html(text: object) -> str:
    """Escape one value for safe inclusion in HTML text or a quoted attribute.

    Uses :func:`html.escape` with ``quote=True`` so both double and single
    quotes are neutralised; this is the guarantee that keeps untrusted
    statement text from closing a tag or injecting a script.
    """
    return html.escape("" if text is None else str(text), quote=True)


def html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a self-contained HTML table with every cell escaped.

    The result carries no external references (no scripts, stylesheets or
    images), so a saved report opens offline exactly as rendered here.
    """
    head = "".join(f"<th>{escape_html(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape_html(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"
    )
