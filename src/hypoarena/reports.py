"""Human-readable run reports rendered to Markdown and self-contained HTML.

Rendering is pure and offline: the same report mapping always produces the same
bytes, hostile text is escaped, and every document embeds the run's limitations
so a rendered report can never be mistaken for a claim about scientific truth.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

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


# A rendered section: a title, table headers and the row values. Keeping the
# extraction separate from the renderers means Markdown and HTML share one
# source of rows and can never drift apart.
Table = tuple[str, list[str], list[list[object]]]

COUNT_FIELDS = ("documents", "claims", "evidence", "links", "edges")
GROUNDING_FIELDS = (
    "total",
    "grounded",
    "weakly_grounded",
    "ungrounded",
    "fabricated",
    "grounded_rate",
    "mean_score",
)
RANKING_HEADERS = (
    "#",
    "subject",
    "elo",
    "played",
    "wins",
    "losses",
    "draws",
    "win_rate",
)
RANKING_FIELDS = (
    "position",
    "subject",
    "elo",
    "played",
    "wins",
    "losses",
    "draws",
    "win_rate",
)
DEDUP_FIELDS = (
    "total",
    "clusters",
    "duplicated",
    "duplicate_rate",
    "method",
    "threshold",
)


def _sub(report: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return ``report[key]`` when it is a mapping, else an empty mapping."""
    value = report.get(key)
    return value if isinstance(value, Mapping) else {}


def _items(report: Mapping[str, object], key: str) -> list[object]:
    """Return ``report[key]`` as a list when it is a sequence, else empty."""
    value = report.get(key)
    return list(value) if isinstance(value, (list, tuple)) else []


def counts_table(report: Mapping[str, object]) -> Table:
    """Extract the artifact-count section."""
    counts = _sub(report, "counts")
    rows = [[name, counts.get(name, 0)] for name in COUNT_FIELDS]
    return ("Counts", ["artifact", "count"], rows)


def grounding_table(report: Mapping[str, object]) -> Table:
    """Extract the grounding-flag summary section."""
    grounding = _sub(report, "grounding")
    rows = [[name, grounding.get(name, 0)] for name in GROUNDING_FIELDS]
    return ("Grounding", ["metric", "value"], rows)


def ranking_table(report: Mapping[str, object]) -> Table:
    """Extract the tournament standings section."""
    rows: list[list[object]] = []
    for entry in _items(report, "ranking"):
        if isinstance(entry, Mapping):
            rows.append([entry.get(field) for field in RANKING_FIELDS])
    return ("Ranking", list(RANKING_HEADERS), rows)


def dedup_table(report: Mapping[str, object]) -> Table | None:
    """Extract the dedup section, or ``None`` when the run performed no dedup."""
    value = report.get("dedup")
    if not isinstance(value, Mapping):
        return None
    rows = [[name, value.get(name, "")] for name in DEDUP_FIELDS]
    return ("Dedup", ["metric", "value"], rows)


def beliefs_table(report: Mapping[str, object]) -> Table:
    """Extract the top-beliefs section."""
    rows: list[list[object]] = []
    for entry in _items(report, "beliefs"):
        if isinstance(entry, Mapping):
            rows.append([entry.get("claim_id"), entry.get("posterior")])
    return ("Beliefs", ["claim_id", "posterior"], rows)


def evolution_table(report: Mapping[str, object]) -> Table:
    """Extract the evolution-progress section."""
    evolution = _sub(report, "evolution")
    rows = [
        [name, evolution.get(name, 0)]
        for name in ("generations", "accepted", "rejected")
    ]
    return ("Evolution", ["metric", "value"], rows)


def recovery_table(report: Mapping[str, object]) -> Table:
    """Extract the planted-link recovery summary section."""
    recovered = _sub(report, "recovered")
    rows = [[name, recovered.get(name, 0)] for name in ("planted", "recovered", "rate")]
    return ("Recovery", ["metric", "value"], rows)


def recovered_links(report: Mapping[str, object]) -> list[tuple[str, bool]]:
    """Return each planted link as ``(statement, was_recovered)``.

    This is the per-link detail behind :func:`recovery_table`; the renderer uses
    it to list exactly which planted links a run did and did not recover.
    """
    links = _sub(report, "recovered").get("links")
    out: list[tuple[str, bool]] = []
    if isinstance(links, (list, tuple)):
        for link in links:
            if isinstance(link, Mapping):
                out.append(
                    (str(link.get("statement", "")), bool(link.get("recovered")))
                )
    return out


def cost_table(report: Mapping[str, object]) -> Table:
    """Extract the token-accounting section (counts only, never priced)."""
    cost = _sub(report, "cost")
    fields = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    rows = [[name, cost.get(name, 0)] for name in fields]
    return ("Cost (tokens counted, not billed)", ["metric", "value"], rows)
