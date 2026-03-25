"""Human-readable run reports rendered to Markdown and self-contained HTML.

Rendering is pure and offline: the same report mapping always produces the same
bytes, hostile text is escaped, and every document embeds the run's limitations
so a rendered report can never be mistaken for a claim about scientific truth.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from hypoarena.artifacts import ArtifactStore

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


# Sections appear in this fixed order in both renderers. A section whose
# extractor returns None (e.g. dedup when the run skipped it) is omitted rather
# than shown empty, so the document reflects what the run actually did.
_TABLES = (
    counts_table,
    grounding_table,
    ranking_table,
    dedup_table,
    beliefs_table,
    evolution_table,
    recovery_table,
    cost_table,
)


def render_markdown(report: Mapping[str, object]) -> str:
    """Render the whole report as a GitHub-flavoured Markdown document.

    The limitations section is always emitted, even when empty, so a rendered
    report can never silently drop its honesty block.
    """
    lines = [f"# Run report: {escape_markdown_cell(report.get('run_id', ''))}", ""]
    lines.append(f"- config fingerprint: `{report.get('config_fingerprint', '')}`")
    lines.append(f"- corpus hash: `{report.get('corpus_hash', '')}`")
    lines.append("")
    for build in _TABLES:
        table = build(report)
        if table is None:
            continue
        title, headers, rows = table
        lines.append(f"## {title}")
        lines.append("")
        lines.append(markdown_table(headers, rows))
        lines.append("")
    lines.append("## Recovered planted links")
    lines.append("")
    for statement, found in recovered_links(report):
        mark = "recovered" if found else "missing"
        lines.append(f"- [{mark}] {escape_markdown_cell(statement)}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in _items(report, "limitations"):
        lines.append(f"- {escape_markdown_cell(item)}")
    lines.append("")
    return "\n".join(lines)


# Inline CSS only: no external stylesheet, script, font or image, so a saved
# report opens offline and renders identically anywhere.
_HTML_STYLE = (
    "body{font-family:system-ui,-apple-system,sans-serif;margin:2rem auto;"
    "max-width:60rem;line-height:1.5;color:#111;padding:0 1rem}"
    "h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:1.6rem}"
    "table{border-collapse:collapse;margin:0.5rem 0}"
    "th,td{border:1px solid #ccc;padding:0.25rem 0.6rem;text-align:left}"
    "th{background:#f5f5f5}code{background:#f5f5f5;padding:0 0.2rem}"
    "ul.limitations{background:#fff8e1;border:1px solid #e0c060;"
    "padding:0.6rem 1.2rem;border-radius:4px}"
)


def render_html(report: Mapping[str, object]) -> str:
    """Render the whole report as a self-contained HTML document.

    Every dynamic value passes through :func:`escape_html`, so hostile statement
    text can never open a tag or inject a script. The limitations block is always
    present and visually highlighted.
    """
    run_id = escape_html(report.get("run_id", ""))
    fingerprint = escape_html(report.get("config_fingerprint", ""))
    corpus_hash = escape_html(report.get("corpus_hash", ""))
    out = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Run report {run_id}</title>",
        f"<style>{_HTML_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>Run report: {run_id}</h1>",
        "<ul>",
        f"<li>config fingerprint: <code>{fingerprint}</code></li>",
        f"<li>corpus hash: <code>{corpus_hash}</code></li>",
        "</ul>",
    ]
    for build in _TABLES:
        table = build(report)
        if table is None:
            continue
        title, headers, rows = table
        out.append(f"<h2>{escape_html(title)}</h2>")
        out.append(html_table(headers, rows))
    out.append("<h2>Recovered planted links</h2>")
    out.append("<ul>")
    for statement, found in recovered_links(report):
        mark = "recovered" if found else "missing"
        out.append(f"<li>[{mark}] {escape_html(statement)}</li>")
    out.append("</ul>")
    out.append("<h2>Limitations</h2>")
    out.append('<ul class="limitations">')
    for item in _items(report, "limitations"):
        out.append(f"<li>{escape_html(item)}</li>")
    out.append("</ul>")
    out.append("</body>")
    out.append("</html>")
    return "\n".join(out)


MARKDOWN_ARTIFACT = "report.md"
HTML_ARTIFACT = "report.html"


def write_reports(
    store: ArtifactStore, report: Mapping[str, object]
) -> tuple[str, str]:
    """Render ``report`` and atomically write report.md and report.html.

    Returns the two artifact names so the caller can record them alongside the
    JSON payload. Both files are pure functions of ``report``, so a resumed run
    rewrites byte-identical documents.
    """
    store.write_text(MARKDOWN_ARTIFACT, render_markdown(report))
    store.write_text(HTML_ARTIFACT, render_html(report))
    return (MARKDOWN_ARTIFACT, HTML_ARTIFACT)
