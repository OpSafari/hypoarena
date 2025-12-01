"""Table extractors turn a report mapping into (title, headers, rows)."""

from __future__ import annotations

from hypoarena.reports import (
    counts_table,
    dedup_table,
    grounding_table,
    ranking_table,
)
from report_fixture import sample_report


def test_counts_table_reads_every_field_in_order() -> None:
    title, headers, rows = counts_table(sample_report())
    assert title == "Counts"
    assert headers == ["artifact", "count"]
    assert rows[0] == ["documents", 6]
    assert [row[0] for row in rows] == [
        "documents",
        "claims",
        "evidence",
        "links",
        "edges",
    ]


def test_counts_table_defaults_missing_values_to_zero() -> None:
    _, _, rows = counts_table({})
    assert rows == [
        [name, 0] for name in ("documents", "claims", "evidence", "links", "edges")
    ]


def test_counts_table_ignores_a_non_mapping_counts_field() -> None:
    _, _, rows = counts_table({"counts": "not-a-mapping"})
    assert rows[0] == ["documents", 0]


def test_grounding_table_exposes_rates() -> None:
    title, _, rows = grounding_table(sample_report())
    assert title == "Grounding"
    as_map = {name: value for name, value in rows}
    assert as_map["grounded_rate"] == 0.75
    assert as_map["fabricated"] == 0


def test_ranking_table_preserves_standing_order_and_fields() -> None:
    title, headers, rows = ranking_table(sample_report())
    assert title == "Ranking"
    assert headers[0] == "#" and headers[1] == "subject"
    assert rows[0][1] == "agt_a"
    assert rows[0][2] == 1560.0
    assert len(rows) == 2


def test_ranking_table_skips_non_mapping_entries() -> None:
    _, _, rows = ranking_table({"ranking": ["junk", {"subject": "x"}]})
    assert len(rows) == 1
    assert rows[0][1] == "x"


def test_dedup_table_is_none_when_absent_or_null() -> None:
    assert dedup_table({}) is None
    assert dedup_table({"dedup": None}) is None


def test_dedup_table_reads_summary_fields() -> None:
    title, _, rows = dedup_table(sample_report())
    assert title == "Dedup"
    as_map = {name: value for name, value in rows}
    assert as_map["method"] == "minhash"
    assert as_map["duplicate_rate"] == 0.5
