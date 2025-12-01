"""The remaining table extractors and the per-link recovery detail."""

from __future__ import annotations

from hypoarena.reports import (
    beliefs_table,
    cost_table,
    evolution_table,
    recovered_links,
    recovery_table,
)
from report_fixture import sample_report


def test_beliefs_table_lists_claim_and_posterior() -> None:
    title, headers, rows = beliefs_table(sample_report())
    assert title == "Beliefs"
    assert headers == ["claim_id", "posterior"]
    assert rows == [["clm_a", 0.9], ["clm_b", 0.25]]


def test_beliefs_table_skips_non_mapping_entries() -> None:
    _, _, rows = beliefs_table({"beliefs": ["x", {"claim_id": "c", "posterior": 0.5}]})
    assert rows == [["c", 0.5]]


def test_evolution_table_defaults_to_zero() -> None:
    _, _, rows = evolution_table({})
    assert rows == [["generations", 0], ["accepted", 0], ["rejected", 0]]


def test_evolution_table_reads_counts() -> None:
    _, _, rows = evolution_table(sample_report())
    assert dict((name, value) for name, value in rows)["accepted"] == 3


def test_recovery_table_summarises_planted_vs_recovered() -> None:
    _, _, rows = recovery_table(sample_report())
    as_map = {name: value for name, value in rows}
    assert as_map == {"planted": 2, "recovered": 2, "rate": 1.0}


def test_recovered_links_returns_statement_and_flag_pairs() -> None:
    links = recovered_links(sample_report())
    assert links == [("A causes B", True), ("B causes C", True)]


def test_recovered_links_is_empty_without_a_recovery_section() -> None:
    assert recovered_links({}) == []
    assert recovered_links({"recovered": {"links": "nope"}}) == []


def test_cost_table_counts_tokens_without_pricing() -> None:
    title, _, rows = cost_table(sample_report())
    assert "not billed" in title
    as_map = {name: value for name, value in rows}
    assert as_map["total_tokens"] == 1500
    assert "price" not in as_map and "cost_usd" not in as_map
