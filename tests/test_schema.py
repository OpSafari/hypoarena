"""Schema enums and record validation."""

from __future__ import annotations

from hypoarena.schema import (
    SCHEMA_VERSION,
    ClaimRelation,
    EvidencePolarity,
    PredictedRelation,
)


def test_schema_version_is_pinned() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_predicted_relation_members_are_exact() -> None:
    assert [member.value for member in PredictedRelation] == [
        "increases",
        "decreases",
        "enables",
        "inhibits",
        "causes",
        "associates",
    ]


def test_evidence_polarity_members_are_exact() -> None:
    assert [member.value for member in EvidencePolarity] == [
        "support",
        "refute",
        "neutral",
    ]


def test_claim_relation_members_are_exact() -> None:
    assert [member.value for member in ClaimRelation] == [
        "entails",
        "contradicts",
        "refines",
    ]


def test_enums_compare_by_value_and_reject_unknowns() -> None:
    assert PredictedRelation("increases") is PredictedRelation.INCREASES
    assert PredictedRelation.INCREASES == "increases"
    try:
        PredictedRelation("teleports")
    except ValueError:
        pass
    else:  # pragma: no cover - guards the enum contract
        raise AssertionError("unknown relation accepted")
