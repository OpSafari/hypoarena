"""Schema enums and record validation."""

from __future__ import annotations

from hypoarena.schema import (
    RELATION_OPPOSITES,
    SCHEMA_VERSION,
    ClaimRelation,
    EvidencePolarity,
    PredictedRelation,
    is_directional,
    opposite_relation,
    relations_conflict,
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


def test_directional_relations_exclude_pure_association() -> None:
    assert is_directional(PredictedRelation.INCREASES) is True
    assert is_directional(PredictedRelation.CAUSES) is True
    assert is_directional(PredictedRelation.ASSOCIATES) is False


def test_opposite_relation_pairs_are_symmetric() -> None:
    for relation, opposite in RELATION_OPPOSITES.items():
        assert RELATION_OPPOSITES[opposite] is relation


def test_opposite_relation_is_none_without_a_counterpart() -> None:
    assert opposite_relation(PredictedRelation.CAUSES) is None
    assert opposite_relation(PredictedRelation.ASSOCIATES) is None


def test_relations_conflict_in_both_directions() -> None:
    assert relations_conflict(PredictedRelation.INCREASES, PredictedRelation.DECREASES)
    assert relations_conflict(PredictedRelation.ENABLES, PredictedRelation.INHIBITS)


def test_relations_do_not_conflict_with_themselves_or_association() -> None:
    assert not relations_conflict(
        PredictedRelation.INCREASES, PredictedRelation.INCREASES
    )
    assert not relations_conflict(PredictedRelation.INCREASES, PredictedRelation.CAUSES)
    assert not relations_conflict(
        PredictedRelation.ASSOCIATES, PredictedRelation.ASSOCIATES
    )
