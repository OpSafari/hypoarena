"""Schema enums and record validation."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import (
    RELATION_OPPOSITES,
    SCHEMA_VERSION,
    ClaimRelation,
    EvidencePolarity,
    PredictedRelation,
    Scope,
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


def test_scope_accepts_a_population_and_conditions() -> None:
    scope = Scope("hek293 cells", ("hypoxia", "serum starved"))
    assert scope.population == "hek293 cells"
    assert scope.conditions == ("hypoxia", "serum starved")


def test_scope_rejects_blank_population_and_conditions() -> None:
    with pytest.raises(ValidationError, match="population"):
        Scope("   ")
    with pytest.raises(ValidationError, match="blank"):
        Scope("cells", ("hypoxia", " "))


def test_scope_rejects_duplicate_conditions() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        Scope("cells", ("hypoxia", "hypoxia"))


def test_narrowing_adds_a_condition_and_is_idempotent() -> None:
    base = Scope("cells")
    once = base.narrowed("hypoxia")
    assert once.conditions == ("hypoxia",)
    assert once.narrowed("hypoxia") is once
    assert base.conditions == ()


def test_narrowing_rejects_blank_conditions() -> None:
    with pytest.raises(ValidationError):
        Scope("cells").narrowed("  ")


def test_is_narrower_than_is_strict_in_both_directions() -> None:
    base = Scope("cells")
    narrow = Scope("cells", ("hypoxia",))
    assert narrow.is_narrower_than(base) is True
    assert base.is_narrower_than(narrow) is False
    assert narrow.is_narrower_than(narrow) is False


def test_is_narrower_than_requires_the_same_population() -> None:
    assert not Scope("mice", ("hypoxia",)).is_narrower_than(Scope("cells"))


def test_scope_signature_ignores_condition_order() -> None:
    first = Scope("HEK293 cells", ("hypoxia", "serum starved"))
    second = Scope("hek293 cells", ("serum starved", "hypoxia"))
    assert first.signature() == second.signature()
    assert first.signature() != Scope("other cells").signature()
