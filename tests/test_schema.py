"""Schema enums and record validation."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import (
    PROVENANCE_ORIGINS,
    RELATION_OPPOSITES,
    SCHEMA_VERSION,
    Citation,
    Claim,
    ClaimRelation,
    Evidence,
    EvidencePolarity,
    PredictedRelation,
    Provenance,
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


def make_citation(**overrides: object) -> Citation:
    payload: dict[str, object] = {
        "document_id": "doc_0123456789ab",
        "start": 10,
        "end": 25,
        "quote": "kinase binds",
    }
    payload.update(overrides)
    return Citation(**payload)  # type: ignore[arg-type]


def test_citation_accepts_a_well_formed_span() -> None:
    citation = make_citation()
    assert citation.length == 15
    assert citation.key() == ("doc_0123456789ab", 10, 25)


def test_citation_rejects_malformed_document_ids() -> None:
    with pytest.raises(ValidationError, match="malformed"):
        make_citation(document_id="paper_1")
    with pytest.raises(ValidationError, match="malformed"):
        make_citation(document_id="clm_0123456789ab")


def test_citation_rejects_bad_offsets() -> None:
    with pytest.raises(ValidationError, match=">= 0"):
        make_citation(start=-1, end=5)
    with pytest.raises(ValidationError, match="non-empty"):
        make_citation(start=10, end=10)
    with pytest.raises(ValidationError, match="non-empty"):
        make_citation(start=10, end=4)


def test_citation_rejects_blank_or_overlong_quotes() -> None:
    with pytest.raises(ValidationError, match="blank"):
        make_citation(quote="   ")
    with pytest.raises(ValidationError, match="longer than its span"):
        make_citation(start=10, end=12, quote="way too long for the span")


def test_citation_overlap_is_symmetric_and_document_scoped() -> None:
    first = make_citation(start=10, end=25)
    second = make_citation(start=20, end=35)
    third = make_citation(start=25, end=40)
    other_doc = make_citation(document_id="doc_ffffffffffff", start=20, end=35)
    assert first.overlaps(second) and second.overlaps(first)
    assert not first.overlaps(third)
    assert not first.overlaps(other_doc)


def test_provenance_accepts_known_origins() -> None:
    for origin in sorted(PROVENANCE_ORIGINS):
        record = Provenance(
            origin=origin,
            agent_id="agt_01" if origin == "agent" else None,
            parents=("clm_0123456789ab",) if origin == "evolved" else (),
        )
        assert record.origin == origin


def test_provenance_rejects_unknown_origins() -> None:
    with pytest.raises(ValidationError, match="unknown provenance origin"):
        Provenance(origin="oracle")


def test_agent_provenance_requires_an_agent_id() -> None:
    with pytest.raises(ValidationError, match="agent_id"):
        Provenance(origin="agent")
    assert Provenance(origin="agent", agent_id="agt_01").agent_id == "agt_01"


def test_evolved_provenance_requires_parents_and_valid_ids() -> None:
    with pytest.raises(ValidationError, match="parent"):
        Provenance(origin="evolved")
    with pytest.raises(ValidationError, match="malformed"):
        Provenance(origin="evolved", parents=("not-an-id",))


def test_provenance_rejects_negative_generations_and_bad_hashes() -> None:
    with pytest.raises(ValidationError, match="generation"):
        Provenance(origin="user", generation=-1)
    with pytest.raises(ValidationError, match="hex"):
        Provenance(origin="user", corpus_hash="xyz")
    with pytest.raises(ValidationError, match="blank"):
        Provenance(origin="user", notes="   ")


def test_is_reproducible_requires_seed_and_corpus_hash() -> None:
    assert not Provenance(origin="synthetic").is_reproducible
    assert not Provenance(origin="synthetic", seed=7).is_reproducible
    assert Provenance(
        origin="synthetic", seed=7, corpus_hash="0123456789abcdef"
    ).is_reproducible
    assert not Provenance(
        origin="user", seed=7, corpus_hash="0123456789abcdef"
    ).is_reproducible


def make_claim(**overrides: object) -> Claim:
    payload: dict[str, object] = {
        "claim_id": "clm_0123456789ab",
        "statement": "Protein A increases the expression of gene B",
        "subject": "protein A",
        "object": "gene B expression",
        "relation": PredictedRelation.INCREASES,
        "scope": Scope("hek293 cells"),
    }
    payload.update(overrides)
    return Claim(**payload)  # type: ignore[arg-type]


def test_claim_accepts_a_well_formed_record() -> None:
    claim = make_claim()
    assert claim.variables == ("protein A", "gene B expression")
    assert claim.is_cited is False
    assert claim.provenance.origin == "user"


def test_claim_rejects_malformed_ids_and_wrong_prefixes() -> None:
    with pytest.raises(ValidationError, match="malformed"):
        make_claim(claim_id="claim_1")
    with pytest.raises(ValidationError, match="expected_prefix"):
        make_claim(claim_id="evd_0123456789ab")


def test_claim_rejects_blank_statements_and_variables() -> None:
    with pytest.raises(ValidationError, match="statement"):
        make_claim(statement="   ")
    with pytest.raises(ValidationError, match="variables"):
        make_claim(subject=" ")


def test_claim_rejects_identical_subject_and_object() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        make_claim(subject="Protein A", object="protein  a")


def test_claim_rejects_wrong_field_types() -> None:
    with pytest.raises(ValidationError, match="PredictedRelation"):
        make_claim(relation="increases")
    with pytest.raises(ValidationError, match="Scope"):
        make_claim(scope="hek293 cells")
    with pytest.raises(ValidationError, match="Provenance"):
        make_claim(provenance="synthetic")


def test_claim_rejects_duplicate_and_non_citation_entries() -> None:
    citation = make_citation()
    with pytest.raises(ValidationError, match="same span twice"):
        make_claim(citations=(citation, citation))
    with pytest.raises(ValidationError, match="Citation objects"):
        make_claim(citations=("doc_0123456789ab:10:25",))


def test_claim_rejects_blank_mechanism_but_allows_none() -> None:
    assert make_claim(mechanism=None).mechanism is None
    with pytest.raises(ValidationError, match="mechanism"):
        make_claim(mechanism="  ")


def test_claim_signature_ignores_ids_citations_and_wording_case() -> None:
    base = make_claim()
    restated = make_claim(
        claim_id="clm_ffffffffffff",
        statement="protein a INCREASES the expression of gene b",
        citations=(make_citation(),),
    )
    assert base.signature() == restated.signature()


def test_claim_signature_separates_relation_and_scope() -> None:
    base = make_claim()
    assert (
        base.signature() != make_claim(relation=PredictedRelation.DECREASES).signature()
    )
    assert base.signature() != make_claim(scope=Scope("mice")).signature()


def test_normalized_statement_is_lowercase_and_collapsed() -> None:
    assert (
        make_claim(statement="  A   Binds  B! ").normalized_statement() == "a binds b"
    )


def make_evidence(**overrides: object) -> Evidence:
    payload: dict[str, object] = {
        "evidence_id": "evd_0123456789ab",
        "statement": "ChIP-seq shows binding enrichment at the promoter",
        "polarity": EvidencePolarity.SUPPORT,
        "strength": 0.8,
        "citations": (make_citation(),),
        "method": "synthetic_finding",
        "provenance": Provenance(origin="synthetic", seed=270106),
    }
    payload.update(overrides)
    return Evidence(**payload)  # type: ignore[arg-type]


def test_evidence_accepts_a_well_formed_record() -> None:
    evidence = make_evidence()
    assert evidence.weighted_polarity == pytest.approx(0.8)
    assert make_evidence(
        polarity=EvidencePolarity.REFUTE
    ).weighted_polarity == pytest.approx(-0.8)


def test_neutral_evidence_carries_zero_weight() -> None:
    evidence = make_evidence(polarity=EvidencePolarity.NEUTRAL, strength=0.0)
    assert evidence.weighted_polarity == 0.0
    with pytest.raises(ValidationError, match="zero strength"):
        make_evidence(polarity=EvidencePolarity.NEUTRAL, strength=0.4)


def test_evidence_strength_bounds_are_inclusive_and_enforced() -> None:
    assert make_evidence(strength=0.0).strength == 0.0
    assert make_evidence(strength=1.0).strength == 1.0
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValidationError, match="strength"):
            make_evidence(strength=bad)


def test_evidence_rejects_integer_strength() -> None:
    with pytest.raises(ValidationError, match="finite float"):
        make_evidence(strength=1)


def test_evidence_requires_citations_and_rejects_duplicates() -> None:
    with pytest.raises(ValidationError, match="at least one corpus span"):
        make_evidence(citations=())
    citation = make_citation()
    with pytest.raises(ValidationError, match="same span twice"):
        make_evidence(citations=(citation, citation))


def test_evidence_rejects_malformed_ids_and_blank_fields() -> None:
    with pytest.raises(ValidationError, match="malformed"):
        make_evidence(evidence_id="clm_0123456789ab")
    with pytest.raises(ValidationError, match="statement"):
        make_evidence(statement=" ")
    with pytest.raises(ValidationError, match="method"):
        make_evidence(method="")


def test_evidence_validates_optional_quantities() -> None:
    assert make_evidence(sample_size=12, effect_size=-0.5).sample_size == 12
    with pytest.raises(ValidationError, match="sample_size"):
        make_evidence(sample_size=0)
    with pytest.raises(ValidationError, match="effect_size"):
        make_evidence(effect_size=float("inf"))
