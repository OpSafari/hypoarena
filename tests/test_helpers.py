"""The shared builders must produce valid, overridable records."""

from __future__ import annotations

import pytest

from helpers import (
    CLAIM_ID,
    DOCUMENT_ID,
    sample_citation,
    sample_claim,
    sample_provenance,
    sample_scope,
)
from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation


def test_builders_produce_valid_records() -> None:
    assert sample_scope().population == "hek293 cells"
    assert sample_citation().document_id == DOCUMENT_ID
    assert sample_provenance().origin == "agent"
    assert sample_claim().claim_id == CLAIM_ID


def test_sample_claim_is_cited_and_directional() -> None:
    claim = sample_claim()
    assert claim.is_cited is True
    assert claim.relation is PredictedRelation.INCREASES
    assert claim.signature()


def test_overrides_replace_exactly_one_field() -> None:
    claim = sample_claim(relation=PredictedRelation.DECREASES)
    assert claim.relation is PredictedRelation.DECREASES
    assert claim.statement == sample_claim().statement


def test_overrides_can_produce_invalid_records() -> None:
    with pytest.raises(ValidationError):
        sample_claim(statement="  ")
    with pytest.raises(ValidationError):
        sample_citation(start=99, end=10)
