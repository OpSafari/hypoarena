"""Synthetic vocabulary: pool coverage, distinct sampling and verb tables."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    CAUSAL_RELATIONS,
    ENTITY_POOLS,
    MODEL_SYSTEMS,
    RELATION_VERBS,
    canonical_verb,
    draw_entities,
    draw_entity,
    pool_names,
    relation_verbs,
    verb_for,
)


def test_pools_are_non_empty_and_named() -> None:
    assert pool_names() == ("gene", "metabolite", "phenotype", "protein")
    assert all(len(ENTITY_POOLS[name]) >= 4 for name in pool_names())
    assert len(MODEL_SYSTEMS) >= 4


def test_every_relation_has_several_verbs() -> None:
    assert set(RELATION_VERBS) == set(PredictedRelation)
    assert all(len(verbs) >= 3 for verbs in RELATION_VERBS.values())
    assert set(CAUSAL_RELATIONS) < set(PredictedRelation)
    assert PredictedRelation.ASSOCIATES not in CAUSAL_RELATIONS


def test_canonical_verb_is_the_first_listed_surface_form() -> None:
    for relation in PredictedRelation:
        assert canonical_verb(relation) == RELATION_VERBS[relation][0]
        assert relation_verbs(relation)[0] == canonical_verb(relation)


def test_sampling_is_seeded_and_distinct() -> None:
    first = draw_entities(Random("hypoarena:1"), "protein", 3)
    second = draw_entities(Random("hypoarena:1"), "protein", 3)
    assert first == second
    assert len(set(first)) == 3
    assert draw_entity(Random("hypoarena:1"), "gene") in ENTITY_POOLS["gene"]


def test_verb_choice_is_seeded() -> None:
    relation = PredictedRelation.INCREASES
    assert verb_for(Random("x"), relation) == verb_for(Random("x"), relation)
    assert verb_for(Random("x"), relation) in RELATION_VERBS[relation]


def test_unknown_pools_and_oversized_samples_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown entity pool"):
        draw_entity(Random(1), "cells")
    with pytest.raises(ValidationError, match="too small"):
        draw_entities(Random(1), "gene", 99)
    with pytest.raises(ValidationError, match="count"):
        draw_entities(Random(1), "gene", -1)
