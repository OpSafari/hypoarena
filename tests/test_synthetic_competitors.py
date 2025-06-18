"""Competing hypotheses and contradictions planted alongside the truth."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.errors import ValidationError
from hypoarena.synthetic import (
    PlantedChain,
    SyntheticConfig,
    competing_links,
    contradiction_links,
    plant_chains,
)


def first_chain() -> PlantedChain:
    config = SyntheticConfig(seed=17, chains=1, chain_length=3)
    return plant_chains(config, config.rng())[0]


def test_competing_links_change_the_relation_only() -> None:
    chain = first_chain()
    rivals = competing_links(chain, Random("c"), 2)
    assert len(rivals) == 2
    for rival, planted in zip(rivals, chain.links, strict=True):
        assert rival.kind == "competing"
        assert (rival.subject, rival.target) == (planted.subject, planted.target)
        assert rival.relation is not planted.relation
        assert rival.is_true is False


def test_competitor_count_is_clamped_to_available_links() -> None:
    chain = first_chain()
    assert len(competing_links(chain, Random("c"), 99)) == len(chain.links)
    assert competing_links(chain, Random("c"), 0) == ()


def test_contradictions_keep_the_relation_and_mark_the_kind() -> None:
    chain = first_chain()
    negated = contradiction_links(chain, Random("n"), 1)
    assert len(negated) == 1
    assert negated[0].kind == "contradiction"
    assert any(
        item.relation is negated[0].relation and item.subject == negated[0].subject
        for item in chain.links
    )


def test_contradiction_selection_is_seeded() -> None:
    chain = first_chain()
    assert contradiction_links(chain, Random("n"), 2) == contradiction_links(
        chain, Random("n"), 2
    )


def test_negative_counts_are_rejected() -> None:
    chain = first_chain()
    with pytest.raises(ValidationError, match="competitor count"):
        competing_links(chain, Random("c"), -1)
    with pytest.raises(ValidationError, match="contradiction count"):
        contradiction_links(chain, Random("n"), -1)
