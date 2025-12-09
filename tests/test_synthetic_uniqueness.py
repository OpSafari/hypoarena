"""Ground truth must be unambiguous: one link key per planted relation."""

from __future__ import annotations

from random import Random

from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    CAUSAL_RELATIONS,
    SyntheticConfig,
    competing_links,
    generate,
    plant_chains,
)
from hypoarena.text import normalize


def planted_keys(generated: object) -> set[tuple[str, str, str]]:
    chains = generated.chains  # type: ignore[attr-defined]
    return {link.key() for chain in chains for link in chain.links}


def test_planted_keys_are_unique_across_chains() -> None:
    config = SyntheticConfig(seed=3, chains=6, chain_length=4)
    chains = plant_chains(config, config.rng())
    keys = [link.key() for chain in chains for link in chain.links]
    assert len(keys) == len(set(keys))


def test_rival_keys_never_collide_with_planted_ones() -> None:
    generated = generate(SyntheticConfig(seed=9, chains=4, chain_length=3))
    rivals = {link.key() for link in generated.competing}
    assert not (planted_keys(generated) & rivals)
    assert len(rivals) == len(generated.competing)


def test_taken_keys_steer_rival_selection() -> None:
    config = SyntheticConfig(seed=9, chains=1, chain_length=2)
    chain = plant_chains(config, config.rng())[0]
    link = chain.links[0]
    allowed = next(item for item in CAUSAL_RELATIONS if item is not link.relation)
    taken = {
        (normalize(link.subject), item.value, normalize(link.target))
        for item in CAUSAL_RELATIONS
        if item is not link.relation and item is not allowed
    }
    before = set(taken)
    rivals = competing_links(chain, Random("r"), 1, taken)
    assert rivals[0].relation is allowed
    assert rivals[0].key() not in before
    assert taken == before  # the caller's set is never mutated


def test_rivals_fall_back_when_every_option_is_taken() -> None:
    config = SyntheticConfig(seed=9, chains=1, chain_length=2)
    chain = plant_chains(config, config.rng())[0]
    link = chain.links[0]
    taken = {
        (normalize(link.subject), relation.value, normalize(link.target))
        for relation in PredictedRelation
    }
    rivals = competing_links(chain, Random("r"), 1, taken)
    assert len(rivals) == 1
    assert rivals[0].relation is not link.relation


def test_exhausted_vocabulary_is_reported_or_avoided() -> None:
    config = SyntheticConfig(seed=3, chains=14, chain_length=5)
    try:
        chains = plant_chains(config, config.rng())
    except ValidationError as error:
        assert "vocabulary exhausted" in str(error)
        return
    keys = [link.key() for chain in chains for link in chain.links]
    assert len(keys) == len(set(keys))


def test_generation_stays_deterministic_with_uniqueness_checks() -> None:
    config = SyntheticConfig(seed=12, chains=4, chain_length=3)
    assert generate(config).corpus.signature() == generate(config).corpus.signature()
