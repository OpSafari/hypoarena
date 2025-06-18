"""Planted chains: structure, determinism and validation."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    PlantedChain,
    PlantedLink,
    SyntheticConfig,
    plant_chains,
)


def link(**overrides: object) -> PlantedLink:
    payload: dict[str, object] = {
        "chain_id": "chn_0123456789ab",
        "subject": "protein A",
        "target": "cell growth",
        "relation": PredictedRelation.INCREASES,
        "system": "HeLa cells",
    }
    payload.update(overrides)
    return PlantedLink(**payload)  # type: ignore[arg-type]


def chain(
    variables: tuple[str, ...] = ("protein A", "gene G1", "cell growth"),
) -> PlantedChain:
    links = tuple(
        link(subject=variables[index], target=variables[index + 1])
        for index in range(len(variables) - 1)
    )
    return PlantedChain("chn_0123456789ab", variables, "HeLa cells", links)


def test_link_exposes_canonical_statement_and_key() -> None:
    planted = link()
    assert planted.statement == "protein A increases cell growth"
    assert planted.key() == ("protein a", "increases", "cell growth")
    assert planted.is_true is True
    assert link(kind="competing").is_true is False


def test_link_rejects_unknown_kinds_and_degenerate_variables() -> None:
    with pytest.raises(ValidationError, match="unknown link kind"):
        link(kind="speculative")
    with pytest.raises(ValidationError, match="must differ"):
        link(subject="Protein A", target="protein  a")
    with pytest.raises(ValidationError, match="blank"):
        link(subject=" ")


def test_chain_requires_one_link_per_pair_and_distinct_variables() -> None:
    assert len(chain().links) == 2
    with pytest.raises(ValidationError, match="distinct"):
        PlantedChain("chn_0123456789ab", ("a", "a"), "HeLa cells", (link(),))
    with pytest.raises(ValidationError, match="at least two"):
        PlantedChain("chn_0123456789ab", ("a",), "HeLa cells", ())
    with pytest.raises(ValidationError, match="one link per variable pair"):
        PlantedChain("chn_0123456789ab", ("a", "b", "c"), "HeLa cells", (link(),))


def test_link_between_finds_and_misses() -> None:
    planted = chain()
    assert planted.link_between("protein A", "gene G1") is planted.links[0]
    assert planted.link_between("gene G1", "protein A") is None


def test_planting_is_deterministic_per_seed() -> None:
    config = SyntheticConfig(seed=11, chains=3, chain_length=3)
    first = plant_chains(config, config.rng())
    second = plant_chains(config, config.rng())
    assert first == second
    assert len(first) == 3
    assert all(len(item.links) == 2 for item in first)


def test_planted_chains_use_distinct_variables_and_end_on_a_phenotype() -> None:
    from hypoarena.synthetic import ENTITY_POOLS

    config = SyntheticConfig(seed=3, chains=4, chain_length=4)
    for planted in plant_chains(config, config.rng()):
        assert len(set(planted.variables)) == 4
        assert planted.variables[-1] in ENTITY_POOLS["phenotype"]
        assert all(item.is_true for item in planted.links)


def test_different_seeds_plant_different_chains() -> None:
    first = plant_chains(SyntheticConfig(seed=1), SyntheticConfig(seed=1).rng())
    second = plant_chains(SyntheticConfig(seed=2), SyntheticConfig(seed=2).rng())
    assert first != second
