"""Evolution configuration: validation, engine construction and fingerprints."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.evolve import (
    DEFAULT_NOVELTY_THRESHOLD,
    OPERATORS,
    EvolutionConfig,
    EvolutionEngine,
)


def test_defaults_are_golden() -> None:
    config = EvolutionConfig()
    assert config.operators == OPERATORS
    assert config.per_operator == 1
    assert config.generations == 1
    assert config.novelty_threshold == DEFAULT_NOVELTY_THRESHOLD == 0.9


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        EvolutionConfig(operators=("teleport",))
    with pytest.raises(ValidationError, match="at least one operator"):
        EvolutionConfig(operators=())
    with pytest.raises(ValidationError, match="per_operator"):
        EvolutionConfig(per_operator=0)
    with pytest.raises(ValidationError, match="generations"):
        EvolutionConfig(generations=0)
    with pytest.raises(ValidationError, match="novelty_threshold"):
        EvolutionConfig(novelty_threshold=0.0)
    with pytest.raises(ValidationError, match="novelty_threshold"):
        EvolutionConfig(novelty_threshold=1.5)


def test_the_novelty_configuration_follows_the_threshold() -> None:
    config = EvolutionConfig(novelty_threshold=0.75)
    novelty = config.novelty_config()
    assert novelty.threshold == 0.75
    assert novelty.content_only is True
    assert novelty.shingle_unit == "word"


def test_engine_construction_carries_every_setting() -> None:
    config = EvolutionConfig(seed=5, operators=("narrow_scope",), per_operator=2)
    engine = config.engine()
    assert isinstance(engine, EvolutionEngine)
    assert engine.seed == 5
    assert engine.operators == ("narrow_scope",)
    assert engine.per_operator == 2
    assert engine.novelty_config is not None


def test_fingerprints_are_stable_and_setting_sensitive() -> None:
    assert EvolutionConfig().fingerprint() == EvolutionConfig().fingerprint()
    assert EvolutionConfig(seed=1).fingerprint() != EvolutionConfig().fingerprint()
    assert (
        EvolutionConfig(generations=3).fingerprint() != EvolutionConfig().fingerprint()
    )
