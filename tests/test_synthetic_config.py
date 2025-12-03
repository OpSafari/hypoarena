"""Generator configuration: defaults, validation and derived counts."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.synthetic import MAX_CHAIN_LENGTH, SyntheticConfig


def test_defaults_are_golden() -> None:
    config = SyntheticConfig()
    assert config.seed == 270106
    assert config.chains == 3
    assert config.chain_length == 3
    assert config.paraphrases_per_link == 2
    assert config.competitors_per_chain == 1
    assert config.contradictions_per_chain == 1
    assert config.distractor_documents == 4
    assert config.filler_sentences == 2
    assert config.source_tag == "synthetic:v1"


def test_expected_document_count_is_exact() -> None:
    config = SyntheticConfig(chains=2, chain_length=4, paraphrases_per_link=3)
    assert config.links_per_chain() == 3
    assert config.expected_documents() == 2 * 3 * 3 + 2 * 1 + 2 * 1 + 4


def test_expected_documents_clamps_per_chain_extras_to_available_links() -> None:
    config = SyntheticConfig(
        chains=1,
        chain_length=2,
        paraphrases_per_link=1,
        competitors_per_chain=5,
        contradictions_per_chain=5,
        distractor_documents=0,
    )
    assert config.links_per_chain() == 1
    assert config.expected_documents() == 3


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="chains"):
        SyntheticConfig(chains=0)
    with pytest.raises(ValidationError, match="chain length"):
        SyntheticConfig(chain_length=1)
    with pytest.raises(ValidationError, match="chain length"):
        SyntheticConfig(chain_length=MAX_CHAIN_LENGTH + 1)
    with pytest.raises(ValidationError, match="paraphrases_per_link"):
        SyntheticConfig(paraphrases_per_link=0)
    with pytest.raises(ValidationError, match="distractor_documents"):
        SyntheticConfig(distractor_documents=-1)
    with pytest.raises(ValidationError, match="source_tag"):
        SyntheticConfig(source_tag="  ")


def test_rng_is_seeded_and_reproducible() -> None:
    assert (
        SyntheticConfig(seed=5).rng().random() == SyntheticConfig(seed=5).rng().random()
    )
    assert (
        SyntheticConfig(seed=5).rng().random() != SyntheticConfig(seed=6).rng().random()
    )


def test_fingerprint_tracks_every_field() -> None:
    baseline = SyntheticConfig().fingerprint()
    assert baseline == SyntheticConfig().fingerprint()
    assert SyntheticConfig(seed=1).fingerprint() != baseline
    assert SyntheticConfig(chains=4).fingerprint() != baseline
    assert len(baseline) == 16
