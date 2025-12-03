"""Paraphrase clusters: size, distinctness and measurable overlap."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    PlantedLink,
    paraphrase_cluster,
    paraphrase_overlap,
)


def link() -> PlantedLink:
    return PlantedLink(
        chain_id="chn_0123456789ab",
        subject="protein A",
        target="cell growth",
        relation=PredictedRelation.INCREASES,
        system="HeLa cells",
    )


def test_cluster_is_deterministic_and_distinct() -> None:
    first = paraphrase_cluster(Random("p"), link(), 4)
    second = paraphrase_cluster(Random("p"), link(), 4)
    assert first == second
    assert len(set(first)) == 4
    assert all("protein A" in form for form in first)


def test_cluster_mentions_the_system_and_target() -> None:
    for form in paraphrase_cluster(Random("q"), link(), 5):
        assert "cell growth" in form
        assert "HeLa cells" in form


def test_cluster_size_is_honoured_beyond_the_template_space() -> None:
    forms = paraphrase_cluster(Random("r"), link(), 20)
    assert len(forms) == 20
    assert len(set(forms)) == 20
    assert any(form.startswith("Replicate") for form in forms)


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="paraphrase count"):
        paraphrase_cluster(Random("s"), link(), 0)


def test_overlap_is_high_within_a_cluster_and_symmetric() -> None:
    forms = paraphrase_cluster(Random("t"), link(), 3)
    score = paraphrase_overlap(forms[0], forms[1])
    assert score == paraphrase_overlap(forms[1], forms[0])
    assert score > 0.3


def test_overlap_separates_unrelated_text() -> None:
    assert paraphrase_overlap("protein A increases cell growth", "batch drift") < 0.2
    assert paraphrase_overlap("", "") == 1.0
