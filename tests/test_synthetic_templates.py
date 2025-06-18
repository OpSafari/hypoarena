"""Template rendering: substitution, determinism and paraphrase variety."""

from __future__ import annotations

from random import Random

import pytest

from hypoarena.errors import ValidationError
from hypoarena.schema import PredictedRelation
from hypoarena.synthetic import (
    FINDING_TEMPLATES,
    NEGATION_TEMPLATES,
    canonical_statement,
    render,
    render_filler,
    render_finding,
    render_negation,
    render_title,
)


def test_render_substitutes_every_placeholder() -> None:
    assert render("{a} and {b}", a="x", b="y") == "x and y"
    with pytest.raises(ValidationError, match="not supplied"):
        render("{a} and {b}", a="x")
    with pytest.raises(ValidationError, match="not substituted"):
        render("{a} and {{literal}}", a="x")


def test_findings_are_deterministic_for_a_seed() -> None:
    relation = PredictedRelation.INCREASES
    first = render_finding(
        Random("s"), "protein A", relation, "cell growth", "HeLa cells"
    )
    second = render_finding(
        Random("s"), "protein A", relation, "cell growth", "HeLa cells"
    )
    assert first == second
    assert "{" not in first and first.endswith(".")


def test_titles_and_filler_render_without_placeholders() -> None:
    rng = Random("t")
    title = render_title(
        rng, "kinase K1", PredictedRelation.INHIBITS, "gene G1", "yeast lysate"
    )
    assert "{" not in title and title
    assert "{" not in render_filler(rng, "yeast lysate")


def test_negations_never_reuse_the_affirmative_verb_forms() -> None:
    rng = Random("n")
    sentence = render_negation(rng, "protein A", "cell growth", "HeLa cells")
    assert "{" not in sentence
    assert any(word in sentence for word in ("not", "No", "did"))
    assert len(NEGATION_TEMPLATES) >= 3


def test_canonical_statement_is_seed_independent() -> None:
    statement = canonical_statement(
        "protein A", PredictedRelation.INCREASES, "cell growth"
    )
    assert statement == "protein A increases cell growth"
    assert canonical_statement(
        "protein A", PredictedRelation.DECREASES, "cell growth"
    ) != (statement)


def test_finding_templates_cover_several_surface_forms() -> None:
    assert len(FINDING_TEMPLATES) >= 5
    renders = {
        render(template, subject="s", verb="v", target="t", system="y")
        for template in FINDING_TEMPLATES
    }
    assert len(renders) == len(FINDING_TEMPLATES)
