"""Deterministic synthetic literature with planted ground truth.

This module generates paper-like corpora whose causal structure is *known*: each
chain of variables is planted with a direction, competing hypotheses and
paraphrase clusters are added deliberately, and the emitted claims carry exact
citations into the generated text.

Everything here is synthetic. The purpose is to give the grounding verifier,
deduplication and tournament machinery a measurable target with an oracle, not
to model real scientific literature. All randomness comes from an explicitly
seeded :class:`random.Random`, so a given seed always produces byte-identical
output.
"""

from __future__ import annotations

from random import Random

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.schema import (
    PredictedRelation,
)

ENTITY_POOLS: dict[str, tuple[str, ...]] = {
    "protein": (
        "protein A",
        "protein B",
        "kinase K1",
        "kinase K2",
        "receptor R",
        "factor F",
    ),
    "gene": ("gene G1", "gene G2", "gene G3", "promoter P1", "enhancer E1"),
    "metabolite": ("metabolite M1", "metabolite M2", "lipid L1", "calcium influx"),
    "phenotype": (
        "cell growth",
        "apoptosis rate",
        "membrane potential",
        "splice fidelity",
        "oxidative stress",
    ),
}
MODEL_SYSTEMS: tuple[str, ...] = (
    "HEK293 cells",
    "HeLa cells",
    "mouse liver",
    "yeast lysate",
    "primary neurons",
)
RELATION_VERBS: dict[PredictedRelation, tuple[str, ...]] = {
    PredictedRelation.INCREASES: ("increases", "upregulates", "elevates"),
    PredictedRelation.DECREASES: ("decreases", "downregulates", "reduces"),
    PredictedRelation.ENABLES: ("enables", "licenses", "is required for"),
    PredictedRelation.INHIBITS: ("inhibits", "blocks", "suppresses"),
    PredictedRelation.CAUSES: ("causes", "triggers", "induces"),
    PredictedRelation.ASSOCIATES: (
        "is associated with",
        "correlates with",
        "co-occurs with",
    ),
}
CAUSAL_RELATIONS: tuple[PredictedRelation, ...] = (
    PredictedRelation.INCREASES,
    PredictedRelation.DECREASES,
    PredictedRelation.ENABLES,
    PredictedRelation.INHIBITS,
    PredictedRelation.CAUSES,
)
SUBJECT_POOLS: tuple[str, ...] = ("protein", "gene", "metabolite")
OBJECT_POOLS: tuple[str, ...] = ("phenotype", "gene", "metabolite")


def pool_names() -> tuple[str, ...]:
    """Return the available entity pool names in sorted order."""
    return tuple(sorted(ENTITY_POOLS))


def draw_entity(rng: Random, pool: str) -> str:
    """Draw one entity from a named pool."""
    if pool not in ENTITY_POOLS:
        raise ValidationError(
            "unknown entity pool", pool=pool, allowed=list(pool_names())
        )
    return rng.choice(ENTITY_POOLS[pool])


def draw_entities(rng: Random, pool: str, count: int) -> tuple[str, ...]:
    """Draw ``count`` distinct entities from a pool, preserving draw order."""
    if count < 0:
        raise ValidationError("entity count must be >= 0", count=count)
    if pool not in ENTITY_POOLS:
        raise ValidationError(
            "unknown entity pool", pool=pool, allowed=list(pool_names())
        )
    if count > len(ENTITY_POOLS[pool]):
        raise ValidationError(
            "pool is too small for the requested sample",
            pool=pool,
            count=count,
            available=len(ENTITY_POOLS[pool]),
        )
    return tuple(rng.sample(list(ENTITY_POOLS[pool]), count))


def verb_for(rng: Random, relation: PredictedRelation) -> str:
    """Return a surface verb for a relation, chosen with ``rng``."""
    return rng.choice(RELATION_VERBS[relation])


def canonical_verb(relation: PredictedRelation) -> str:
    """Return the deterministic verb used for canonical statements."""
    return RELATION_VERBS[relation][0]


FINDING_TEMPLATES: tuple[str, ...] = (
    "{subject} {verb} {target} in {system}.",
    "We observed that {subject} {verb} {target} in {system}.",
    "Assays in {system} show that {subject} {verb} {target}.",
    "In {system}, {subject} {verb} {target}.",
    "Our measurements indicate that {subject} {verb} {target} in {system}.",
)
TITLE_TEMPLATES: tuple[str, ...] = (
    "{subject} {verb} {target}",
    "On the relation between {subject} and {target}",
    "Evidence that {subject} {verb} {target} in {system}",
    "A {system} study of {subject} and {target}",
)
FILLER_TEMPLATES: tuple[str, ...] = (
    "Samples were collected in triplicate and processed in {system}.",
    "Controls were run in parallel across all batches in {system}.",
    "Replicates agreed within the accepted tolerance for {system}.",
    "The full protocol is summarized in the supplementary notes for {system}.",
    "Batch effects were regressed out before summarizing {system} results.",
)
NEGATION_TEMPLATES: tuple[str, ...] = (
    "We did not observe that {subject} affects {target} in {system}.",
    "No significant effect of {subject} on {target} was detected in {system}.",
    "In {system}, {subject} does not appear to change {target}.",
)


def render(template: str, **values: str) -> str:
    """Fill a template, verifying that every placeholder was supplied."""
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError) as error:
        raise ValidationError(
            "template placeholder was not supplied",
            template=template,
            missing=str(error),
            provided=sorted(values),
        ) from None
    if "{" in rendered or "}" in rendered:
        raise ValidationError(
            "template placeholder was not substituted",
            template=template,
            provided=sorted(values),
        )
    return rendered


def render_finding(
    rng: Random, subject: str, relation: PredictedRelation, target: str, system: str
) -> str:
    """Render one affirmative finding sentence."""
    return render(
        rng.choice(FINDING_TEMPLATES),
        subject=subject,
        verb=verb_for(rng, relation),
        target=target,
        system=system,
    )


def render_title(
    rng: Random, subject: str, relation: PredictedRelation, target: str, system: str
) -> str:
    """Render a paper title for a finding."""
    return render(
        rng.choice(TITLE_TEMPLATES),
        subject=subject,
        verb=verb_for(rng, relation),
        target=target,
        system=system,
    )


def render_filler(rng: Random, system: str) -> str:
    """Render a methodological sentence that carries no causal claim."""
    return render(rng.choice(FILLER_TEMPLATES), system=system)


def render_negation(rng: Random, subject: str, target: str, system: str) -> str:
    """Render a sentence contradicting an asserted effect."""
    return render(
        rng.choice(NEGATION_TEMPLATES), subject=subject, target=target, system=system
    )


def canonical_statement(subject: str, relation: PredictedRelation, target: str) -> str:
    """Return the deterministic statement used for planted ground truth."""
    return f"{subject} {canonical_verb(relation)} {target}"
