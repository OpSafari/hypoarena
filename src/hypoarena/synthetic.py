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

from dataclasses import asdict, dataclass
from random import Random

from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
    make_id,
)
from hypoarena.schema import (
    PredictedRelation,
)
from hypoarena.text import (
    content_tokens,
    normalize,
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


MIN_CHAIN_LENGTH = 2
MAX_CHAIN_LENGTH = 6


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for one synthetic corpus generation.

    Counts are exact, not targets: :meth:`expected_documents` is what
    :func:`build_corpus` must produce, and a test asserts the equality so a
    generator change cannot silently shrink a corpus.
    """

    seed: int = 270106
    chains: int = 3
    chain_length: int = 3
    paraphrases_per_link: int = 2
    competitors_per_chain: int = 1
    contradictions_per_chain: int = 1
    distractor_documents: int = 4
    filler_sentences: int = 2
    source_tag: str = "synthetic:v1"

    def __post_init__(self) -> None:
        if self.chains < 1:
            raise ValidationError("chains must be >= 1", chains=self.chains)
        if not MIN_CHAIN_LENGTH <= self.chain_length <= MAX_CHAIN_LENGTH:
            raise ValidationError(
                "chain length out of range",
                chain_length=self.chain_length,
                minimum=MIN_CHAIN_LENGTH,
                maximum=MAX_CHAIN_LENGTH,
            )
        if self.paraphrases_per_link < 1:
            raise ValidationError(
                "paraphrases_per_link must be >= 1",
                paraphrases_per_link=self.paraphrases_per_link,
            )
        for name in (
            "competitors_per_chain",
            "contradictions_per_chain",
            "distractor_documents",
            "filler_sentences",
        ):
            if getattr(self, name) < 0:
                raise ValidationError(
                    f"{name} must be >= 0", **{name: getattr(self, name)}
                )
        if not self.source_tag.strip():
            raise ValidationError("source_tag must not be blank")

    def rng(self) -> Random:
        """Return the seeded generator used for every draw in a build."""
        return Random(f"hypoarena:synthetic:{self.seed}")

    def links_per_chain(self) -> int:
        """Number of causal links in one chain."""
        return self.chain_length - 1

    def expected_documents(self) -> int:
        """Exact document count a build with this configuration must produce."""
        planted = self.chains * self.links_per_chain() * self.paraphrases_per_link
        competing = self.chains * min(
            self.competitors_per_chain, self.links_per_chain()
        )
        contradictions = self.chains * min(
            self.contradictions_per_chain, self.links_per_chain()
        )
        return planted + competing + contradictions + self.distractor_documents

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(asdict(self))


LINK_KINDS = ("planted", "competing", "contradiction")


@dataclass(frozen=True)
class PlantedLink:
    """One planted relation between two variables.

    ``kind`` records the role the link plays in the ground truth: ``planted``
    links are the ones a discovery run should recover, ``competing`` links are
    plausible alternatives about the same variable pair, and ``contradiction``
    links are negated surface forms of a planted link.
    """

    chain_id: str
    subject: str
    target: str
    relation: PredictedRelation
    system: str
    kind: str = "planted"

    def __post_init__(self) -> None:
        if self.kind not in LINK_KINDS:
            raise ValidationError(
                "unknown link kind", kind=self.kind, allowed=list(LINK_KINDS)
            )
        if not self.subject.strip() or not self.target.strip():
            raise ValidationError("planted link variables must not be blank")
        if normalize(self.subject) == normalize(self.target):
            raise ValidationError(
                "planted link variables must differ",
                subject=self.subject,
                target=self.target,
            )

    @property
    def statement(self) -> str:
        """The canonical statement of this link."""
        return canonical_statement(self.subject, self.relation, self.target)

    def key(self) -> tuple[str, str, str]:
        """Normalized identity used to match recovered claims against truth."""
        return (
            normalize(self.subject),
            self.relation.value,
            normalize(self.target),
        )

    @property
    def is_true(self) -> bool:
        """True for links that the ground truth considers correct."""
        return self.kind == "planted"


@dataclass(frozen=True)
class PlantedChain:
    """A chain of variables plus the links that connect consecutive pairs."""

    chain_id: str
    variables: tuple[str, ...]
    system: str
    links: tuple[PlantedLink, ...]

    def __post_init__(self) -> None:
        if len(self.variables) < MIN_CHAIN_LENGTH:
            raise ValidationError(
                "a chain needs at least two variables", count=len(self.variables)
            )
        if len(set(self.variables)) != len(self.variables):
            raise ValidationError(
                "chain variables must be distinct", variables=self.variables
            )
        if len(self.links) != len(self.variables) - 1:
            raise ValidationError(
                "chain must have one link per variable pair",
                variables=len(self.variables),
                links=len(self.links),
            )

    def link_between(self, subject: str, target: str) -> PlantedLink | None:
        """Return the link connecting two variables, if any."""
        for link in self.links:
            if link.subject == subject and link.target == target:
                return link
        return None


def draw_variables(rng: Random, count: int) -> tuple[str, ...]:
    """Draw ``count`` distinct variables, ending on a phenotype."""
    variables: list[str] = []
    pools = [*SUBJECT_POOLS, *OBJECT_POOLS]
    attempts = 0
    while len(variables) < count and attempts < count * 20:
        attempts += 1
        pool = pools[attempts % len(pools)]
        candidate = draw_entity(rng, pool)
        if candidate not in variables:
            variables.append(candidate)
    if len(variables) < count:
        raise ValidationError(
            "vocabulary is too small for the requested chain length", count=count
        )
    phenotype = draw_entity(rng, "phenotype")
    while phenotype in variables:
        phenotype = draw_entity(rng, "phenotype")
    variables[-1] = phenotype
    return tuple(variables)


def plant_chains(config: SyntheticConfig, rng: Random) -> tuple[PlantedChain, ...]:
    """Plant ``config.chains`` causal chains with randomly chosen directions."""
    chains: list[PlantedChain] = []
    for index in range(config.chains):
        chain_id = make_id("chn", config.seed, "chain", index)
        system = rng.choice(MODEL_SYSTEMS)
        variables = draw_variables(rng, config.chain_length)
        links = tuple(
            PlantedLink(
                chain_id=chain_id,
                subject=variables[position],
                target=variables[position + 1],
                relation=rng.choice(CAUSAL_RELATIONS),
                system=system,
            )
            for position in range(len(variables) - 1)
        )
        chains.append(
            PlantedChain(
                chain_id=chain_id, variables=variables, system=system, links=links
            )
        )
    return tuple(chains)


def paraphrase_cluster(rng: Random, link: PlantedLink, count: int) -> tuple[str, ...]:
    """Return ``count`` distinct surface forms of one planted finding.

    Variety comes from the template and verb tables; when those are exhausted a
    numbered replicate form keeps the cluster size exact, so paraphrase recall
    measured on the cluster is never limited by the generator running out of
    surface forms.
    """
    if count < 1:
        raise ValidationError("paraphrase count must be >= 1", count=count)
    forms: list[str] = []
    attempts = 0
    while len(forms) < count and attempts < count * 12:
        attempts += 1
        candidate = render_finding(
            rng, link.subject, link.relation, link.target, link.system
        )
        if candidate not in forms:
            forms.append(candidate)
    while len(forms) < count:
        index = len(forms) + 1
        forms.append(f"Replicate {index} in {link.system}: {link.statement.lower()}.")
    return tuple(forms)


def paraphrase_overlap(first: str, second: str) -> float:
    """Return the word-level Jaccard overlap of two surface forms.

    Bundled here because the generator is what decides how similar a paraphrase
    cluster is; dedup measurements quote this value to explain their recall.
    """
    left = set(content_tokens(first))
    right = set(content_tokens(second))
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)
