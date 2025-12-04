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

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from random import Random

from hypoarena.corpus import (
    Corpus,
    Document,
)
from hypoarena.errors import (
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
    make_id,
)
from hypoarena.schema import (
    Citation,
    Claim,
    PredictedRelation,
    Provenance,
    Scope,
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


def competing_links(
    chain: PlantedChain, rng: Random, count: int
) -> tuple[PlantedLink, ...]:
    """Return up to ``count`` alternative hypotheses about planted variable pairs.

    A competing link keeps the variable pair and asserts a *different* causal
    relation, which is what makes it a rival explanation rather than a paraphrase.
    """
    if count < 0:
        raise ValidationError("competitor count must be >= 0", count=count)
    alternatives: list[PlantedLink] = []
    for link in chain.links[:count]:
        options = [item for item in CAUSAL_RELATIONS if item is not link.relation]
        alternatives.append(
            PlantedLink(
                chain_id=chain.chain_id,
                subject=link.subject,
                target=link.target,
                relation=rng.choice(options),
                system=chain.system,
                kind="competing",
            )
        )
    return tuple(alternatives)


def contradiction_links(
    chain: PlantedChain, rng: Random, count: int
) -> tuple[PlantedLink, ...]:
    """Return up to ``count`` links whose surface text negates a planted one.

    The relation is preserved on purpose: the contradiction lives in the
    *sentence*, so a grounding verifier that only checks entities and numbers
    will still see overlap — the polarity check is what has to catch it.
    """
    if count < 0:
        raise ValidationError("contradiction count must be >= 0", count=count)
    chosen = rng.sample(list(chain.links), min(count, len(chain.links)))
    return tuple(
        PlantedLink(
            chain_id=chain.chain_id,
            subject=link.subject,
            target=link.target,
            relation=link.relation,
            system=chain.system,
            kind="contradiction",
        )
        for link in sorted(chosen, key=lambda item: (item.subject, item.target))
    )


DISTRACTOR_TEMPLATES: tuple[str, ...] = (
    "Batch {batch} was sequenced more deeply than the rest of the cohort.",
    "Instrument calibration drifted during week {batch} of the study.",
    "Sample handling time explains part of the variance in cohort {batch}.",
    "The repository migrated its identifier scheme in release {batch}.",
)


@dataclass(frozen=True)
class PlacedFinding:
    """A rendered sentence, the document carrying it and its exact span."""

    link: PlantedLink
    document: Document
    citation: Citation
    sentence: str

    @property
    def document_id(self) -> str:
        """Identifier of the document carrying this finding."""
        return self.document.document_id

    @property
    def is_negated(self) -> bool:
        """True when the sentence denies the planted relation."""
        return self.link.kind == "contradiction"


def join_sentences(
    items: Sequence[tuple[str, bool]],
) -> tuple[str, list[tuple[int, int]]]:
    """Join ``(sentence, is_finding)`` pairs and return text plus finding spans.

    Sentences are separated by single spaces and offsets are collected while the
    text is assembled, so they stay correct no matter how the caller ordered the
    items.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence, is_finding in items:
        if is_finding:
            spans.append((cursor, cursor + len(sentence)))
        parts.append(sentence)
        cursor += len(sentence) + 1
    return " ".join(parts), spans


def assemble_document(
    rng: Random,
    document_id: str,
    title: str,
    findings: Sequence[str],
    filler: Sequence[str],
    source: str,
    attributes: tuple[tuple[str, str], ...] = (),
) -> tuple[Document, list[tuple[int, int]]]:
    """Build one document, shuffling sentences while keeping spans exact."""
    items: list[tuple[str, bool]] = [(sentence, True) for sentence in findings]
    items.extend((sentence, False) for sentence in filler)
    rng.shuffle(items)
    text, spans = join_sentences(items)
    return Document(document_id, title, text, source, attributes), spans


def finding_documents(
    config: SyntheticConfig,
    rng: Random,
    link: PlantedLink,
    forms: Sequence[str],
    chain_index: int,
    link_index: int,
) -> list[PlacedFinding]:
    """Create one document per paraphrase, each citing its own exact span."""
    placed: list[PlacedFinding] = []
    for paraphrase_index, sentence in enumerate(forms):
        document_id = make_id(
            "doc",
            config.seed,
            link.kind,
            chain_index,
            link_index,
            paraphrase_index,
        )
        title = render_title(rng, link.subject, link.relation, link.target, link.system)
        filler = [
            render_filler(rng, link.system) for _ in range(config.filler_sentences)
        ]
        document, spans = assemble_document(
            rng,
            document_id,
            title,
            [sentence],
            filler,
            config.source_tag,
            (("kind", link.kind), ("chain", link.chain_id)),
        )
        start, end = spans[0]
        placed.append(
            PlacedFinding(
                link=link,
                document=document,
                citation=Citation(document_id, start, end, document.text[start:end]),
                sentence=sentence,
            )
        )
    return placed


def distractor_documents(config: SyntheticConfig, rng: Random) -> tuple[Document, ...]:
    """Create documents that mention none of the planted variables."""
    documents: list[Document] = []
    for index in range(config.distractor_documents):
        document_id = make_id("doc", config.seed, "distractor", index)
        batch = str(rng.randrange(1, 20))
        sentences = [
            render(rng.choice(DISTRACTOR_TEMPLATES), batch=batch) for _ in range(2)
        ]
        sentences.extend(
            render_filler(rng, rng.choice(MODEL_SYSTEMS))
            for _ in range(config.filler_sentences)
        )
        documents.append(
            Document(
                document_id,
                f"Cohort note {index + 1}",
                " ".join(sentences),
                config.source_tag,
                (("kind", "distractor"),),
            )
        )
    return tuple(documents)


COMPETING_OFFSET = 1000
CONTRADICTION_OFFSET = 2000


@dataclass(frozen=True)
class GeneratedCorpus:
    """Everything one generation produced, kept together for reproducibility."""

    config: SyntheticConfig
    corpus: Corpus
    chains: tuple[PlantedChain, ...]
    competing: tuple[PlantedLink, ...]
    contradictions: tuple[PlantedLink, ...]
    findings: tuple[PlacedFinding, ...]
    distractor_ids: tuple[str, ...]

    def findings_for(self, link: PlantedLink) -> tuple[PlacedFinding, ...]:
        """Return the placed findings that render one link."""
        return tuple(
            finding
            for finding in self.findings
            if finding.link.key() == link.key() and finding.link.kind == link.kind
        )


def generate(config: SyntheticConfig) -> GeneratedCorpus:
    """Generate a corpus with planted chains, rivals, negations and distractors.

    A single seeded generator drives every draw, in a fixed order, so the same
    configuration always yields byte-identical documents. The document count
    equals :meth:`SyntheticConfig.expected_documents`.
    """
    rng = config.rng()
    chains = plant_chains(config, rng)
    findings: list[PlacedFinding] = []
    documents: list[Document] = []
    competing: list[PlantedLink] = []
    contradictions: list[PlantedLink] = []

    def place(
        link: PlantedLink, forms: Sequence[str], chain_index: int, slot: int
    ) -> None:
        placed = finding_documents(config, rng, link, forms, chain_index, slot)
        findings.extend(placed)
        documents.extend(finding.document for finding in placed)

    for chain_index, chain in enumerate(chains):
        for link_index, link in enumerate(chain.links):
            forms = paraphrase_cluster(rng, link, config.paraphrases_per_link)
            place(link, forms, chain_index, link_index)
        rivals = competing_links(chain, rng, config.competitors_per_chain)
        competing.extend(rivals)
        for rival_index, rival in enumerate(rivals):
            place(
                rival,
                paraphrase_cluster(rng, rival, 1),
                chain_index,
                COMPETING_OFFSET + rival_index,
            )
        negations = contradiction_links(chain, rng, config.contradictions_per_chain)
        contradictions.extend(negations)
        for negation_index, negation in enumerate(negations):
            sentence = render_negation(
                rng, negation.subject, negation.target, negation.system
            )
            place(
                negation, [sentence], chain_index, CONTRADICTION_OFFSET + negation_index
            )

    distractors = distractor_documents(config, rng)
    documents.extend(distractors)
    return GeneratedCorpus(
        config=config,
        corpus=Corpus(documents),
        chains=chains,
        competing=tuple(competing),
        contradictions=tuple(contradictions),
        findings=tuple(findings),
        distractor_ids=tuple(document.document_id for document in distractors),
    )


def build_corpus(seed: int = 270106, **overrides: object) -> Corpus:
    """Convenience wrapper returning just the corpus for a seed."""
    config = SyntheticConfig(seed=seed, **overrides)  # type: ignore[arg-type]
    return generate(config).corpus


@dataclass(frozen=True)
class PlantedCluster:
    """The documents that restate one planted finding in different words.

    ``document_ids`` is sorted, so :meth:`pairs` enumerates a canonical set of
    document pairs that dedup measurements can be scored against.
    """

    link_key: tuple[str, str, str]
    statement: str
    document_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        """Number of documents in the cluster."""
        return len(self.document_ids)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """All unordered document pairs inside the cluster, sorted."""
        return tuple(
            (first, second)
            for index, first in enumerate(self.document_ids)
            for second in self.document_ids[index + 1 :]
        )


@dataclass(frozen=True)
class PlantedTruth:
    """Ground truth of a generated corpus, used to score recovery."""

    chains: tuple[PlantedChain, ...]
    competing: tuple[PlantedLink, ...]
    contradictions: tuple[PlantedLink, ...]
    clusters: tuple[PlantedCluster, ...]
    distractor_ids: tuple[str, ...]

    def true_links(self) -> tuple[PlantedLink, ...]:
        """Every planted link, in chain then position order."""
        return tuple(link for chain in self.chains for link in chain.links)

    def cluster_for(self, link_key: tuple[str, str, str]) -> PlantedCluster | None:
        """Return the paraphrase cluster of one link key."""
        for cluster in self.clusters:
            if cluster.link_key == link_key:
                return cluster
        return None

    def paraphrase_pairs(self) -> int:
        """Total number of same-finding document pairs a dedup pass should find."""
        return sum(len(cluster.pairs()) for cluster in self.clusters)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary for reports and artifacts."""
        return {
            "chains": len(self.chains),
            "true_links": len(self.true_links()),
            "competing": len(self.competing),
            "contradictions": len(self.contradictions),
            "clusters": len(self.clusters),
            "paraphrase_pairs": self.paraphrase_pairs(),
            "distractors": len(self.distractor_ids),
        }


def truth_of(generated: GeneratedCorpus) -> PlantedTruth:
    """Derive the ground truth record from a generated corpus."""
    clusters = tuple(
        PlantedCluster(
            link_key=link.key(),
            statement=link.statement,
            document_ids=tuple(
                sorted(finding.document_id for finding in generated.findings_for(link))
            ),
        )
        for chain in generated.chains
        for link in chain.links
    )
    return PlantedTruth(
        chains=generated.chains,
        competing=generated.competing,
        contradictions=generated.contradictions,
        clusters=clusters,
        distractor_ids=generated.distractor_ids,
    )


def gold_claims(generated: GeneratedCorpus, *, corpus_hash: str) -> tuple[Claim, ...]:
    """Build the gold claim set: one claim per planted link and per rival.

    Contradiction findings deliberately get no claim of their own — they become
    refuting evidence against the planted claim they deny. Every claim carries
    the exact citations of the sentences that state it, so a grounding verifier
    run over this set has a known-correct answer.
    """
    links = [link for chain in generated.chains for link in chain.links]
    links.extend(generated.competing)
    config = generated.config
    claims: list[Claim] = []
    for link in links:
        claims.append(
            Claim(
                claim_id=make_id(
                    "clm", config.seed, link.chain_id, link.key(), link.kind
                ),
                statement=link.statement,
                subject=link.subject,
                object=link.target,
                relation=link.relation,
                scope=Scope(population=link.system),
                citations=tuple(
                    finding.citation for finding in generated.findings_for(link)
                ),
                mechanism=None,
                provenance=Provenance(
                    origin="synthetic",
                    seed=config.seed,
                    corpus_hash=corpus_hash,
                    notes=link.kind,
                ),
            )
        )
    return tuple(claims)
