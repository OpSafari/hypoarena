"""Gold claims: identifiers, citations and provenance."""

from __future__ import annotations

from hypoarena.schema import Claim
from hypoarena.synthetic import (
    GeneratedCorpus,
    SyntheticConfig,
    generate,
    gold_claims,
)


def generated() -> GeneratedCorpus:
    return generate(SyntheticConfig(seed=31, chains=2, chain_length=3))


def test_gold_claims_cover_planted_links_and_rivals() -> None:
    bundle = generated()
    produced = gold_claims(bundle, corpus_hash=bundle.corpus.signature())
    planted = sum(len(chain.links) for chain in bundle.chains)
    assert len(produced) == planted + len(bundle.competing)
    assert len({claim.claim_id for claim in produced}) == len(produced)


def test_every_gold_citation_resolves_in_the_corpus() -> None:
    bundle = generated()
    for claim in gold_claims(bundle, corpus_hash=bundle.corpus.signature()):
        assert claim.is_cited is True
        for citation in claim.citations:
            assert bundle.corpus.resolve(citation) == citation.quote


def test_gold_claims_use_the_canonical_statement() -> None:
    bundle = generated()
    produced = gold_claims(bundle, corpus_hash=bundle.corpus.signature())
    link = bundle.chains[0].links[0]
    matching = [claim for claim in produced if claim.statement == link.statement]
    assert matching
    assert matching[0].subject == link.subject
    assert matching[0].relation is link.relation
    assert matching[0].scope.population == link.system


def test_provenance_records_seed_corpus_hash_and_role() -> None:
    bundle = generated()
    digest = bundle.corpus.signature()
    produced = gold_claims(bundle, corpus_hash=digest)
    roles = {claim.provenance.notes for claim in produced}
    assert roles == {"planted", "competing"}
    for claim in produced:
        assert claim.provenance.origin == "synthetic"
        assert claim.provenance.seed == bundle.config.seed
        assert claim.provenance.corpus_hash == digest
        assert claim.provenance.is_reproducible is True


def test_rival_claims_disagree_with_the_planted_relation() -> None:
    bundle = generated()
    produced = gold_claims(bundle, corpus_hash=bundle.corpus.signature())
    by_pair: dict[tuple[str, str], list[Claim]] = {}
    for claim in produced:
        by_pair.setdefault((claim.subject, claim.object), []).append(claim)
    rivals = [group for group in by_pair.values() if len(group) > 1]
    assert rivals
    for group in rivals:
        assert len({claim.relation for claim in group}) == len(group)
