"""Claim identifiers are derived from the link, not from lookup tables."""

from __future__ import annotations

from hypoarena.synthetic import (
    SyntheticConfig,
    build_bundle,
    claim_id_for,
    generate,
    gold_claims,
)


def test_claim_ids_match_the_gold_claims() -> None:
    generated = generate(SyntheticConfig(seed=71, chains=2, chain_length=3))
    produced = gold_claims(generated, corpus_hash=generated.corpus.signature())
    ids = {claim.claim_id for claim in produced}
    for link in [item for chain in generated.chains for item in chain.links]:
        assert claim_id_for(generated.config, link) in ids
    for rival in generated.competing:
        assert claim_id_for(generated.config, rival) in ids


def test_negated_findings_point_at_the_planted_claim() -> None:
    generated = generate(SyntheticConfig(seed=71, chains=2, chain_length=3))
    planted_ids = {
        claim_id_for(generated.config, link)
        for chain in generated.chains
        for link in chain.links
    }
    negated = [finding for finding in generated.findings if finding.is_negated]
    assert negated
    for finding in negated:
        assert finding.claim_id in planted_ids


def test_placed_findings_carry_the_owning_claim_id() -> None:
    generated = generate(SyntheticConfig(seed=71, chains=1, chain_length=2))
    for finding in generated.findings:
        assert finding.claim_id == claim_id_for(generated.config, finding.link)


def test_bundle_claims_are_all_referenced_by_some_finding() -> None:
    bundle = build_bundle(SyntheticConfig(seed=71, chains=2, chain_length=3))
    referenced = {finding.claim_id for finding in bundle.findings}
    assert {claim.claim_id for claim in bundle.claims} == referenced


def test_claim_ids_are_seed_dependent() -> None:
    first = generate(SyntheticConfig(seed=1, chains=1, chain_length=2))
    second = generate(SyntheticConfig(seed=2, chains=1, chain_length=2))
    assert claim_id_for(first.config, first.chains[0].links[0]) != claim_id_for(
        second.config, second.chains[0].links[0]
    )
