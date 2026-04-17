"""Golden over the synthetic bundle summary for a fixed configuration.

``bundle.summary()`` is the count vector every report and CLI line draws on. For
a fixed seed and chain shape the generator is deterministic, so the exact counts
are a stable contract; pinning them turns any change to the planting logic into a
deliberate, reviewed edit.
"""

from __future__ import annotations

from hypoarena.synthetic import SyntheticConfig, build_bundle

GOLDEN = {
    "documents": 22,
    "claims": 9,
    "evidence": 18,
    "planted_links": 6,
    "competing": 3,
    "contradictions": 3,
    "paraphrase_pairs": 6,
    "distractors": 4,
}


def test_bundle_summary_matches_the_golden() -> None:
    bundle = build_bundle(SyntheticConfig(seed=7, chains=3, chain_length=3))
    assert bundle.summary() == GOLDEN


def test_bundle_summary_is_deterministic() -> None:
    config = SyntheticConfig(seed=7, chains=3, chain_length=3)
    assert build_bundle(config).summary() == build_bundle(config).summary()


def test_summary_counts_agree_with_the_records() -> None:
    bundle = build_bundle(SyntheticConfig(seed=7, chains=3, chain_length=3))
    summary = bundle.summary()
    assert summary["documents"] == len(bundle.corpus)
    assert summary["claims"] == len(bundle.claims)
    assert summary["evidence"] == len(bundle.evidence)
    assert summary["planted_links"] == len(bundle.truth.true_links())
