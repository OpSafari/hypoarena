"""A fixed seed yields byte-identical bundle serializations.

The synthetic factory and the serializers must be pure functions of the seed:
two builds of one configuration produce identical corpus, graph and truth lines.
That is what makes every downstream digest, report and resume stable.
"""

from __future__ import annotations

from hypoarena.serialize import corpus_to_lines, graph_to_lines, truth_to_lines
from hypoarena.synthetic import SyntheticConfig, build_bundle


def config() -> SyntheticConfig:
    return SyntheticConfig(seed=99, chains=3, chain_length=3)


def test_corpus_signature_is_stable() -> None:
    assert build_bundle(config()).corpus_hash == build_bundle(config()).corpus_hash


def test_serialized_corpus_is_identical_across_builds() -> None:
    first = corpus_to_lines(build_bundle(config()).corpus)
    second = corpus_to_lines(build_bundle(config()).corpus)
    assert first == second


def test_serialized_graph_and_truth_are_identical_across_builds() -> None:
    first = build_bundle(config())
    second = build_bundle(config())
    assert graph_to_lines(first.graph()) == graph_to_lines(second.graph())
    assert truth_to_lines(first.truth) == truth_to_lines(second.truth)


def test_a_different_seed_changes_the_serialization() -> None:
    other = SyntheticConfig(seed=100, chains=3, chain_length=3)
    assert corpus_to_lines(build_bundle(config()).corpus) != corpus_to_lines(
        build_bundle(other).corpus
    )
