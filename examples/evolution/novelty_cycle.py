"""Evolve a hypothesis graph and measure novelty via dedup (offline).

The evolution engine applies scope-narrowing, substitution, crossover and
decomposition operators, each gated by a novelty check; the dedup finder then
clusters near-duplicate statements before and after. The growth and novelty
numbers below are measured on a synthetic planted graph - they characterise the
operators, not any real discovery process.

Run from the repository root:

    .venv/bin/python examples/evolution/novelty_cycle.py
"""

from __future__ import annotations

from hypoarena.dedup import DedupConfig, DuplicateFinder
from hypoarena.evolve import EvolutionConfig
from hypoarena.synthetic import SyntheticConfig, build_bundle


def main() -> int:
    """Grow a planted graph with the evolution operators and measure novelty."""
    bundle = build_bundle(SyntheticConfig(seed=5, chains=3, chain_length=3))
    graph = bundle.graph()
    before = len(graph.claims)
    finder = DuplicateFinder(DedupConfig())
    dedup_before = finder.report({c.claim_id: c.statement for c in graph.claims})

    config = EvolutionConfig(generations=3)
    steps = config.engine().run(graph, config.generations)
    accepted = sum(step.accepted_count for step in steps)
    rejected = sum(len(step.rejected) for step in steps)
    after = len(graph.claims)
    dedup_after = finder.report({c.claim_id: c.statement for c in graph.claims})

    print("evolution + dedup novelty cycle (synthetic, offline, seed=5)")
    print(f"  claims before={before}  after={after}  grew={after - before}")
    print(
        f"  generations={len(steps)}  accepted={accepted}  "
        f"rejected(novelty/invalid)={rejected}"
    )
    print(
        f"  duplicate clusters before={len(dedup_before.clusters)}  "
        f"after={len(dedup_after.clusters)}"
    )
    print(
        f"  duplicate_rate before={dedup_before.duplicate_rate:.3f}  "
        f"after={dedup_after.duplicate_rate:.3f}"
    )
    print(
        "note: numbers describe the operators on a synthetic planted graph only; "
        "no claim about real discovery."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
