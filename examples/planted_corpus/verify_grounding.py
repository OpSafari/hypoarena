"""Generate a planted synthetic corpus and verify claim grounding (offline).

Fully offline and deterministic. The corpus is synthetic with planted causal
chains; the gold claims cite real spans by construction, so the grounding
verifier should accept them, while a claim whose citation is doctored to point
at text that is not in the document gets flagged. The numbers printed here
describe the mechanism on this synthetic corpus only - they are not a benchmark
of any real model and say nothing about scientific truth.

Run from the repository root:

    .venv/bin/python examples/planted_corpus/verify_grounding.py
"""

from __future__ import annotations

from dataclasses import replace

from hypoarena.grounding import (
    GroundingVerifier,
    VerifierConfig,
    summarize_reports,
)
from hypoarena.synthetic import SyntheticConfig, build_bundle


def main() -> int:
    """Build a planted corpus, verify its claims and probe a doctored citation."""
    bundle = build_bundle(SyntheticConfig(seed=7, chains=3, chain_length=3))
    graph = bundle.graph()
    verifier = GroundingVerifier(bundle.corpus, VerifierConfig())
    summary = summarize_reports(verifier.verify_graph(graph))

    print("planted-corpus grounding check (synthetic, offline, seed=7)")
    print(f"  documents={len(bundle.corpus)}  gold claims={len(bundle.claims)}")
    print(f"  claims graded={summary.total}")
    print(
        f"  grounded={summary.grounded}  weak={summary.weakly_grounded}  "
        f"ungrounded={summary.ungrounded}  fabricated={summary.fabricated}"
    )
    print(
        f"  grounded_rate={summary.grounded_rate:.3f}  "
        f"mean_score={summary.mean_score:.3f}"
    )

    gold = bundle.claims[0]
    # point the citation at a well-formed document id that is not in the corpus:
    # the claim stays structurally valid but its evidence cannot be resolved
    doctored = replace(
        gold,
        citations=(replace(gold.citations[0], document_id="doc_ffffffffffff"),),
    )
    gold_flag = verifier.verify(gold).flag.value
    doctored_flag = verifier.verify(doctored).flag.value
    print(f"  a gold claim is grounded as: {gold_flag}")
    print(f"  the same claim citing an absent document: {doctored_flag}")
    print(
        "note: synthetic corpus with planted ground truth; no real literature "
        "and no model benchmark."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
