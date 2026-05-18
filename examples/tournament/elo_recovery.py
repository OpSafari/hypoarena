"""Show that an Elo tournament recovers a planted skill order (offline).

The qualities below are planted by this script, not learned: a deterministic
judge compares real gold claims whose strengths we chose, adding bounded noise.
The recovery numbers are a measured property of the rating mechanism on synthetic
input - they are not a claim about any language model's ability to rank real
hypotheses.

Run from the repository root:

    .venv/bin/python examples/tournament/elo_recovery.py
"""

from __future__ import annotations

from hypoarena.synthetic import SyntheticConfig, build_bundle
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    kendall_tau,
    order_recovery,
    transitivity_rate,
)

SUBJECT_COUNT = 5
QUALITIES = (0.95, 0.80, 0.60, 0.40, 0.20)


def main() -> int:
    """Rank planted-quality claims and report how well Elo recovers the order."""
    bundle = build_bundle(SyntheticConfig(seed=11, chains=3, chain_length=3))
    subjects = tuple(bundle.claims[:SUBJECT_COUNT])
    if len(subjects) < 2:
        print("not enough gold claims to run a tournament")
        return 1
    qualities = {
        claim.claim_id: quality
        for claim, quality in zip(subjects, QUALITIES, strict=False)
    }
    planted_order = tuple(claim.claim_id for claim in subjects)  # best -> worst
    judge = PlantedJudge(qualities, noise=0.1, seed=0)
    result = Tournament(judge, TournamentConfig(seed=0, repeats=4)).run(subjects)
    recovered = result.ranking()

    print("Elo tournament recovery (synthetic, planted qualities, offline)")
    print(
        f"  subjects={len(subjects)}  matches={len(result.matches)}  "
        f"repeats=4  noise=0.1"
    )
    print(f"  planted qualities (best->worst): {list(QUALITIES[: len(subjects)])}")
    print(f"  recovered order equals planted order: {recovered == planted_order}")
    print(
        f"  order_recovery={order_recovery(planted_order, recovered):.3f}  "
        f"kendall_tau={kendall_tau(planted_order, recovered):.3f}  "
        f"transitivity={transitivity_rate(result.matches):.3f}"
    )
    print(f"  top subject elo={result.standings()[0]['elo']}")
    print(
        "note: qualities were planted by this script; the numbers measure the "
        "rating mechanism, not any real model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
