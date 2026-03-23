"""A canonical, fully-populated run report used across the report tests.

The values are synthetic and hand-picked so renderings are stable and easy to
read; nothing here is measured from a real run. Tests get a fresh deep copy so
one test cannot mutate another's fixture.
"""

from __future__ import annotations

import copy
from typing import Any

_SAMPLE: dict[str, Any] = {
    "run_id": "demo",
    "config_fingerprint": "abc123",
    "corpus_hash": "0123456789abcdef",
    "counts": {"documents": 6, "claims": 4, "evidence": 8, "links": 4, "edges": 3},
    "grounding": {
        "total": 4,
        "grounded": 3,
        "weakly_grounded": 1,
        "ungrounded": 0,
        "fabricated": 0,
        "grounded_rate": 0.75,
        "mean_score": 0.8125,
    },
    "dedup": {
        "total": 4,
        "clusters": 1,
        "duplicated": 2,
        "duplicate_rate": 0.5,
        "method": "minhash",
        "threshold": 0.8,
    },
    "ranking": [
        {
            "position": 1,
            "subject": "agt_a",
            "elo": 1560.0,
            "played": 3,
            "wins": 3,
            "losses": 0,
            "draws": 0,
            "win_rate": 1.0,
            "score_points": 3.0,
        },
        {
            "position": 2,
            "subject": "agt_b",
            "elo": 1440.0,
            "played": 3,
            "wins": 0,
            "losses": 3,
            "draws": 0,
            "win_rate": 0.0,
            "score_points": 0.0,
        },
    ],
    "beliefs": [
        {"claim_id": "clm_a", "posterior": 0.9},
        {"claim_id": "clm_b", "posterior": 0.25},
    ],
    "evolution": {"generations": 2, "accepted": 3, "rejected": 1},
    "recovered": {
        "planted": 2,
        "recovered": 2,
        "rate": 1.0,
        "links": [
            {"statement": "A causes B", "recovered": True},
            {"statement": "B causes C", "recovered": True},
        ],
    },
    "cost": {
        "entries": 4,
        "calls": 12,
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "total_tokens": 1500,
    },
    "limitations": [
        "Synthetic corpus; no real literature was read.",
        "Rankings reflect the judge rubric, not scientific truth.",
    ],
}


def sample_report(**overrides: Any) -> dict[str, Any]:
    """Return a fresh copy of the canonical report, with optional overrides."""
    report = copy.deepcopy(_SAMPLE)
    report.update(overrides)
    return report
