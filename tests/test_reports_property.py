"""Property sweep: arbitrary payloads always render and keep the honesty block.

Feeding the renderers seeded, deliberately malformed report mappings checks two
invariants that must hold for *any* input: rendering never raises, and the
limitations block is always present. It also re-checks the HTML escaping
guarantee across many random payloads.
"""

from __future__ import annotations

import random
from typing import Any


def random_report(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)

    def maybe_map() -> Any:
        choice = rng.randint(0, 3)
        if choice == 0:
            return {}
        if choice == 1:
            return None
        if choice == 2:
            return "not-a-map"
        return {
            "total": rng.randint(0, 9),
            "grounded": rng.randint(0, 9),
            "grounded_rate": rng.random(),
            "mean_score": rng.random(),
        }

    def maybe_ranking() -> Any:
        if rng.random() < 0.3:
            return rng.choice([[], None, "junk"])
        return [
            {
                "position": index,
                "subject": f"s{index}",
                "elo": rng.random() * 100,
                "played": index,
                "wins": index,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.5,
            }
            for index in range(rng.randint(0, 3))
        ]

    return {
        "run_id": rng.choice(["a", "b", None, 123]),
        "counts": maybe_map(),
        "grounding": maybe_map(),
        "dedup": rng.choice([None, {"total": 1, "clusters": 1, "method": "exact"}]),
        "ranking": maybe_ranking(),
        "beliefs": rng.choice([[], [{"claim_id": "c", "posterior": 0.5}]]),
        "evolution": maybe_map(),
        "recovered": rng.choice(
            [
                {},
                {
                    "planted": 1,
                    "recovered": 1,
                    "rate": 1.0,
                    "links": [{"statement": "x|y<z>&w", "recovered": True}],
                },
            ]
        ),
        "cost": maybe_map(),
        "limitations": rng.choice([[], ["limit one", "two|three<four>"]]),
    }


def _render(report: dict[str, Any]) -> tuple[str, str]:
    from hypoarena.reports import render_html, render_markdown

    return render_markdown(report), render_html(report)


def test_random_payloads_render_without_raising() -> None:
    for seed in range(40):
        markdown, html = _render(random_report(seed))
        assert isinstance(markdown, str) and isinstance(html, str)


def test_limitations_header_survives_every_payload() -> None:
    for seed in range(40):
        markdown, html = _render(random_report(seed))
        assert "## Limitations" in markdown
        assert "Limitations" in html


def test_no_random_payload_leaks_a_raw_script_tag() -> None:
    for seed in range(40):
        _, html = _render(random_report(seed))
        assert "<script" not in html


def test_rendering_is_deterministic_per_payload() -> None:
    for seed in range(10):
        assert _render(random_report(seed)) == _render(random_report(seed))
