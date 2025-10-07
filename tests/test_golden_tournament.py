"""Golden standings for one fully specified tournament.

Configuration: four claims with planted qualities 0.95 / 0.70 / 0.45 / 0.20, a
noise-free planted judge, ``TournamentConfig(seed=0, repeats=1)`` and the default
Elo model (initial 1500, K 32 with 0.97 decay and a floor of 6, scale 400, draw
margin 0.05). Every number below was produced by running that tournament; they
are pinned so a change to the update rule, the schedule or the rounding is a
deliberate decision.
"""

from __future__ import annotations

import json

import pytest

from helpers import sample_claim
from hypoarena.serialize import tournament_to_lines
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    order_recovery,
    transitivity_rate,
)

SUBJECTS = (
    "clm_0123456789ab",
    "clm_111111111111",
    "clm_222222222222",
    "clm_ffffffffffff",
)
QUALITIES = {
    SUBJECTS[0]: 0.95,
    SUBJECTS[1]: 0.70,
    SUBJECTS[2]: 0.45,
    SUBJECTS[3]: 0.20,
}
GOLDEN_STANDINGS = [
    {
        "subject": "clm_0123456789ab",
        "elo": 1544.4877,
        "played": 3,
        "wins": 3,
        "losses": 0,
        "draws": 0,
        "win_rate": 1.0,
        "score_points": 3.0,
    },
    {
        "subject": "clm_111111111111",
        "elo": 1515.2968,
        "played": 3,
        "wins": 2,
        "losses": 1,
        "draws": 0,
        "win_rate": 0.6667,
        "score_points": 2.0,
    },
    {
        "subject": "clm_222222222222",
        "elo": 1485.3609,
        "played": 3,
        "wins": 1,
        "losses": 2,
        "draws": 0,
        "win_rate": 0.3333,
        "score_points": 1.0,
    },
    {
        "subject": "clm_ffffffffffff",
        "elo": 1454.8128,
        "played": 3,
        "wins": 0,
        "losses": 3,
        "draws": 0,
        "win_rate": 0.0,
        "score_points": 0.0,
    },
]
GOLDEN_SIGNATURE = "12d3d2414a233631"
GOLDEN_FINGERPRINT = "cb4ce249503577be"


def tournament() -> object:
    claims = [sample_claim(claim_id=subject) for subject in SUBJECTS]
    return Tournament(PlantedJudge(QUALITIES), TournamentConfig(seed=0, repeats=1)).run(
        claims
    )


def test_standings_match_the_golden_table() -> None:
    assert list(tournament().standings()) == [
        {"position": position, **row}
        for position, row in enumerate(GOLDEN_STANDINGS, start=1)
    ]


def test_signature_and_fingerprint_are_pinned() -> None:
    produced = tournament()
    assert produced.signature() == GOLDEN_SIGNATURE
    assert produced.config.fingerprint() == GOLDEN_FINGERPRINT


def test_the_planted_order_is_recovered_exactly() -> None:
    produced = tournament()
    assert produced.ranking() == SUBJECTS
    assert order_recovery(SUBJECTS, produced.ranking()) == 1.0
    assert transitivity_rate(produced.matches) == 1.0


def test_the_first_match_record_is_golden() -> None:
    produced = tournament()
    payload = json.loads(tournament_to_lines(produced)[2])
    assert payload["record"] == "rating"
    match = produced.matches[0].as_dict()
    assert match["left"] == "clm_0123456789ab"
    assert match["right"] == "clm_ffffffffffff"
    assert match["left_total"] == 0.95
    assert match["right_total"] == 0.2
    assert match["outcome"] == 1.0
    assert match["round_index"] == 0
    assert match["seed"] == 0


def test_total_rating_is_nearly_conserved() -> None:
    # Updates are zero-sum only while both sides have played the same number of
    # matches; the experience-dependent K factor leaves a small residual drift.
    produced = tournament()
    total = sum(rating.elo for rating in produced.ratings)
    assert total == pytest.approx(4 * 1500.0, abs=1.0)
    assert abs(total - 6000.0) < 0.1
