"""Tournament serialization: roundtrips, audit replay and rejections."""

from __future__ import annotations

import json

import pytest

from helpers import sample_claim
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.serialize import (
    match_from_dict,
    match_to_dict,
    rating_from_dict,
    rating_to_dict,
    rubric_score_from_dict,
    rubric_score_to_dict,
    tournament_from_dict,
    tournament_from_lines,
    tournament_signature,
    tournament_to_dict,
    tournament_to_lines,
)
from hypoarena.tournament import (
    PlantedJudge,
    Tournament,
    TournamentConfig,
    replay_ratings,
)

SUBJECTS = ("clm_0123456789ab", "clm_111111111111", "clm_222222222222")
QUALITIES = {SUBJECTS[0]: 0.9, SUBJECTS[1]: 0.5, SUBJECTS[2]: 0.1}


def result() -> object:
    claims = [sample_claim(claim_id=subject) for subject in SUBJECTS]
    return Tournament(PlantedJudge(QUALITIES), TournamentConfig(seed=3, repeats=2)).run(
        claims
    )


def test_scores_and_ratings_roundtrip() -> None:
    produced = result()
    score = produced.matches[0].left_score
    assert rubric_score_from_dict(rubric_score_to_dict(score)) == score
    rating = produced.ratings[0]
    assert rating_from_dict(rating_to_dict(rating)) == rating


def test_matches_roundtrip_with_their_audit_fields() -> None:
    produced = result()
    for match in produced.matches:
        assert match_from_dict(match_to_dict(match)) == match


def test_full_results_roundtrip() -> None:
    produced = result()
    restored = tournament_from_dict(tournament_to_dict(produced))
    assert restored.ratings == produced.ratings
    assert restored.matches == produced.matches
    assert restored.config == produced.config
    assert restored.subjects == produced.subjects


def test_lines_roundtrip_and_verify_the_header() -> None:
    produced = result()
    lines = tournament_to_lines(produced)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds[0] == "meta"
    assert kinds[1] == "config"
    assert kinds.count("rating") == len(produced.ratings)
    assert kinds.count("match") == len(produced.matches)
    restored = tournament_from_lines(lines)
    assert restored.ratings == produced.ratings
    assert tournament_signature(restored) == json.loads(lines[0])["signature"]


def test_a_serialized_trail_still_replays_to_the_same_standings() -> None:
    produced = result()
    restored = tournament_from_lines(tournament_to_lines(produced))
    assert replay_ratings(restored.matches, restored.config.model) == restored.ratings


def test_truncation_and_unknown_records_are_rejected() -> None:
    produced = result()
    lines = tournament_to_lines(produced)
    with pytest.raises(SchemaError, match="counts disagree"):
        tournament_from_lines(lines[:-1])
    with pytest.raises(SchemaError, match="unknown tournament record type"):
        tournament_from_lines(['{"record":"verdict","winner":"x"}'])
    with pytest.raises(SchemaError, match="no config line"):
        tournament_from_lines([lines[0], *lines[2:]])


def test_invalid_values_are_rejected_on_decode() -> None:
    payload = tournament_to_dict(result())
    payload["ratings"][0]["wins"] = 99
    with pytest.raises(ValidationError, match="tally disagrees"):
        tournament_from_dict(payload)
    payload = tournament_to_dict(result())
    payload["matches"][0]["outcome"] = 0.75
    with pytest.raises(ValidationError, match="outcome"):
        tournament_from_dict(payload)
    payload = tournament_to_dict(result())
    payload["matches"][0]["left_score"]["novelty"] = 2.0
    with pytest.raises(SchemaError, match="above the maximum"):
        tournament_from_dict(payload)
    payload = tournament_to_dict(result())
    payload["surprise"] = 1
    with pytest.raises(SchemaError, match="unknown keys"):
        tournament_from_dict(payload)
