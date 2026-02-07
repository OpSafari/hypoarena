"""Pairing schedules: coverage, determinism and side balancing."""

from __future__ import annotations

from collections import Counter

import pytest

from hypoarena.errors import ValidationError
from hypoarena.tournament import pair_count, round_robin_pairings

SUBJECTS = (
    "clm_0123456789ab",
    "clm_111111111111",
    "clm_ffffffffffff",
    "clm_222222222222",
)


def test_one_round_covers_every_pair_exactly_once() -> None:
    schedule = round_robin_pairings(SUBJECTS)
    assert len(schedule) == pair_count(SUBJECTS) == 6
    pairs = Counter(tuple(sorted(pair)) for pair in schedule)
    assert set(pairs.values()) == {1}


def test_repeats_multiply_the_schedule_and_stay_balanced() -> None:
    schedule = round_robin_pairings(SUBJECTS, repeats=3)
    assert len(schedule) == 18
    pairs = Counter(tuple(sorted(pair)) for pair in schedule)
    assert set(pairs.values()) == {3}
    appearances = Counter(subject for pair in schedule for subject in pair)
    assert set(appearances.values()) == {9}


def test_schedules_are_deterministic_for_a_seed() -> None:
    first = round_robin_pairings(SUBJECTS, repeats=2, seed=7)
    second = round_robin_pairings(SUBJECTS, repeats=2, seed=7)
    assert first == second


def test_different_seeds_reorder_without_changing_coverage() -> None:
    first = round_robin_pairings(SUBJECTS, repeats=2, seed=1)
    second = round_robin_pairings(SUBJECTS, repeats=2, seed=2)
    assert first != second
    assert Counter(tuple(sorted(p)) for p in first) == Counter(
        tuple(sorted(p)) for p in second
    )


def test_odd_repeats_swap_the_sides() -> None:
    schedule = round_robin_pairings(SUBJECTS, repeats=2, seed=3)
    first_round = schedule[: pair_count(SUBJECTS)]
    second_round = schedule[pair_count(SUBJECTS) :]
    assert Counter(first_round) != Counter(second_round)
    assert Counter(tuple(sorted(p)) for p in first_round) == Counter(
        tuple(sorted(p)) for p in second_round
    )


def test_input_order_and_duplicates_do_not_matter() -> None:
    shuffled = list(reversed(SUBJECTS)) + [SUBJECTS[0]]
    assert round_robin_pairings(shuffled) == round_robin_pairings(SUBJECTS)


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least two subjects"):
        round_robin_pairings(("clm_0123456789ab",))
    with pytest.raises(ValidationError, match="at least two subjects"):
        round_robin_pairings([])
    with pytest.raises(ValidationError, match="repeats"):
        round_robin_pairings(SUBJECTS, repeats=0)


def test_no_subject_is_paired_with_itself() -> None:
    for pair in round_robin_pairings(SUBJECTS, repeats=3, seed=5):
        assert pair[0] != pair[1]
