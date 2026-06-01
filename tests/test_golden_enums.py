"""Exhaustive golden over every public enum: exact members and value round-trip.

Enums are part of the serialized contract (their values land in JSONL), so a
member added, removed, renamed or reordered must be a deliberate change that
updates this file - never silent drift.
"""

from __future__ import annotations

from enum import Enum

import pytest

from hypoarena.belief import ContradictionPolicy
from hypoarena.grounding import GroundingFlag
from hypoarena.schema import ClaimRelation, EvidencePolarity, PredictedRelation

ENUMS: dict[type[Enum], tuple[str, ...]] = {
    PredictedRelation: (
        "increases",
        "decreases",
        "enables",
        "inhibits",
        "causes",
        "associates",
    ),
    EvidencePolarity: ("support", "refute", "neutral"),
    ClaimRelation: ("entails", "contradicts", "refines"),
    GroundingFlag: ("grounded", "weakly_grounded", "ungrounded", "fabricated"),
    ContradictionPolicy: ("ignore", "downweight", "discount", "reject"),
}


@pytest.mark.parametrize("enum", list(ENUMS), ids=lambda e: e.__name__)
def test_enum_values_match_the_golden_snapshot(enum: type[Enum]) -> None:
    assert tuple(member.value for member in enum) == ENUMS[enum]


@pytest.mark.parametrize("enum", list(ENUMS), ids=lambda e: e.__name__)
def test_every_member_round_trips_by_value(enum: type[Enum]) -> None:
    for member in enum:
        assert enum(member.value) is member


@pytest.mark.parametrize("enum", list(ENUMS), ids=lambda e: e.__name__)
def test_member_count_is_pinned(enum: type[Enum]) -> None:
    assert len(list(enum)) == len(ENUMS[enum])


def test_all_enums_are_string_valued() -> None:
    for enum in ENUMS:
        for member in enum:
            assert isinstance(member.value, str)
