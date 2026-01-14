"""Debate records: validation, change detection and transcript lines."""

from __future__ import annotations

import pytest

from hypoarena.agents import AgentResponse
from hypoarena.debate import Critique, DebateTurn
from hypoarena.errors import ValidationError


def critique(agent: str = "critic", text: str = "names no assay") -> Critique:
    return Critique(agent=agent, text=text, request_id="req_0123456789ab")


def turn(**overrides: object) -> DebateTurn:
    payload: dict[str, object] = {
        "round_index": 0,
        "statement": "protein A increases cell growth",
        "critiques": (critique(),),
        "revised": "protein A increases cell growth in the assayed population",
    }
    payload.update(overrides)
    return DebateTurn(**payload)  # type: ignore[arg-type]


def test_critiques_reject_blank_content() -> None:
    with pytest.raises(ValidationError, match="name its agent"):
        Critique(agent=" ", text="x", request_id="r")
    with pytest.raises(ValidationError, match="blank"):
        Critique(agent="c", text="  ", request_id="r")


def test_critiques_can_be_built_from_responses() -> None:
    response = AgentResponse("req_ffffffffffff", "a verdict", "critic", 3, 2)
    built = Critique.from_response(response)
    assert built.agent == "critic"
    assert built.text == "a verdict"
    assert built.request_id == "req_ffffffffffff"


def test_turns_validate_index_and_statement() -> None:
    with pytest.raises(ValidationError, match="round_index"):
        turn(round_index=-1)
    with pytest.raises(ValidationError, match="statement"):
        turn(statement="   ")


def test_changed_ignores_case_and_whitespace_only_edits() -> None:
    assert turn().changed is True
    same = turn(revised="  Protein A   increases cell growth ")
    assert same.changed is False


def test_transcript_lists_the_round_critiques_and_revision() -> None:
    lines = turn().transcript()
    assert lines[0].startswith("round 0: protein A")
    assert lines[1] == "  critique[critic]: names no assay"
    assert lines[-1].startswith("  revised: ")
    assert len(lines) == 3


def test_transcript_grows_with_the_number_of_critics() -> None:
    two = turn(critiques=(critique("a"), critique("b", "scope too wide")))
    assert len(two.transcript()) == 4
    assert any("critique[b]: scope too wide" in line for line in two.transcript())
