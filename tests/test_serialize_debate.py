"""Debate transcript serialization: roundtrips, integrity and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.agents import ScriptedAgent
from hypoarena.debate import DebateConfig, DebateLoop, DebateResult
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.serialize import (
    critique_from_dict,
    critique_to_dict,
    debate_from_lines,
    debate_result_from_dict,
    debate_result_to_dict,
    debate_signature,
    debate_to_lines,
    turn_from_dict,
    turn_to_dict,
)

CONTEXT = ("kinase K1 phosphorylates protein A",)


def result(rounds: int = 2) -> DebateResult:
    loop = DebateLoop(
        ScriptedAgent("proposer", quality=0.9),
        [ScriptedAgent("critic", quality=0.5)],
        config=DebateConfig(rounds=rounds, critics=1),
    )
    return loop.run(CONTEXT)


def test_critiques_and_turns_roundtrip() -> None:
    produced = result(rounds=1)
    critique = produced.turns[0].critiques[0]
    assert critique_from_dict(critique_to_dict(critique)) == critique
    turn = produced.turns[0]
    assert turn_from_dict(turn_to_dict(turn)) == turn


def test_full_results_roundtrip_with_their_turns() -> None:
    produced = result()
    restored = debate_result_from_dict(debate_result_to_dict(produced))
    assert restored == produced
    assert len(restored.turns) == len(produced.turns)


def test_result_dict_keys_are_golden() -> None:
    payload = debate_result_to_dict(result(rounds=1))
    assert sorted(payload) == [
        "agents",
        "config_fingerprint",
        "context",
        "converged",
        "final_statement",
        "proposal",
        "rounds_run",
        "schema_version",
        "turns",
        "usage",
    ]
    assert sorted(payload["turns"][0]) == [
        "critiques",
        "revised",
        "round_index",
        "statement",
    ]


def test_lines_roundtrip_and_keep_the_record_order() -> None:
    produced = result()
    lines = debate_to_lines(produced)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds[0] == "meta"
    assert kinds[-1] == "result"
    assert kinds.count("turn") == len(produced.turns)
    assert debate_from_lines(lines) == produced


def test_header_counts_and_signature_are_verified() -> None:
    produced = result()
    lines = debate_to_lines(produced)
    meta = json.loads(lines[0])
    assert meta["counts"] == {"turns": len(produced.turns)}
    assert meta["signature"] == debate_signature(produced)
    with pytest.raises(SchemaError, match="counts disagree"):
        debate_from_lines([lines[0], *lines[2:]])


def test_a_document_without_a_result_line_is_rejected() -> None:
    produced = result(rounds=1)
    lines = debate_to_lines(produced)
    with pytest.raises(SchemaError, match="no result line"):
        debate_from_lines(lines[:-1], verify_meta=False)


def test_unknown_records_keys_and_values_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown debate record type"):
        debate_from_lines(['{"record":"verdict","text":"x"}'])
    payload = debate_result_to_dict(result(rounds=1))
    payload["judge"] = "anon"
    with pytest.raises(SchemaError, match="unknown keys"):
        debate_result_from_dict(payload)
    payload = debate_result_to_dict(result(rounds=1))
    payload["converged"] = "yes"
    with pytest.raises(SchemaError, match="boolean"):
        debate_result_from_dict(payload)
    payload = debate_result_to_dict(result(rounds=1))
    payload["turns"][0]["round_index"] = -1
    with pytest.raises(SchemaError, match="below the minimum"):
        debate_result_from_dict(payload)


def test_blank_revisions_survive_a_roundtrip() -> None:
    produced = result(rounds=1)
    payload = debate_result_to_dict(produced)
    payload["turns"][0]["revised"] = ""
    restored = debate_result_from_dict(payload)
    assert restored.turns[0].revised == ""


def test_critique_validation_still_applies_after_decoding() -> None:
    payload = critique_to_dict(result(rounds=1).turns[0].critiques[0])
    payload["text"] = "  "
    with pytest.raises(ValidationError, match="blank"):
        critique_from_dict(payload)
