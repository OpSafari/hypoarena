"""Evolution serialization: roundtrips, integrity and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.errors import SchemaError, ValidationError
from hypoarena.evolve import EvolutionEngine, Rejection
from hypoarena.serialize import (
    evolution_counts,
    evolution_from_lines,
    evolution_signature,
    evolution_step_from_dict,
    evolution_step_to_dict,
    evolution_to_lines,
    rejection_from_dict,
    rejection_to_dict,
)
from hypoarena.synthetic import SyntheticConfig, build_bundle


def steps() -> tuple[object, ...]:
    graph = build_bundle(SyntheticConfig(seed=31, chains=2, chain_length=3)).graph()
    return EvolutionEngine(seed=4).run(graph, generations=2)


def test_rejections_roundtrip() -> None:
    rejection = Rejection(
        "narrow_scope",
        ("clm_0123456789ab",),
        "not novel",
        statement="protein A increases cell growth",
        nearest="clm_ffffffffffff",
        similarity=0.95,
    )
    assert rejection_from_dict(rejection_to_dict(rejection)) == rejection


def test_steps_roundtrip_with_their_children() -> None:
    produced = steps()
    for step in produced:
        assert evolution_step_from_dict(evolution_step_to_dict(step)) == step


def test_step_dict_keys_are_golden() -> None:
    payload = evolution_step_to_dict(steps()[0])
    assert sorted(payload) == ["accepted", "generation", "rejected", "schema_version"]
    assert sorted(payload["accepted"][0]) == [
        "child",
        "operator",
        "parents",
        "rationale",
    ]


def test_lines_roundtrip_and_verify_the_header() -> None:
    produced = steps()
    lines = evolution_to_lines(produced)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "step", "step"]
    assert evolution_from_lines(lines) == produced
    meta = json.loads(lines[0])
    assert meta["counts"] == evolution_counts(produced)
    assert meta["signature"] == evolution_signature(produced)


def test_truncation_is_detected() -> None:
    produced = steps()
    lines = evolution_to_lines(produced)
    with pytest.raises(SchemaError, match="counts disagree"):
        evolution_from_lines(lines[:-1])


def test_unknown_records_and_keys_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown evolution record type"):
        evolution_from_lines(['{"record":"mutation","operator":"x"}'])
    payload = evolution_step_to_dict(steps()[0])
    payload["operator"] = "extra"
    with pytest.raises(SchemaError, match="unknown keys"):
        evolution_step_from_dict(payload)


def test_invalid_values_are_rejected_on_decode() -> None:
    payload = evolution_step_to_dict(steps()[0])
    payload["generation"] = -1
    with pytest.raises(SchemaError, match="below the minimum"):
        evolution_step_from_dict(payload)
    payload = evolution_step_to_dict(steps()[0])
    payload["accepted"][0]["operator"] = "teleport"
    with pytest.raises(ValidationError, match="unknown evolution operator"):
        evolution_step_from_dict(payload)
    payload = evolution_step_to_dict(steps()[0])
    payload["accepted"][0]["rationale"] = "  "
    with pytest.raises(ValidationError, match="rationale"):
        evolution_step_from_dict(payload)


def test_documents_without_a_header_still_rebuild() -> None:
    produced = steps()
    lines = evolution_to_lines(produced, include_meta=False)
    assert evolution_from_lines(lines) == produced
