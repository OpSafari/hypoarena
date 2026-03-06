"""Belief serialization: roundtrips, self-describing config and integrity."""

from __future__ import annotations

import json

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    BeliefConfig,
    ContradictionPolicy,
    LikelihoodModel,
    accumulate,
)
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.schema import EvidencePolarity
from hypoarena.serialize import (
    belief_config_from_dict,
    belief_config_to_dict,
    belief_from_lines,
    belief_signature,
    belief_state_from_dict,
    belief_state_to_dict,
    belief_to_lines,
    likelihood_model_from_dict,
    likelihood_model_to_dict,
)

CLAIM = "clm_0123456789ab"


def states(config: BeliefConfig | None = None) -> list[object]:
    settings = config or BeliefConfig()
    return [
        accumulate(
            CLAIM,
            [
                sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=1.0),
                sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=0.5),
            ],
            settings,
        ),
        accumulate("clm_ffffffffffff", [], settings),
    ]


def test_models_and_configs_roundtrip() -> None:
    model = LikelihoodModel(support_ratio=4.0, refute_ratio=2.0, neutral_ratio=1.0)
    assert likelihood_model_from_dict(likelihood_model_to_dict(model)) == model
    config = BeliefConfig(
        prior=0.3,
        likelihood=model,
        contradiction_policy=ContradictionPolicy.REJECT,
        downweight_factor=0.25,
        contradiction_threshold=3,
    )
    assert belief_config_from_dict(belief_config_to_dict(config)) == config


def test_states_roundtrip_without_losing_precision() -> None:
    for state in states():
        restored = belief_state_from_dict(belief_state_to_dict(state))
        assert restored == state
        assert restored.posterior == state.posterior


def test_state_dict_keys_are_golden() -> None:
    payload = belief_state_to_dict(states()[0])
    assert sorted(payload) == [
        "claim_id",
        "likelihood_ratio",
        "neutral",
        "posterior",
        "prior",
        "refuting",
        "supporting",
    ]


def test_lines_carry_the_configuration_and_verify_the_header() -> None:
    config = BeliefConfig(prior=0.4)
    produced = states(config)
    lines = belief_to_lines(produced, config)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "config", "state", "state"]
    restored, restored_config = belief_from_lines(lines)
    assert restored == tuple(produced)
    assert restored_config == config
    assert json.loads(lines[0])["signature"] == belief_signature(produced, config)


def test_truncation_is_detected() -> None:
    config = BeliefConfig()
    lines = belief_to_lines(states(config), config)
    with pytest.raises(SchemaError, match="counts disagree"):
        belief_from_lines(lines[:-1])


def test_unknown_records_and_invalid_values_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown belief record type"):
        belief_from_lines(['{"record":"posterior","value":0.5}'])
    payload = belief_state_to_dict(states()[0])
    payload["posterior"] = 1.5
    with pytest.raises(SchemaError, match="above the maximum"):
        belief_state_from_dict(payload)
    config_payload = belief_config_to_dict(BeliefConfig())
    config_payload["contradiction_policy"] = "shrug"
    with pytest.raises(SchemaError, match="not a valid ContradictionPolicy"):
        belief_config_from_dict(config_payload)
    config_payload = belief_config_to_dict(BeliefConfig())
    config_payload["contradiction_threshold"] = 0
    with pytest.raises(ValidationError, match="contradiction_threshold"):
        belief_config_from_dict(config_payload)
