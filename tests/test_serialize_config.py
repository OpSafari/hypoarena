"""Run configuration serialization: roundtrips and strict rejection."""

from __future__ import annotations

import pytest

from hypoarena.belief import BeliefConfig, ContradictionPolicy
from hypoarena.config import STAGES, RunConfig
from hypoarena.debate import DebateConfig
from hypoarena.dedup import DedupConfig
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.evolve import EvolutionConfig
from hypoarena.grounding import VerifierConfig
from hypoarena.serialize import (
    debate_config_from_dict,
    debate_config_to_dict,
    evolution_config_from_dict,
    evolution_config_to_dict,
    run_config_from_dict,
    run_config_to_dict,
    tournament_config_from_dict,
    tournament_config_to_dict,
    verifier_config_from_dict,
    verifier_config_to_dict,
)
from hypoarena.synthetic import SyntheticConfig
from hypoarena.tournament import TournamentConfig


def test_sub_configs_roundtrip() -> None:
    verifier = VerifierConfig(min_entity_overlap=0.5, min_quote_length=20)
    assert verifier_config_from_dict(verifier_config_to_dict(verifier)) == verifier
    debate = DebateConfig(rounds=3, critics=1, stop_on_unchanged=False)
    assert debate_config_from_dict(debate_config_to_dict(debate)) == debate
    tournament = TournamentConfig(seed=4, repeats=3)
    assert tournament_config_from_dict(tournament_config_to_dict(tournament)) == (
        tournament
    )
    evolution = EvolutionConfig(seed=2, operators=("narrow_scope",), generations=4)
    assert evolution_config_from_dict(evolution_config_to_dict(evolution)) == evolution


def test_a_full_run_config_roundtrips() -> None:
    config = RunConfig(
        seed=5,
        stages=("corpus", "generate", "verify", "report"),
        run_id="demo",
        corpus=SyntheticConfig(seed=5, chains=1, chain_length=2),
        grounding=VerifierConfig(min_quote_length=8),
        dedup=DedupConfig(method="jaccard", threshold=0.7),
        debate=DebateConfig(rounds=2, critics=1),
        tournament=TournamentConfig(seed=5, repeats=2),
        belief=BeliefConfig(prior=0.3, contradiction_policy=ContradictionPolicy.REJECT),
        evolution=EvolutionConfig(seed=5, generations=2),
    )
    assert run_config_from_dict(run_config_to_dict(config)) == config


def test_defaults_roundtrip_and_keep_the_stage_order() -> None:
    config = RunConfig()
    restored = run_config_from_dict(run_config_to_dict(config))
    assert restored == config
    assert restored.stages == STAGES


def test_config_dict_keys_are_golden() -> None:
    payload = run_config_to_dict(RunConfig())
    assert sorted(payload) == [
        "belief",
        "corpus",
        "debate",
        "dedup",
        "evolution",
        "grounding",
        "run_id",
        "seed",
        "stages",
        "tournament",
    ]


def test_unknown_keys_and_bad_values_are_rejected() -> None:
    payload = run_config_to_dict(RunConfig())
    payload["owner"] = "someone"
    with pytest.raises(SchemaError, match="unknown keys"):
        run_config_from_dict(payload)
    payload = run_config_to_dict(RunConfig())
    payload["stages"] = ["corpus", "publish"]
    with pytest.raises(ValidationError, match="unknown pipeline stages"):
        run_config_from_dict(payload)
    payload = run_config_to_dict(RunConfig())
    payload["grounding"]["min_quote_length"] = 0
    with pytest.raises(SchemaError, match="below the minimum"):
        run_config_from_dict(payload)
    payload = run_config_to_dict(RunConfig())
    payload["belief"]["contradiction_policy"] = "shrug"
    with pytest.raises(SchemaError, match="not a valid ContradictionPolicy"):
        run_config_from_dict(payload)
