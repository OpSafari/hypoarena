"""Determinism and measured convergence of the debate loop.

Every number here comes from running the loop with scripted agents over
synthetic corpora. They characterise the mechanism — how many rounds a fixed
point takes, how many calls a configuration implies — and say nothing about any
language model.
"""

from __future__ import annotations

import pytest

from hypoarena.agents import ScriptedAgent
from hypoarena.debate import (
    DebateConfig,
    DebateLoop,
    corpus_context,
)
from hypoarena.synthetic import SyntheticConfig, build_bundle

TIERS = (0.1, 0.5, 0.9)
CONTEXT = ("kinase K1 phosphorylates protein A", "protein A increases cell growth")


def loop_for(quality: float, config: DebateConfig | None = None) -> DebateLoop:
    return DebateLoop(
        ScriptedAgent("proposer", quality=quality),
        [
            ScriptedAgent("critic-0", quality=quality),
            ScriptedAgent("critic-1", quality=quality),
        ],
        config=config or DebateConfig(rounds=5, critics=2),
    )


def test_identical_setups_produce_identical_transcripts() -> None:
    first = loop_for(0.9).run(CONTEXT)
    second = loop_for(0.9).run(CONTEXT)
    assert first.transcript() == second.transcript()
    assert first.signature() == second.signature()
    assert first.usage == second.usage


def test_every_tier_reaches_a_fixed_point_in_two_rounds() -> None:
    for quality in TIERS:
        result = loop_for(quality).run(CONTEXT)
        assert result.converged is True
        assert result.rounds_run == 2
        assert result.turns[0].changed is True
        assert result.turns[1].changed is False


def test_final_statements_reflect_the_quality_tier() -> None:
    finals = {
        quality: loop_for(quality).run(CONTEXT).final_statement for quality in TIERS
    }
    assert finals[0.1].endswith("(unchanged)")
    assert "assayed" in finals[0.5]
    assert "dose response" in finals[0.9]
    lengths = [len(finals[quality]) for quality in TIERS]
    assert lengths == sorted(lengths)


def test_call_counts_follow_the_configured_pattern() -> None:
    for rounds in (1, 3):
        for critics in (1, 2):
            config = DebateConfig(
                rounds=rounds, critics=critics, stop_on_unchanged=False
            )
            result = loop_for(0.9, config).run(CONTEXT)
            expected = 1 + rounds * (critics + 1)
            total = result.usage["total"]
            assert isinstance(total, dict)
            assert total["calls"] == expected
            assert result.rounds_run == rounds


def test_context_is_recorded_verbatim() -> None:
    result = loop_for(0.5).run(CONTEXT)
    assert result.context == CONTEXT
    assert loop_for(0.5).run(()).context == ()


def test_corpus_seeds_change_proposals_but_stay_reproducible() -> None:
    first = build_bundle(SyntheticConfig(seed=11, chains=1, chain_length=2))
    second = build_bundle(SyntheticConfig(seed=12, chains=1, chain_length=2))
    context_a = corpus_context(first.corpus, limit=3)
    context_b = corpus_context(second.corpus, limit=3)
    assert context_a != context_b
    proposal_a = loop_for(0.9).run(context_a).proposal
    assert proposal_a == loop_for(0.9).run(context_a).proposal
    assert proposal_a != loop_for(0.9).run(context_b).proposal


@pytest.mark.parametrize("quality", TIERS)
def test_config_fingerprint_is_carried_into_the_result(quality: float) -> None:
    config = DebateConfig(rounds=2, critics=1)
    result = loop_for(quality, config).run(CONTEXT)
    assert result.config_fingerprint == config.fingerprint()
    assert len(result.config_fingerprint) == 16
