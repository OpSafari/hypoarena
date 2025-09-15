"""Structural sweep over debate configurations.

The invariants checked here are the ones reports and the runner rely on: the
transcript layout is predictable from the configuration, usage accounting matches
the number of agent calls, and nothing depends on critic ordering.
"""

from __future__ import annotations

import pytest

from hypoarena.agents import ScriptedAgent
from hypoarena.debate import DebateConfig, DebateLoop

CONTEXT = ("kinase K1 phosphorylates protein A",)
ROUND_COUNTS = (1, 2, 3)
CRITIC_COUNTS = (1, 2, 3)


def run_debate(rounds: int, critics: int) -> object:
    loop = DebateLoop(
        ScriptedAgent("proposer", quality=0.9),
        [ScriptedAgent(f"critic-{index}", quality=0.5) for index in range(critics)],
        config=DebateConfig(rounds=rounds, critics=critics, stop_on_unchanged=False),
    )
    return loop.run(CONTEXT)


@pytest.mark.parametrize("rounds", ROUND_COUNTS)
@pytest.mark.parametrize("critics", CRITIC_COUNTS)
def test_the_round_budget_is_used_exactly(rounds: int, critics: int) -> None:
    result = run_debate(rounds, critics)
    assert result.rounds_run == rounds
    assert [turn.round_index for turn in result.turns] == list(range(rounds))
    assert all(len(turn.critiques) == critics for turn in result.turns)


@pytest.mark.parametrize("rounds", ROUND_COUNTS)
@pytest.mark.parametrize("critics", CRITIC_COUNTS)
def test_transcript_layout_follows_the_configuration(rounds: int, critics: int) -> None:
    result = run_debate(rounds, critics)
    expected = 1 + rounds * (2 + critics) + 2
    assert len(result.transcript()) == expected


@pytest.mark.parametrize("critics", CRITIC_COUNTS)
def test_usage_matches_the_number_of_calls(critics: int) -> None:
    result = run_debate(2, critics)
    total = result.usage["total"]
    assert isinstance(total, dict)
    assert total["calls"] == 1 + 2 * (critics + 1)
    agents = result.usage["agents"]
    assert isinstance(agents, dict)
    assert len(agents) == 1 + critics


def test_extra_critics_beyond_the_config_are_not_used() -> None:
    loop = DebateLoop(
        ScriptedAgent("proposer", quality=0.9),
        [ScriptedAgent(f"critic-{index}", quality=0.5) for index in range(4)],
        config=DebateConfig(rounds=1, critics=2),
    )
    result = loop.run(CONTEXT)
    assert len(result.turns[0].critiques) == 2
    assert result.agents == ("proposer", "critic-0", "critic-1")


def test_runs_are_reproducible_across_the_whole_sweep() -> None:
    for rounds in ROUND_COUNTS:
        for critics in CRITIC_COUNTS:
            first = run_debate(rounds, critics)
            second = run_debate(rounds, critics)
            assert first.signature() == second.signature()
