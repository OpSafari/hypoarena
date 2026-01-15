"""The debate loop: roles, rounds, convergence and accounting."""

from __future__ import annotations

import pytest

from hypoarena.agents import ReplayAgent, ReplayEntry, ScriptedAgent
from hypoarena.debate import DebateConfig, DebateLoop, DebateResult
from hypoarena.errors import ValidationError

CONTEXT = ("kinase K1 phosphorylates protein A",)


def loop(quality: float = 0.9, config: DebateConfig | None = None) -> DebateLoop:
    proposer = ScriptedAgent("proposer", quality=quality)
    critics = [ScriptedAgent(f"critic-{index}", quality=quality) for index in range(2)]
    return DebateLoop(proposer, critics, config=config or DebateConfig(critics=2))


def test_a_run_returns_a_full_result() -> None:
    result = loop().run(CONTEXT)
    assert isinstance(result, DebateResult)
    assert result.proposal
    assert result.final_statement
    assert result.agents == ("proposer", "critic-0", "critic-1")
    assert result.rounds_run >= 1


def test_round_budget_is_respected() -> None:
    config = DebateConfig(rounds=3, critics=1, stop_on_unchanged=False)
    result = loop(config=config).run(CONTEXT)
    assert result.rounds_run == 3
    assert [turn.round_index for turn in result.turns] == [0, 1, 2]


def test_convergence_stops_the_loop_early() -> None:
    config = DebateConfig(rounds=5, critics=1)
    result = loop(quality=0.9, config=config).run(CONTEXT)
    assert result.converged is True
    assert result.rounds_run == 2
    assert result.turns[-1].changed is False


def test_convergence_can_be_disabled() -> None:
    config = DebateConfig(rounds=4, critics=1, stop_on_unchanged=False)
    result = loop(config=config).run(CONTEXT)
    assert result.rounds_run == 4
    assert result.converged is True


def test_each_round_collects_one_critique_per_critic() -> None:
    result = loop(config=DebateConfig(rounds=1, critics=2)).run(CONTEXT)
    assert len(result.turns[0].critiques) == 2
    assert {critique.agent for critique in result.turns[0].critiques} == {
        "critic-0",
        "critic-1",
    }


def test_the_reviser_can_be_a_separate_agent() -> None:
    proposer = ScriptedAgent("proposer", quality=0.2)
    reviser = ScriptedAgent("reviser", quality=0.9)
    debate = DebateLoop(
        proposer,
        [ScriptedAgent("critic", quality=0.5)],
        reviser,
        DebateConfig(critics=1),
    )
    result = debate.run(CONTEXT)
    assert result.agents == ("proposer", "critic", "reviser")
    assert "dose response" in result.final_statement


def test_usage_is_summarized_per_agent_and_in_total() -> None:
    result = loop(config=DebateConfig(rounds=1, critics=2)).run(CONTEXT)
    agents = result.usage["agents"]
    assert isinstance(agents, dict)
    assert set(agents) == {"proposer", "critic-0", "critic-1"}
    total = result.usage["total"]
    assert isinstance(total, dict)
    assert total["calls"] == 4  # one proposal, two critiques, one revision


def test_loops_validate_their_roles() -> None:
    proposer = ScriptedAgent("proposer")
    with pytest.raises(ValidationError, match="at least one critic"):
        DebateLoop(proposer, [])
    with pytest.raises(ValidationError, match="proposer must satisfy"):
        DebateLoop("not an agent", [ScriptedAgent("c")])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="critics must satisfy"):
        DebateLoop(proposer, ["nope"])  # type: ignore[list-item]


def test_too_few_critics_for_the_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not enough critics"):
        DebateLoop(
            ScriptedAgent("p"), [ScriptedAgent("c")], config=DebateConfig(critics=3)
        )


def test_transcript_and_signature_cover_the_whole_run() -> None:
    result = loop(config=DebateConfig(rounds=1, critics=1)).run(CONTEXT)
    transcript = result.transcript()
    assert transcript[0].startswith("proposal: ")
    assert transcript[-1].startswith("converged=")
    assert result.signature() == result.signature()
    assert len(result.signature()) == 16


def test_replay_fixtures_can_drive_a_debate() -> None:
    config = DebateConfig(rounds=1, critics=1)
    statement = "kinase K1 increases A"
    revised = "kinase K1 increases A in vitro"
    entries = [
        ReplayEntry("propose", config.proposal_prompt, statement, context=CONTEXT),
        ReplayEntry(
            "critique",
            config.critique_prompt_for(statement),
            "no assay named",
            context=CONTEXT,
        ),
        ReplayEntry("revise", statement, revised),
    ]
    agent = ReplayAgent("fixture", entries, mode="sequence")
    result = DebateLoop(agent, [agent], config=config).run(CONTEXT)
    assert result.final_statement == revised
    assert result.rounds_run == 1
