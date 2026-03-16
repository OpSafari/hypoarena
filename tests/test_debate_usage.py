"""Debate accounting must describe the debate, not the whole process.

Agents are shared across debates and pipeline stages. If a transcript carried the
agents' cumulative counters, the same debate would serialize differently
depending on what had already run — which would break the byte-identity of
resumed runs. These tests pin the delta behaviour.
"""

from __future__ import annotations

from hypoarena.agents import ScriptedAgent, Usage
from hypoarena.debate import DebateConfig, DebateLoop
from hypoarena.serialize import debate_to_lines

CONTEXT = ("kinase K1 phosphorylates protein A",)
PROMPT = DebateConfig(rounds=1, critics=1).proposal_prompt


def loop(proposer: ScriptedAgent, critic: ScriptedAgent) -> DebateLoop:
    return DebateLoop(proposer, [critic], config=DebateConfig(rounds=1, critics=1))


def test_repeated_debates_report_the_same_usage() -> None:
    proposer = ScriptedAgent("proposer", quality=0.9)
    critic = ScriptedAgent("critic", quality=0.5)
    debate = loop(proposer, critic)
    first = debate.run(CONTEXT)
    second = debate.run(CONTEXT)
    assert first.usage == second.usage
    assert debate_to_lines(first) == debate_to_lines(second)


def test_usage_counts_only_the_current_debate() -> None:
    proposer = ScriptedAgent("proposer", quality=0.9)
    critic = ScriptedAgent("critic", quality=0.5)
    debate = loop(proposer, critic)
    result = debate.run(CONTEXT)
    total = result.usage["total"]
    assert isinstance(total, dict)
    # one proposal, one critique and one revision
    assert total["calls"] == 3
    assert total["by_task"] == {"critique": 1, "propose": 1, "revise": 1}


def test_a_snapshot_without_arguments_reports_cumulative_usage() -> None:
    proposer = ScriptedAgent("proposer", quality=0.9)
    critic = ScriptedAgent("critic", quality=0.5)
    debate = loop(proposer, critic)
    debate.run(CONTEXT)
    cumulative = debate.usage_summary()
    agents = cumulative["agents"]
    assert isinstance(agents, dict)
    assert agents["proposer"]["calls"] == 2  # propose + revise


def test_deltas_are_never_negative() -> None:
    current = Usage(3, 30, 12, {"propose": 3})
    base = Usage(1, 10, 4, {"propose": 1})
    delta = DebateLoop.usage_delta(current, base)
    assert (delta.calls, delta.prompt_tokens, delta.completion_tokens) == (2, 20, 8)
    assert delta.by_task == {"propose": 2}
    empty = DebateLoop.usage_delta(base, base)
    assert empty.calls == 0
    assert empty.by_task == {}


def test_a_shared_reviser_is_counted_once() -> None:
    agent = ScriptedAgent("all-roles", quality=0.9)
    debate = DebateLoop(agent, [agent], agent, DebateConfig(rounds=1, critics=1))
    result = debate.run(CONTEXT)
    agents = result.usage["agents"]
    assert isinstance(agents, dict)
    assert list(agents) == ["all-roles"]
    total = result.usage["total"]
    assert isinstance(total, dict)
    assert total["calls"] == 3
