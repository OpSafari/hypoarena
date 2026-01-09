"""Run-level token accounting: merging, summaries and the no-billing rule."""

from __future__ import annotations

from hypoarena.agents import ScriptedAgent, Usage, merge_usage, usage_summary

CONTEXT = ("kinase K1 phosphorylates protein A",)


def used_agent(name: str, quality: float) -> ScriptedAgent:
    agent = ScriptedAgent(name, quality=quality)
    agent.propose("propose a hypothesis", CONTEXT)
    agent.critique("kinase K1 increases protein A", CONTEXT)
    return agent


def test_merge_sums_calls_tokens_and_tasks() -> None:
    left = Usage(calls=2, prompt_tokens=10, completion_tokens=4, by_task={"propose": 2})
    right = Usage(
        calls=1, prompt_tokens=5, completion_tokens=3, by_task={"critique": 1}
    )
    merged = merge_usage(left, right)
    assert merged.as_dict() == {
        "calls": 3,
        "prompt_tokens": 15,
        "completion_tokens": 7,
        "total_tokens": 22,
        "by_task": {"critique": 1, "propose": 2},
    }


def test_merge_of_nothing_is_empty_and_inputs_are_untouched() -> None:
    assert merge_usage().as_dict()["calls"] == 0
    left = Usage(calls=1, prompt_tokens=2, completion_tokens=3, by_task={"propose": 1})
    merge_usage(left, left)
    assert left.calls == 1


def test_usage_summary_is_keyed_by_agent_name() -> None:
    summary = usage_summary([used_agent("proposer", 0.9), used_agent("critic", 0.2)])
    assert sorted(summary["agents"]) == ["critic", "proposer"]  # type: ignore[arg-type]
    total = summary["total"]
    assert isinstance(total, dict)
    assert total["calls"] == 4
    assert total["by_task"] == {"critique": 2, "propose": 2}


def test_accounting_records_counts_only_never_prices() -> None:
    summary = usage_summary([used_agent("a", 0.5)])
    total = summary["total"]
    assert isinstance(total, dict)
    assert sorted(total) == [
        "by_task",
        "calls",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
    ]
    forbidden = {"cost", "price", "usd", "billing", "currency"}
    assert not (forbidden & {str(key).lower() for key in total})


def test_token_totals_agree_with_the_individual_responses() -> None:
    agent = ScriptedAgent("a", quality=0.9)
    responses = [
        agent.propose("propose a hypothesis", CONTEXT),
        agent.critique("kinase K1 increases protein A", CONTEXT),
    ]
    assert agent.usage.prompt_tokens == sum(item.prompt_tokens for item in responses)
    assert agent.usage.completion_tokens == sum(
        item.completion_tokens for item in responses
    )
    assert agent.usage.total_tokens == sum(item.total_tokens for item in responses)
