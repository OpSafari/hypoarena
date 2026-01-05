"""Seeded sweeps over the scripted adapter.

The property that matters for tournaments is monotonicity: for the same request,
a higher-quality agent must produce a strictly more specific reply. Everything
here is deterministic, so the sweep doubles as a regression net for the tier
boundaries.
"""

from __future__ import annotations

import pytest

from hypoarena.agents import DiscoveryAgent, ScriptedAgent

QUALITIES = (0.0, 0.16, 0.33, 0.34, 0.5, 0.66, 0.67, 0.85, 1.0)
TASKS = ("propose", "critique", "revise")
CONTEXT = (
    "Protein A increases cell growth in HeLa cells.",
    "Kinase K1 phosphorylates protein A.",
    "Batch effects were reviewed across all replicates.",
)
PROMPTS = {
    "propose": "Propose one hypothesis supported by the context.",
    "critique": "protein A increases cell growth",
    "revise": "protein A increases cell growth",
}


@pytest.mark.parametrize("task", TASKS)
def test_replies_are_deterministic_for_every_tier(task: str) -> None:
    for quality in QUALITIES:
        first = ScriptedAgent("a", quality=quality).run(task, PROMPTS[task], CONTEXT)
        second = ScriptedAgent("a", quality=quality).run(task, PROMPTS[task], CONTEXT)
        assert first.text == second.text
        assert first.request_id == second.request_id


@pytest.mark.parametrize("task", TASKS)
def test_specificity_never_decreases_with_quality(task: str) -> None:
    lengths = [
        len(ScriptedAgent("a", quality=quality).run(task, PROMPTS[task], CONTEXT).text)
        for quality in (0.1, 0.5, 0.9)
    ]
    assert lengths == sorted(lengths)


@pytest.mark.parametrize("quality", QUALITIES)
def test_every_tier_satisfies_the_protocol(quality: float) -> None:
    agent = ScriptedAgent("swept", quality=quality)
    assert isinstance(agent, DiscoveryAgent)
    for task in TASKS:
        response = agent.run(task, PROMPTS[task], CONTEXT)
        assert response.agent == "swept"
        assert response.request_id
        assert response.completion_tokens == len(response.text.split())
    assert agent.usage.calls == len(TASKS)
    assert set(agent.usage.by_task) == set(TASKS)


def test_agent_names_distinguish_replies_in_audit_trails() -> None:
    left = ScriptedAgent("left", quality=0.9).propose("propose", CONTEXT)
    right = ScriptedAgent("right", quality=0.9).propose("propose", CONTEXT)
    assert left.text == right.text
    assert left.agent != right.agent
    assert left.request_id == right.request_id


def test_context_size_changes_proposals_but_not_determinism() -> None:
    short = ScriptedAgent("a", quality=0.9).propose("propose", CONTEXT[:1]).text
    full = ScriptedAgent("a", quality=0.9).propose("propose", CONTEXT).text
    assert short != full
    assert ScriptedAgent("a", quality=0.9).propose("propose", CONTEXT).text == full
