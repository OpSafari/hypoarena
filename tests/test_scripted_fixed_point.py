"""Scripted revision must converge, otherwise no loop can ever stop.

Applying a revision twice has to return the same text: the debate loop detects
convergence by comparing statements, so a tier that keeps appending clauses would
never reach a fixed point and every run would burn its full round budget.
"""

from __future__ import annotations

import pytest

from hypoarena.agents import ScriptedAgent

CLAIM = "protein A increases cell growth"
CRITIQUE = ("the claim about protein A does not name a testable assay",)
QUALITIES = (0.1, 0.5, 0.9)


@pytest.mark.parametrize("quality", QUALITIES)
def test_a_second_revision_changes_nothing(quality: float) -> None:
    agent = ScriptedAgent("s", quality=quality)
    first = agent.revise(CLAIM, CRITIQUE).text
    second = agent.revise(first, CRITIQUE).text
    assert second == first


@pytest.mark.parametrize("quality", QUALITIES)
def test_the_first_revision_still_makes_progress(quality: float) -> None:
    agent = ScriptedAgent("s", quality=quality)
    assert agent.revise(CLAIM, CRITIQUE).text != CLAIM


@pytest.mark.parametrize("quality", QUALITIES)
def test_repeated_revision_is_stable_from_any_start(quality: float) -> None:
    agent = ScriptedAgent("s", quality=quality)
    statement = CLAIM
    for _ in range(4):
        statement = agent.revise(statement, CRITIQUE).text
    assert agent.revise(statement, CRITIQUE).text == statement


def test_higher_tiers_still_produce_longer_final_statements() -> None:
    finals = [
        ScriptedAgent("s", quality=quality).revise(CLAIM, CRITIQUE).text
        for quality in QUALITIES
    ]
    assert len(finals) == len(set(finals))
    assert [len(text) for text in finals] == sorted(len(text) for text in finals)
