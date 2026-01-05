"""Scripted revisions: progress per tier and whitespace normalization."""

from __future__ import annotations

from hypoarena.agents import ScriptedAgent

CLAIM = "protein A   increases cell growth"
CRITIQUE = ("the claim about protein A does not name a testable assay",)


def test_vague_revisions_do_not_progress() -> None:
    response = ScriptedAgent("v", quality=0.1).revise(CLAIM, CRITIQUE)
    assert response.text.endswith("(unchanged)")
    assert "population" not in response.text


def test_focused_revisions_narrow_the_scope() -> None:
    response = ScriptedAgent("f", quality=0.5).revise(CLAIM, CRITIQUE)
    assert response.text == "protein A increases cell growth in the assayed population"


def test_mechanistic_revisions_add_a_measurement() -> None:
    response = ScriptedAgent("m", quality=0.9).revise(CLAIM, CRITIQUE)
    assert response.text.endswith("measured by dose response")
    assert "assayed population" in response.text


def test_revision_collapses_prompt_whitespace() -> None:
    response = ScriptedAgent("f", quality=0.5).revise("  a   b  ", CRITIQUE)
    assert response.text.startswith("a b ")


def test_revision_length_grows_with_quality() -> None:
    lengths = [
        len(ScriptedAgent("a", quality=quality).revise(CLAIM, CRITIQUE).text)
        for quality in (0.1, 0.5, 0.9)
    ]
    assert lengths == sorted(lengths)


def test_all_three_tasks_are_supported() -> None:
    agent = ScriptedAgent("a", quality=0.9)
    assert sorted(agent.handlers()) == ["critique", "propose", "revise"]
    for task in sorted(agent.handlers()):
        assert agent.run(task, "prompt text", ("context line",)).text
