"""Scripted critiques: tier content, determinism and usage."""

from __future__ import annotations

from hypoarena.agents import ScriptedAgent

CONTEXT = ("Protein A increases cell growth in HeLa cells.",)
CLAIM = "protein A increases cell growth"


def test_vague_critiques_are_content_free() -> None:
    response = ScriptedAgent("v", quality=0.1).critique(CLAIM, CONTEXT)
    assert response.text == "the claim needs more support"
    assert CLAIM.split()[0] not in response.text


def test_focused_critiques_name_the_subject_and_one_gap() -> None:
    response = ScriptedAgent("f", quality=0.5).critique(CLAIM, CONTEXT)
    assert "protein" in response.text
    assert "assay" in response.text
    assert "mechanism" not in response.text


def test_mechanistic_critiques_list_every_gap() -> None:
    response = ScriptedAgent("m", quality=0.9).critique(CLAIM, CONTEXT)
    for keyword in ("mechanism", "assay", "scope"):
        assert keyword in response.text


def test_critiques_without_context_still_name_a_subject() -> None:
    response = ScriptedAgent("f", quality=0.5).critique(CLAIM)
    assert "the subject" in response.text


def test_critique_length_grows_with_quality() -> None:
    lengths = [
        len(ScriptedAgent("a", quality=quality).critique(CLAIM, CONTEXT).text)
        for quality in (0.1, 0.5, 0.9)
    ]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == 3


def test_critique_usage_is_recorded() -> None:
    agent = ScriptedAgent("a", quality=0.5)
    agent.critique(CLAIM, CONTEXT)
    agent.critique(CLAIM, CONTEXT)
    assert agent.usage.by_task == {"critique": 2}
    assert agent.usage.total_tokens > 0
