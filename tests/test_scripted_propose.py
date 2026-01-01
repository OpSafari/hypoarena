"""Scripted proposals: tier behaviour, determinism and token proxy."""

from __future__ import annotations

import pytest

from hypoarena.agents import (
    QUALITY_TIERS,
    VAGUE_PROPOSAL,
    ScriptedAgent,
    quality_tier,
)
from hypoarena.errors import AdapterError, ValidationError

CONTEXT = (
    "Protein A increases cell growth in HeLa cells.",
    "Kinase K1 phosphorylates protein A.",
)


def test_quality_tiers_are_golden_and_ordered() -> None:
    assert QUALITY_TIERS == ("vague", "focused", "mechanistic")
    assert quality_tier(0.0) == "vague"
    assert quality_tier(0.33) == "vague"
    assert quality_tier(0.34) == "focused"
    assert quality_tier(0.66) == "focused"
    assert quality_tier(0.67) == "mechanistic"
    assert quality_tier(1.0) == "mechanistic"


def test_out_of_range_quality_is_rejected() -> None:
    with pytest.raises(ValidationError, match="quality"):
        quality_tier(1.5)
    with pytest.raises(ValidationError, match="quality"):
        ScriptedAgent(quality=-0.1)


def test_vague_agents_ignore_the_context() -> None:
    agent = ScriptedAgent("vague", quality=0.1)
    assert agent.propose("propose", CONTEXT).text == VAGUE_PROPOSAL


def test_focused_agents_use_context_entities() -> None:
    agent = ScriptedAgent("focused", quality=0.5)
    text = agent.propose("propose", ("kinase K1 phosphorylates protein A",)).text
    assert text == "kinase increases protein in the assayed population"
    assert agent.tier == "focused"


def test_mechanistic_agents_add_a_mechanism_and_a_measurement() -> None:
    text = ScriptedAgent("m", quality=0.9).propose("propose", CONTEXT).text
    assert " through " in text
    assert text.endswith("measured by dose response")


def test_proposal_specificity_is_monotone_in_quality() -> None:
    lengths = [
        len(ScriptedAgent("a", quality=quality).propose("propose", CONTEXT).text)
        for quality in (0.1, 0.5, 0.9)
    ]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == 3


def test_single_entity_contexts_fall_back_to_the_generic_form() -> None:
    agent = ScriptedAgent("m", quality=0.9)
    assert agent.propose("propose", ("protein protein protein",)).text == (
        VAGUE_PROPOSAL
    )


def test_proposals_without_context_fall_back_to_the_generic_form() -> None:
    agent = ScriptedAgent("focused", quality=0.5)
    assert agent.propose("propose").text == VAGUE_PROPOSAL


def test_replies_are_deterministic_per_request() -> None:
    first = ScriptedAgent("a", quality=0.5).propose("propose", CONTEXT)
    second = ScriptedAgent("a", quality=0.5).propose("propose", CONTEXT)
    assert first.text == second.text
    assert first.prompt_tokens == second.prompt_tokens


def test_token_counts_are_word_based_and_nonzero() -> None:
    response = ScriptedAgent("a", quality=0.5).propose("propose a hypothesis", CONTEXT)
    assert response.prompt_tokens == 3 + sum(len(line.split()) for line in CONTEXT)
    assert response.completion_tokens == len(response.text.split())
    assert response.model == "scripted"
    assert response.finish_reason == "stop"


def test_unsupported_tasks_are_reported() -> None:
    agent = ScriptedAgent("a", quality=0.5)
    with pytest.raises(AdapterError, match="cannot handle this task"):
        agent.run("judge", "score these two claims")
