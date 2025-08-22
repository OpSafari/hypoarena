"""Debate configuration: defaults, placeholder rules and fingerprints."""

from __future__ import annotations

import pytest

from hypoarena.debate import (
    DEFAULT_CRITIQUE_PROMPT,
    DEFAULT_PROPOSAL_PROMPT,
    DebateConfig,
)
from hypoarena.errors import ValidationError


def test_defaults_are_golden() -> None:
    config = DebateConfig()
    assert config.rounds == 2
    assert config.critics == 2
    assert config.stop_on_unchanged is True
    assert config.proposal_prompt == DEFAULT_PROPOSAL_PROMPT
    assert config.critique_prompt == DEFAULT_CRITIQUE_PROMPT


def test_counts_are_validated_in_both_directions() -> None:
    assert DebateConfig(rounds=1, critics=1).rounds == 1
    with pytest.raises(ValidationError, match="rounds"):
        DebateConfig(rounds=0)
    with pytest.raises(ValidationError, match="at least one critic"):
        DebateConfig(critics=0)


def test_prompts_must_not_be_blank_and_must_carry_the_statement() -> None:
    with pytest.raises(ValidationError, match="blank"):
        DebateConfig(proposal_prompt="  ")
    with pytest.raises(ValidationError, match="placeholder"):
        DebateConfig(critique_prompt="critique it")


def test_prompt_rendering_substitutes_the_statement() -> None:
    config = DebateConfig()
    assert config.critique_prompt_for("A increases B") == (
        "Critique this hypothesis: A increases B"
    )


def test_custom_prompts_render_with_the_same_contract() -> None:
    config = DebateConfig(critique_prompt="{statement} -- what is missing?")
    assert config.critique_prompt_for("X") == "X -- what is missing?"


def test_fingerprint_tracks_every_field() -> None:
    baseline = DebateConfig().fingerprint()
    assert baseline == DebateConfig().fingerprint()
    assert DebateConfig(rounds=3).fingerprint() != baseline
    assert DebateConfig(stop_on_unchanged=False).fingerprint() != baseline
    assert len(baseline) == 16
