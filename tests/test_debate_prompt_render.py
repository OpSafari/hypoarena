"""DebateConfig prompt rendering follows the documented revision convention.

Revision deliberately has no template: the statement is the prompt and the
critiques are the context. These pin that behaviour and the critique rendering.
"""

from __future__ import annotations

from hypoarena.debate import DebateConfig


def test_revise_prompt_for_returns_the_statement_unchanged() -> None:
    config = DebateConfig()
    assert config.revise_prompt_for("claim X causes Y") == "claim X causes Y"


def test_critique_prompt_for_embeds_the_statement() -> None:
    config = DebateConfig()
    assert "claim X causes Y" in config.critique_prompt_for("claim X causes Y")


def test_a_custom_critique_prompt_is_used_verbatim() -> None:
    config = DebateConfig(critique_prompt="Judge: {statement}")
    assert config.critique_prompt_for("Z") == "Judge: Z"
