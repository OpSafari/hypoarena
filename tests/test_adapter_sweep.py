"""Every adapter must drive the same loop with the same accounting shape.

The sweep runs scripted, replay and loopback-HTTP adapters through one transcript
and checks the invariants the debate loop depends on: protocol conformance,
request identity, deterministic text for a fixed input, and usage that adds up.
"""

from __future__ import annotations

import pytest

from hypoarena.agents import (
    DiscoveryAgent,
    ReplayAgent,
    ReplayEntry,
    ScriptedAgent,
    replay_transcript,
)
from hypoarena.http_agent import HttpAgent, HttpConfig
from loopback import MockChatServer, chat_body

PROMPT = "Propose one hypothesis supported by the context."
CONTEXT = ("kinase K1 phosphorylates protein A",)


def scripted() -> DiscoveryAgent:
    return ScriptedAgent("scripted", quality=0.9)


def replay() -> DiscoveryAgent:
    entries = [
        ReplayEntry(
            "propose", PROMPT, "kinase K1 increases protein A", context=CONTEXT
        ),
        ReplayEntry(
            "critique",
            "kinase K1 increases protein A",
            "the claim names no assay",
            context=CONTEXT,
        ),
        ReplayEntry(
            "revise",
            "kinase K1 increases protein A",
            "kinase K1 increases protein A in the assayed population",
            context=("the claim names no assay",),
        ),
    ]
    return ReplayAgent("replay", entries, mode="sequence")


def http_agent(server: MockChatServer) -> DiscoveryAgent:
    return HttpAgent(HttpConfig(base_url=server.base_url), name="http")


@pytest.mark.parametrize("factory", [scripted, replay], ids=["scripted", "replay"])
def test_offline_adapters_produce_a_full_transcript(factory) -> None:  # noqa: ANN001
    agent = factory()
    turns = replay_transcript(agent, PROMPT, CONTEXT)
    assert [turn.task for turn in turns] == ["propose", "critique", "revise"]
    assert isinstance(agent, DiscoveryAgent)
    assert agent.usage.calls == 3
    assert agent.usage.total_tokens > 0


def test_the_http_adapter_produces_the_same_shape() -> None:
    replies = [
        (200, chat_body("kinase K1 increases protein A")),
        (200, chat_body("the claim names no assay")),
        (200, chat_body("kinase K1 increases protein A in the assayed population")),
    ]
    with MockChatServer(replies) as server:
        agent = http_agent(server)
        turns = replay_transcript(agent, PROMPT, CONTEXT)
        assert len(server.received) == 3
    assert [turn.task for turn in turns] == ["propose", "critique", "revise"]
    assert agent.usage.calls == 3
    assert [turn.response.text for turn in turns] == [
        "kinase K1 increases protein A",
        "the claim names no assay",
        "kinase K1 increases protein A in the assayed population",
    ]


def test_request_ids_depend_only_on_request_content() -> None:
    first = scripted().propose(PROMPT, CONTEXT)
    second = replay().propose(PROMPT, CONTEXT)
    assert first.request_id == second.request_id
    assert first.request_id != scripted().critique(PROMPT, CONTEXT).request_id
    assert first.request_id != scripted().propose(PROMPT, ()).request_id


def test_offline_adapters_never_open_a_socket() -> None:
    with MockChatServer([(200, chat_body("unused"))]) as server:
        for agent in (scripted(), replay()):
            replay_transcript(agent, PROMPT, CONTEXT)
        assert server.received == []
