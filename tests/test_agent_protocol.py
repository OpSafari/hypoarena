"""Protocol conformance, request identity and usage bookkeeping."""

from __future__ import annotations

import pytest

from hypoarena.agents import (
    AgentRequest,
    AgentResponse,
    BaseAgent,
    DiscoveryAgent,
    Usage,
)
from hypoarena.errors import AdapterError, ValidationError


class EchoAgent(BaseAgent):
    """Minimal adapter used to exercise the template machinery."""

    def __init__(self, name: str = "echo", *, echo_id: bool = True) -> None:
        super().__init__(name)
        self.echo_id = echo_id

    def respond(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.request_id if self.echo_id else "req_ffffffffffff",
            text=f"{request.task}: {request.prompt}",
            agent=self.name,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=2,
        )


def test_base_agent_implements_the_protocol() -> None:
    agent = EchoAgent()
    assert isinstance(agent, DiscoveryAgent)
    assert isinstance(agent.usage, Usage)


def test_structural_conformance_does_not_require_inheritance() -> None:
    class Foreign:
        name = "foreign"
        usage = Usage()

        def respond(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(request.request_id, "ok", self.name)

        def propose(self, prompt: str, context: object = ()) -> AgentResponse:
            return self.respond(AgentRequest("propose", prompt))

        def critique(self, prompt: str, context: object = ()) -> AgentResponse:
            return self.respond(AgentRequest("critique", prompt))

        def revise(self, prompt: str, context: object = ()) -> AgentResponse:
            return self.respond(AgentRequest("revise", prompt))

    assert isinstance(Foreign(), DiscoveryAgent)


def test_blank_agent_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="name"):
        EchoAgent("   ")


def test_template_methods_route_the_task_and_record_usage() -> None:
    agent = EchoAgent()
    assert agent.propose("a hypothesis").text == "propose: a hypothesis"
    assert agent.critique("a hypothesis").text == "critique: a hypothesis"
    assert agent.revise("a hypothesis").text == "revise: a hypothesis"
    assert agent.usage.as_dict()["by_task"] == {
        "critique": 1,
        "propose": 1,
        "revise": 1,
    }
    assert agent.usage.calls == 3


def test_responses_for_other_requests_are_rejected() -> None:
    agent = EchoAgent(echo_id=False)
    with pytest.raises(AdapterError, match="different request"):
        agent.propose("a hypothesis")
    assert agent.usage.calls == 0


def test_a_bare_base_agent_is_not_usable() -> None:
    with pytest.raises(NotImplementedError):
        BaseAgent("bare").respond(AgentRequest("propose", "x"))
