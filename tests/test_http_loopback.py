"""Adapter integration against the loopback mock: requests and replies."""

from __future__ import annotations

from hypoarena.http_agent import HttpAgent, HttpConfig
from loopback import MockChatServer, chat_body


def agent_for(server: MockChatServer, **overrides: object) -> HttpAgent:
    config = HttpConfig(base_url=server.base_url, **overrides)  # type: ignore[arg-type]
    return HttpAgent(config, name="loopback")


def test_a_call_reaches_the_mock_and_returns_its_text() -> None:
    with MockChatServer([(200, chat_body("kinase K1 increases protein A"))]) as server:
        response = agent_for(server).propose("propose", ("kinase K1 binds protein A",))
    assert response.text == "kinase K1 increases protein A"
    assert response.model == "mock-model"
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 2
    assert response.finish_reason == "stop"


def test_the_mock_receives_the_expected_payload() -> None:
    with MockChatServer([(200, chat_body())]) as server:
        agent_for(server, model="demo").propose("propose a hypothesis", ("context",))
        sent = server.received[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["payload"]["model"] == "demo"
    assert sent["payload"]["temperature"] == 0.0
    assert "max_tokens" not in sent["payload"]
    assert sent["payload"]["messages"] == [
        {"role": "user", "content": "propose a hypothesis\ncontext"}
    ]


def test_sequential_calls_are_served_in_order() -> None:
    replies = [(200, chat_body("first")), (200, chat_body("second"))]
    with MockChatServer(replies) as server:
        agent = agent_for(server)
        texts = [agent.propose("propose").text for _ in range(2)]
        assert len(server.received) == 2
    assert texts == ["first", "second"]


def test_usage_is_recorded_from_the_upstream_counters() -> None:
    body = chat_body("ok", prompt_tokens=11, completion_tokens=7)
    with MockChatServer([(200, body)]) as server:
        agent = agent_for(server)
        agent.propose("propose")
    assert agent.usage.as_dict() == {
        "calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "by_task": {"propose": 1},
    }


def test_a_system_prompt_is_sent_as_its_own_message() -> None:
    with MockChatServer([(200, chat_body())]) as server:
        agent = HttpAgent(
            HttpConfig(base_url=server.base_url), system_prompt="answer in one line"
        )
        agent.critique("a claim")
    messages = server.received[0]["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "answer in one line"}
    assert messages[1]["role"] == "user"
