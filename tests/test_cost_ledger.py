"""Token accounting: entries, grouping and the no-pricing rule."""

from __future__ import annotations

import pytest

from hypoarena.agents import ScriptedAgent, Usage
from hypoarena.cost import CostEntry, CostLedger
from hypoarena.errors import ValidationError


def usage(calls: int = 2, prompt: int = 10, completion: int = 4) -> Usage:
    return Usage(
        calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        by_task={"propose": calls},
    )


def test_entries_validate_their_fields() -> None:
    entry = CostEntry("generate", "scripted", 2, 10, 4)
    assert entry.total_tokens == 14
    assert entry.as_dict()["total_tokens"] == 14
    with pytest.raises(ValidationError, match="stage"):
        CostEntry(" ", "agent", 1, 1, 1)
    with pytest.raises(ValidationError, match="agent"):
        CostEntry("generate", " ", 1, 1, 1)
    with pytest.raises(ValidationError, match="calls"):
        CostEntry("generate", "agent", -1, 1, 1)


def test_recording_appends_and_totals() -> None:
    ledger = CostLedger()
    ledger.record("generate", "a", usage())
    ledger.record("generate", "b", usage(calls=1, prompt=5, completion=2))
    ledger.record("debate", "a", usage(calls=3, prompt=1, completion=1))
    assert ledger.totals() == {
        "entries": 3,
        "calls": 6,
        "prompt_tokens": 16,
        "completion_tokens": 7,
        "total_tokens": 23,
    }


def test_grouping_is_by_stage() -> None:
    ledger = CostLedger()
    ledger.record("generate", "a", usage())
    ledger.record("debate", "a", usage())
    grouped = ledger.by_stage()
    assert sorted(grouped) == ["debate", "generate"]
    assert grouped["generate"]["calls"] == 2
    assert len(ledger.for_stage("debate")) == 1
    assert ledger.for_stage("report") == ()


def test_an_empty_ledger_reports_zeros() -> None:
    ledger = CostLedger()
    assert ledger.totals() == {
        "entries": 0,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert ledger.as_dict()["entries"] == []


def test_adapter_usage_can_be_recorded_directly() -> None:
    agent = ScriptedAgent("proposer", quality=0.9)
    agent.propose("propose a hypothesis", ("kinase K1 phosphorylates protein A",))
    ledger = CostLedger()
    ledger.record("generate", agent.name, agent.usage)
    entry = ledger.for_stage("generate")[0]
    assert entry.calls == 1
    assert entry.total_tokens == agent.usage.total_tokens


def test_no_pricing_fields_exist() -> None:
    payload = CostLedger().as_dict()
    forbidden = {"cost", "price", "usd", "billing", "currency", "rate"}
    keys = {str(key).lower() for key in payload}
    keys |= {str(key).lower() for key in CostEntry("s", "a", 1, 1, 1).as_dict()}
    assert not (keys & forbidden)
