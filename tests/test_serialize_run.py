"""Run summary and cost ledger serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypoarena.agents import Usage
from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.cost import CostEntry, CostLedger
from hypoarena.errors import SchemaError
from hypoarena.runner import Pipeline, StageResult
from hypoarena.serialize import (
    cost_ledger_from_dict,
    cost_ledger_to_dict,
    run_summary_from_dict,
    stage_result_from_dict,
)
from hypoarena.synthetic import SyntheticConfig


def summary_payload() -> dict:
    return {
        "run_id": "demo",
        "config_fingerprint": "0123456789abcdef",
        "completed": True,
        "stages": [
            {
                "stage": "corpus",
                "records": 12,
                "artifacts": ["corpus.jsonl"],
                "skipped": False,
            }
        ],
        "executed": ["corpus"],
        "skipped": [],
    }


def test_stage_results_decode() -> None:
    payload = summary_payload()["stages"][0]
    result = stage_result_from_dict(payload)
    assert isinstance(result, StageResult)
    assert result.records == 12
    assert result.artifacts == ("corpus.jsonl",)
    assert result.skipped is False


def test_run_summaries_decode_and_re_encode() -> None:
    summary = run_summary_from_dict(summary_payload())
    assert summary.run_id == "demo"
    assert summary.completed is True
    assert summary.executed == ("corpus",)
    encoded = summary.as_dict()
    assert json.loads(json.dumps(encoded))["stages"] == summary_payload()["stages"]


def test_unknown_summary_keys_are_rejected() -> None:
    payload = summary_payload()
    payload["owner"] = "someone"
    with pytest.raises(SchemaError, match="unknown keys"):
        run_summary_from_dict(payload)


def test_ledger_helpers_are_available_from_the_serialization_layer() -> None:
    from hypoarena.serialize import cost_entry_to_dict

    assert sorted(cost_entry_to_dict(CostEntry("generate", "agent", 1, 2, 3))) == [
        "agent",
        "calls",
        "completion_tokens",
        "prompt_tokens",
        "stage",
        "total_tokens",
    ]


def test_ledgers_roundtrip_and_verify_their_totals() -> None:
    ledger = CostLedger()
    ledger.record("generate", "scripted-0.9", Usage(2, 20, 8, {"propose": 2}))
    ledger.record("debate", "scripted-0.5", Usage(3, 9, 6, {"critique": 3}))
    restored = cost_ledger_from_dict(cost_ledger_to_dict(ledger))
    assert [entry.as_dict() for entry in restored.entries] == [
        entry.as_dict() for entry in ledger.entries
    ]
    assert restored.totals() == ledger.totals()


def test_tampered_totals_are_rejected() -> None:
    ledger = CostLedger()
    ledger.record("generate", "agent", Usage(1, 2, 3))
    payload = cost_ledger_to_dict(ledger)
    payload["totals"]["calls"] = 99
    with pytest.raises(SchemaError, match="totals disagree"):
        cost_ledger_from_dict(payload)


def test_entries_validate_on_decode() -> None:
    entry = CostEntry("generate", "agent", 1, 2, 3)
    payload = dict(entry.as_dict())
    payload["calls"] = -1
    with pytest.raises(SchemaError, match="below the minimum"):
        from hypoarena.serialize import cost_entry_from_dict

        cost_entry_from_dict(payload)


def test_a_real_run_summary_decodes(tmp_path: Path) -> None:
    config = RunConfig(
        seed=29,
        run_id="real",
        stages=STAGES[:3],
        corpus=SyntheticConfig(seed=29, chains=1, chain_length=2),
    )
    store = ArtifactStore(tmp_path, "real")
    Pipeline(config, store).run()
    restored = run_summary_from_dict(store.read_json("summary.json"))
    assert restored.run_id == "real"
    assert tuple(stage.stage for stage in restored.stages) == STAGES[:3]
    assert restored.config_fingerprint == config.fingerprint()
