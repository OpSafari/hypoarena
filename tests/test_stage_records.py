"""Stage records: validation, views and the runner re-export."""

from __future__ import annotations

import pytest

from hypoarena.errors import ValidationError
from hypoarena.runner import RunSummary as SummaryFromRunner
from hypoarena.runner import StageResult as ResultFromRunner
from hypoarena.stages import RunSummary, StageResult


def test_the_runner_still_exposes_the_records() -> None:
    assert SummaryFromRunner is RunSummary
    assert ResultFromRunner is StageResult


def test_stage_results_validate_and_serialize() -> None:
    result = StageResult("corpus", 12, ("corpus.jsonl",), skipped=False)
    assert result.as_dict() == {
        "stage": "corpus",
        "records": 12,
        "artifacts": ["corpus.jsonl"],
        "skipped": False,
    }
    with pytest.raises(ValidationError, match="stage name"):
        StageResult(" ", 0, ())
    with pytest.raises(ValidationError, match="records"):
        StageResult("corpus", -1, ())


def test_summaries_split_executed_and_skipped_stages() -> None:
    summary = RunSummary(
        run_id="demo",
        config_fingerprint="0" * 16,
        stages=(
            StageResult("corpus", 3, ("corpus.jsonl",)),
            StageResult("verify", 0, (), skipped=True),
        ),
        completed=True,
    )
    assert summary.executed == ("corpus",)
    assert summary.skipped == ("verify",)
    assert summary.as_dict()["completed"] is True
    assert len(summary.signature()) == 16


def test_a_summary_of_skipped_stages_only_reports_no_work() -> None:
    summary = RunSummary("demo", "0" * 16, (StageResult("corpus", 0, (), True),), True)
    assert summary.executed == ()
    assert summary.skipped == ("corpus",)
