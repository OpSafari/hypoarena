"""Golden over key public constants that other modules and docs depend on.

The stage order, rubric dimensions, artifact file names and the count of the
report's limitations block are referenced across the runner, the CLI, the
renderers and the documentation. Pinning them keeps those references honest.
"""

from __future__ import annotations

from hypoarena import reports, runner
from hypoarena.artifacts import METADATA_ARTIFACT
from hypoarena.config import STAGES
from hypoarena.tournament import RUBRIC_DIMENSIONS


def test_pipeline_stage_order_is_pinned() -> None:
    assert STAGES == (
        "corpus",
        "generate",
        "verify",
        "dedup",
        "debate",
        "rank",
        "evolve",
        "accumulate",
        "report",
    )


def test_rubric_dimensions_are_pinned() -> None:
    assert RUBRIC_DIMENSIONS == ("novelty", "testability", "grounding", "consistency")


def test_artifact_name_constants_are_pinned() -> None:
    assert runner.REPORT_ARTIFACT == "report.json"
    assert runner.SUMMARY_ARTIFACT == "summary.json"
    assert runner.COST_ARTIFACT == "cost.json"
    assert reports.MARKDOWN_ARTIFACT == "report.md"
    assert reports.HTML_ARTIFACT == "report.html"
    assert METADATA_ARTIFACT == "run.json"


def test_report_limitations_block_has_five_entries() -> None:
    assert len(runner.REPORT_LIMITATIONS) == 5
    assert all(entry.strip() for entry in runner.REPORT_LIMITATIONS)


def test_report_limitations_mention_the_synthetic_scope() -> None:
    joined = " ".join(runner.REPORT_LIMITATIONS).lower()
    assert "synthetic" in joined
    assert "not" in joined
