"""Golden over the run_report payload structure.

The report is consumed by the renderers, the CLI and downstream tooling, so its
top-level keys and the shape of each nested section are a contract. This pins
them against a small, fully offline run (computed once for the module).
"""

from __future__ import annotations

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.runner import Pipeline, run_report
from hypoarena.synthetic import SyntheticConfig

TOP_LEVEL = {
    "run_id",
    "config_fingerprint",
    "corpus_hash",
    "counts",
    "grounding",
    "dedup",
    "ranking",
    "beliefs",
    "evolution",
    "recovered",
    "cost",
    "limitations",
}


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("run-report")
    config = RunConfig(
        seed=270106,
        run_id="golden",
        stages=STAGES,
        corpus=SyntheticConfig(seed=270106, chains=2, chain_length=2),
    )
    pipeline = Pipeline(config, ArtifactStore(root, "golden"))
    pipeline.run()
    return run_report(pipeline)


def test_top_level_keys_are_exact(payload: dict) -> None:
    assert set(payload) == TOP_LEVEL


def test_nested_section_keys_are_exact(payload: dict) -> None:
    assert set(payload["counts"]) == {
        "documents",
        "claims",
        "evidence",
        "links",
        "edges",
    }
    assert set(payload["recovered"]) == {"planted", "recovered", "rate", "links"}
    assert set(payload["evolution"]) == {"generations", "accepted", "rejected"}
    assert set(payload["cost"]) == {
        "entries",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }


def test_limitations_are_always_five_strings(payload: dict) -> None:
    limitations = payload["limitations"]
    assert isinstance(limitations, list)
    assert len(limitations) == 5
    assert all(isinstance(item, str) and item for item in limitations)


def test_recovered_links_carry_a_statement_and_flag(payload: dict) -> None:
    links = payload["recovered"]["links"]
    assert links
    assert all(set(link) == {"statement", "recovered"} for link in links)
