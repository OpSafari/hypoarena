"""Golden digests over the analysis-stage serialized artifacts of a fixed run.

Grounding reports, dedup clusters, beliefs and evolution steps are the analyzed
output of the pipeline. Pinning their digests catches any drift in how these
records are encoded, ordered or rounded.
"""

from __future__ import annotations

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.ids import stable_hash
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig

DIGESTS = {
    "grounding.jsonl": "9a23ae4fcd42ead3",
    "dedup.jsonl": "9de7cff9e6e543b3",
    "beliefs.jsonl": "81b181d700d24da3",
    "evolution.jsonl": "00ea44ae0a0be637",
}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ArtifactStore:
    root = tmp_path_factory.mktemp("analysis-digests")
    config = RunConfig(
        seed=270106,
        run_id="golden",
        stages=STAGES,
        corpus=SyntheticConfig(seed=270106, chains=2, chain_length=2),
    )
    Pipeline(config, ArtifactStore(root, "golden")).run()
    return ArtifactStore(root, "golden")


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_serialized_analysis_digest_is_pinned(store: ArtifactStore, name: str) -> None:
    text = store.path(name).read_text(encoding="utf-8")
    assert stable_hash(text) == DIGESTS[name]


def test_analysis_artifacts_are_present_and_non_empty(store: ArtifactStore) -> None:
    for name in DIGESTS:
        assert store.exists(name)
        assert store.path(name).read_text(encoding="utf-8").strip()
