"""Golden digests over the core serialized records of a fixed run.

The corpus, graph and planted-truth artifacts are the deterministic seed of
every run. Pinning their content digests means any change to the schema, the
canonical line encoding or the generator surfaces here as a deliberate edit
rather than silent drift.
"""

from __future__ import annotations

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.ids import stable_hash
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig

DIGESTS = {
    "corpus.jsonl": "ca3f3f4c6102690b",
    "graph.jsonl": "3b8a7789f1c57317",
    "truth.jsonl": "6baa7235b33fe84d",
}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ArtifactStore:
    root = tmp_path_factory.mktemp("record-digests")
    config = RunConfig(
        seed=270106,
        run_id="golden",
        stages=STAGES,
        corpus=SyntheticConfig(seed=270106, chains=2, chain_length=2),
    )
    Pipeline(config, ArtifactStore(root, "golden")).run()
    return ArtifactStore(root, "golden")


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_serialized_record_digest_is_pinned(store: ArtifactStore, name: str) -> None:
    text = store.path(name).read_text(encoding="utf-8")
    assert stable_hash(text) == DIGESTS[name]


def test_pinned_records_are_non_empty(store: ArtifactStore) -> None:
    for name in DIGESTS:
        assert store.path(name).read_text(encoding="utf-8").strip()
