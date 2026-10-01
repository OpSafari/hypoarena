"""Golden digests over the orchestration-stage serialized artifacts.

The tournament audit trail, debate transcripts and token ledger are the records
that make a run auditable. Pinning their digests catches drift in match ordering,
transcript encoding or usage accounting for a fixed seed.
"""

from __future__ import annotations

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.ids import stable_hash
from hypoarena.runner import Pipeline
from hypoarena.synthetic import SyntheticConfig

DIGESTS = {
    "tournament.jsonl": "18306870bba6b771",
    "debates.jsonl": "57668455b17cc5ac",
    "cost.json": "62f935ffda68ac5e",
}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ArtifactStore:
    root = tmp_path_factory.mktemp("orchestration-digests")
    config = RunConfig(
        seed=270106,
        run_id="golden",
        stages=STAGES,
        corpus=SyntheticConfig(seed=270106, chains=2, chain_length=2),
    )
    Pipeline(config, ArtifactStore(root, "golden")).run()
    return ArtifactStore(root, "golden")


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_serialized_orchestration_digest_is_pinned(
    store: ArtifactStore, name: str
) -> None:
    text = store.path(name).read_text(encoding="utf-8")
    assert stable_hash(text) == DIGESTS[name]


def test_the_ledger_counts_tokens_but_never_prices_them(
    store: ArtifactStore,
) -> None:
    text = store.path("cost.json").read_text(encoding="utf-8").lower()
    assert "token" in text
    assert "price" not in text
    assert "usd" not in text
