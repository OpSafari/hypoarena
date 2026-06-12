"""Cross-module integration: invariants hold across the whole pipeline.

After a full offline run the graph must still be valid, beliefs must stay inside
the open unit interval, every planted link must be recovered, and the tournament
may only have ranked claims that are actually in the graph.
"""

from __future__ import annotations

import pytest

from hypoarena.artifacts import ArtifactStore
from hypoarena.config import STAGES, RunConfig
from hypoarena.runner import Pipeline, run_report
from hypoarena.synthetic import SyntheticConfig


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    root = tmp_path_factory.mktemp("integration")
    config = RunConfig(
        seed=23,
        run_id="run",
        stages=STAGES,
        corpus=SyntheticConfig(seed=23, chains=3, chain_length=3),
    )
    runner = Pipeline(config, ArtifactStore(root, "run"))
    runner.run()
    return runner


def test_the_graph_is_valid_after_every_stage(pipeline: Pipeline) -> None:
    pipeline.state.graph.validate()


def test_beliefs_stay_strictly_within_bounds(pipeline: Pipeline) -> None:
    assert pipeline.state.beliefs
    assert all(0.0 < belief.posterior < 1.0 for belief in pipeline.state.beliefs)


def test_every_planted_link_is_recovered(pipeline: Pipeline) -> None:
    assert run_report(pipeline)["recovered"]["rate"] == 1.0


def test_ranking_subjects_are_a_subset_of_the_graph(pipeline: Pipeline) -> None:
    tournament = pipeline.state.tournament
    assert tournament is not None
    claim_ids = {claim.claim_id for claim in pipeline.state.graph.claims}
    assert set(tournament.subjects) <= claim_ids
