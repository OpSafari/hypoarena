"""The feature judge: explainable dimensions over real claim structures."""

from __future__ import annotations

import pytest

from helpers import sample_claim
from hypoarena.errors import ValidationError
from hypoarena.grounding import GroundingVerifier
from hypoarena.schema import ClaimRelation, PredictedRelation, Scope
from hypoarena.synthetic import SyntheticConfig, build_bundle
from hypoarena.tournament import FeatureJudge, Judge, graph_contradiction_counts


def test_the_judge_satisfies_the_protocol() -> None:
    assert isinstance(FeatureJudge(), Judge)
    assert FeatureJudge().name == "features"


def test_grounding_follows_the_report_when_one_is_known() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=1, chain_length=2))
    verifier = GroundingVerifier(bundle.corpus)
    reports = {claim.claim_id: verifier.verify(claim) for claim in bundle.claims}
    judge = FeatureJudge(reports=reports)
    claim = bundle.claims[0]
    assert judge.grounding_for(claim) == pytest.approx(reports[claim.claim_id].score)


def test_grounding_falls_back_to_citation_presence() -> None:
    judge = FeatureJudge()
    assert judge.grounding_for(sample_claim()) == 1.0
    assert judge.grounding_for(sample_claim(citations=())) == 0.0


def test_testability_accumulates_explainable_features() -> None:
    judge = FeatureJudge()
    bare = sample_claim(
        relation=PredictedRelation.ASSOCIATES, mechanism=None, scope=Scope("cells")
    )
    assert judge.testability_for(bare) == pytest.approx(0.4)
    directional = sample_claim(
        relation=PredictedRelation.INCREASES, mechanism=None, scope=Scope("cells")
    )
    assert judge.testability_for(directional) == pytest.approx(0.6)
    assert judge.testability_for(sample_claim(scope=Scope("cells"))) == pytest.approx(
        0.8
    )
    assert judge.testability_for(sample_claim(scope=Scope("cells", ("hypoxia",)))) == (
        pytest.approx(1.0)
    )


def test_novelty_scales_with_statement_specificity() -> None:
    judge = FeatureJudge(novelty_scale=12)
    short = sample_claim(statement="A binds B")
    long = sample_claim(
        statement="protein A increases gene B expression through phosphorylation"
    )
    assert judge.novelty_for(short) < judge.novelty_for(long)
    assert judge.novelty_for(long) <= 1.0
    with pytest.raises(ValidationError, match="novelty_scale"):
        FeatureJudge(novelty_scale=0)


def test_consistency_is_penalised_per_contradiction() -> None:
    judge = FeatureJudge(contradictions={"clm_0123456789ab": 2})
    assert judge.consistency_for(sample_claim()) == pytest.approx(0.5)
    assert judge.consistency_for(sample_claim(claim_id="clm_ffffffffffff")) == 1.0
    heavy = FeatureJudge(contradictions={"clm_0123456789ab": 9})
    assert heavy.consistency_for(sample_claim()) == 0.0


def test_scores_combine_all_four_dimensions() -> None:
    judge = FeatureJudge()
    score = judge.score(sample_claim())
    assert score.as_dict()["grounding"] == 1.0
    assert 0.0 <= score.novelty <= 1.0
    # directional relation and a mechanism, but no scope condition: 0.4 + 0.2 * 2
    assert score.testability == pytest.approx(0.8)
    narrowed = judge.score(sample_claim(scope=Scope("cells", ("hypoxia",))))
    assert narrowed.testability == pytest.approx(1.0)
    assert narrowed.consistency == score.consistency == 1.0


def test_contradiction_counts_come_straight_from_the_graph() -> None:
    bundle = build_bundle(SyntheticConfig(seed=3, chains=2, chain_length=2))
    graph = bundle.graph()
    counts = graph_contradiction_counts(graph)
    assert sum(counts.values()) == 2 * graph.edge_count
    assert all(edge.relation is ClaimRelation.CONTRADICTS for edge in graph.edges)
    for claim_id in counts:
        assert graph.has_claim(claim_id)
