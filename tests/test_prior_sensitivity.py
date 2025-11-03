"""Prior sensitivity: spread, robustness and monotonicity."""

from __future__ import annotations

import pytest

from helpers import sample_evidence
from hypoarena.belief import (
    DEFAULT_PRIOR_GRID,
    BeliefConfig,
    ContradictionPolicy,
    is_prior_robust,
    prior_sensitivity,
    sensitivity_spread,
)
from hypoarena.errors import ValidationError
from hypoarena.schema import EvidencePolarity


def support(strength: float = 1.0) -> object:
    return sample_evidence(polarity=EvidencePolarity.SUPPORT, strength=strength)


def refute() -> object:
    return sample_evidence(polarity=EvidencePolarity.REFUTE, strength=1.0)


def test_the_grid_is_golden_and_used_by_default() -> None:
    assert DEFAULT_PRIOR_GRID == (0.1, 0.25, 0.5, 0.75, 0.9)
    pairs = prior_sensitivity([support()])
    assert [prior for prior, _ in pairs] == list(DEFAULT_PRIOR_GRID)


def test_posteriors_increase_with_the_prior() -> None:
    pairs = prior_sensitivity([support()])
    posteriors = [posterior for _, posterior in pairs]
    assert posteriors == sorted(posteriors)
    assert posteriors[0] < posteriors[-1]


def test_strong_evidence_shrinks_the_spread() -> None:
    weak = sensitivity_spread([support(0.2)])
    strong = sensitivity_spread([support(), support(), support()])
    assert strong < weak


def test_no_evidence_leaves_the_spread_at_its_maximum() -> None:
    assert sensitivity_spread([]) == pytest.approx(0.9 - 0.1)


def test_robustness_reports_one_sided_conclusions() -> None:
    # a high threshold keeps REJECT from flooring, so refutations still count
    counting = BeliefConfig(
        contradiction_policy=ContradictionPolicy.REJECT, contradiction_threshold=99
    )
    # three full-strength supports keep even a 0.1 prior above the threshold
    # (two would land exactly on 0.5 for that prior, which counts as undecided)
    assert is_prior_robust([support(), support(), support()]) is True
    assert is_prior_robust([support(), support()]) is False
    assert is_prior_robust([]) is False
    # four refutations pull even a 0.9 prior below the threshold
    assert is_prior_robust([refute()] * 4, threshold=0.5, config=counting) is True
    assert is_prior_robust([refute()], threshold=0.5, config=counting) is False
    # ignoring contradictions leaves every posterior at its prior, so no side wins
    ignoring = BeliefConfig(contradiction_policy=ContradictionPolicy.IGNORE)
    assert is_prior_robust([refute()] * 4, config=ignoring) is False


def test_an_empty_grid_is_rejected() -> None:
    with pytest.raises(ValidationError, match="prior grid"):
        prior_sensitivity([support()], ())


def test_policies_are_held_fixed_across_the_grid() -> None:
    config = BeliefConfig(contradiction_policy=ContradictionPolicy.REJECT)
    pairs = prior_sensitivity([support(), refute()], config=config)
    assert all(posterior == pairs[0][1] for _, posterior in pairs)
