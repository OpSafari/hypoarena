"""Golden snapshots of the ranker modules' public API.

The NumPy layer (``ranker``) imports without torch and is always pinned. The
torch layer (``ranker_torch``) can only be imported when the optional extra is
present, so its snapshot skips cleanly otherwise and is marked ``model``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest


def defined_public_names(module: Any) -> tuple[str, ...]:
    """Return the sorted public classes/functions defined in ``module``."""
    names = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, ModuleType):
            continue
        defined_here = getattr(obj, "__module__", None) == module.__name__
        if (isinstance(obj, type) or callable(obj)) and defined_here:
            names.append(name)
    return tuple(sorted(names))


def test_ranker_public_api() -> None:
    from hypoarena import ranker

    assert defined_public_names(ranker) == (
        "CalibrationBin",
        "LabeledExample",
        "RankerData",
        "RankerDatasetConfig",
        "RankerFeaturizer",
        "calibration_curve",
        "planted_quality",
        "require_torch",
        "spearman_correlation",
        "synthetic_ranking_dataset",
    )


@pytest.mark.model
def test_ranker_torch_public_api() -> None:
    pytest.importorskip("torch", reason="optional torch extra not installed")
    from hypoarena import ranker_torch

    assert defined_public_names(ranker_torch) == (
        "FitResult",
        "RubricRanker",
        "TrainResult",
        "fit_ranker",
        "predict_scores",
        "train_ranker",
    )
