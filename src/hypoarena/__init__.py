"""hypoarena — an offline workbench for hypothesis-discovery pipelines.

The package implements the *mechanisms* used by generate-debate-evolve
discovery systems: grounded claim/evidence graphs, a synthetic literature
factory with planted ground truth, span-level grounding verification,
model-agnostic agent adapters, Bradley-Terry/Elo tournaments, paraphrase
deduplication, evolution operators and Bayesian evidence accumulation.

Nothing in this package measures or claims the scientific ability of any
particular language model. Every bundled test and example runs offline
against deterministic scripted adapters and synthetic corpora.
"""

from hypoarena._version import __version__

__all__ = ["__version__"]
