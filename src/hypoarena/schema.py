"""Versioned schemas for claims, citations, evidence and provenance.

Design rules used throughout this module:

* every record is a frozen dataclass, so hashes and equality are structural;
* ``__post_init__`` performs validation, so an invalid record cannot exist —
  constructors raise instead of returning a broken object;
* ``to_dict``/``from_dict`` are exact inverses and go through
  :mod:`hypoarena.codec`, which rejects unknown keys;
* only top-level records (claims, evidence) carry ``schema_version``; nested
  value objects stay version-free to keep payloads small.
"""

from __future__ import annotations

from enum import StrEnum, unique

SCHEMA_VERSION = "1.0"


@unique
class PredictedRelation(StrEnum):
    """Direction a claim predicts between its subject and object variables."""

    INCREASES = "increases"
    DECREASES = "decreases"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    CAUSES = "causes"
    ASSOCIATES = "associates"


@unique
class EvidencePolarity(StrEnum):
    """How an evidence item bears on the claim it is attached to."""

    SUPPORT = "support"
    REFUTE = "refute"
    NEUTRAL = "neutral"


@unique
class ClaimRelation(StrEnum):
    """Typed edges between two claims in the hypothesis graph."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
