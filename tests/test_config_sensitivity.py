"""RunConfig.fingerprint must be stable and sensitive to what it covers.

The fingerprint is the run's identity: identical configuration must hash
identically (so resumes match) and any setting that affects artifacts must change
it (so a changed run is never mistaken for a cached one). Both directions matter.
"""

from __future__ import annotations

from dataclasses import replace

from hypoarena.config import STAGES, RunConfig
from hypoarena.synthetic import SyntheticConfig


def base() -> RunConfig:
    return RunConfig(seed=1, run_id="x")


def test_identical_configs_share_a_fingerprint() -> None:
    assert base().fingerprint() == base().fingerprint()


def test_fingerprint_is_a_16_hex_digest() -> None:
    digest = base().fingerprint()
    assert len(digest) == 16
    assert all(character in "0123456789abcdef" for character in digest)


def test_fingerprint_changes_with_seed() -> None:
    assert base().fingerprint() != replace(base(), seed=2).fingerprint()


def test_fingerprint_changes_with_run_id() -> None:
    assert base().fingerprint() != replace(base(), run_id="y").fingerprint()


def test_fingerprint_changes_with_stages() -> None:
    assert base().fingerprint() != replace(base(), stages=STAGES[:3]).fingerprint()


def test_fingerprint_changes_with_corpus_shape() -> None:
    config = base()
    more_chains = replace(
        config, corpus=replace(config.corpus, chains=config.corpus.chains + 1)
    )
    longer = replace(
        config,
        corpus=replace(config.corpus, chain_length=config.corpus.chain_length + 1),
    )
    assert more_chains.fingerprint() != config.fingerprint()
    assert longer.fingerprint() != config.fingerprint()


def test_fingerprint_changes_with_a_whole_subconfig() -> None:
    config = base()
    other = replace(config, corpus=SyntheticConfig(seed=99, chains=4, chain_length=3))
    assert other.fingerprint() != config.fingerprint()
