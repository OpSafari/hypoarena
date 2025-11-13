"""Run metadata: construction, serialization and the no-clock rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from hypoarena._version import __version__
from hypoarena.artifacts import METADATA_ARTIFACT, ArtifactStore, RunMetadata
from hypoarena.config import RunConfig
from hypoarena.errors import ValidationError
from hypoarena.schema import SCHEMA_VERSION


def test_metadata_is_derived_from_the_configuration() -> None:
    config = RunConfig(seed=9, run_id="demo", stages=("corpus", "verify"))
    metadata = RunMetadata.from_config(config)
    assert metadata.run_id == "demo"
    assert metadata.seed == 9
    assert metadata.stages == ("corpus", "verify")
    assert metadata.package_version == __version__
    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.config_fingerprint == config.fingerprint()


def test_no_timestamp_is_invented() -> None:
    assert RunMetadata.from_config(RunConfig()).created_at is None
    stamped = RunMetadata.from_config(RunConfig(), created_at="2026-09-01T00:00:00Z")
    assert stamped.created_at == "2026-09-01T00:00:00Z"


def test_metadata_roundtrips_through_its_dict() -> None:
    metadata = RunMetadata.from_config(RunConfig(), created_at="2026-09-01T00:00:00Z")
    assert RunMetadata.from_dict(metadata.as_dict()) == metadata
    assert sorted(metadata.as_dict()) == [
        "config_fingerprint",
        "created_at",
        "package_version",
        "run_id",
        "schema_version",
        "seed",
        "stages",
    ]


def test_unknown_metadata_keys_are_rejected() -> None:
    payload = RunMetadata.from_config(RunConfig()).as_dict()
    payload["owner"] = "someone"
    with pytest.raises(ValidationError, match="unknown run metadata keys"):
        RunMetadata.from_dict(payload)  # type: ignore[arg-type]


def test_the_store_persists_metadata_as_run_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "demo")
    metadata = RunMetadata.from_config(RunConfig(run_id="demo"))
    store.write_metadata(metadata)
    assert store.path(METADATA_ARTIFACT).is_file()
    assert store.read_metadata() == metadata
    assert METADATA_ARTIFACT in store.listing()


def test_reading_absent_metadata_raises(tmp_path: Path) -> None:
    from hypoarena.errors import ArtifactError

    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path, "demo").read_metadata()
