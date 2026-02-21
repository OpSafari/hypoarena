"""Dedup report serialization: roundtrips, integrity and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.dedup import DedupConfig, DuplicateFinder
from hypoarena.errors import SchemaError, ValidationError
from hypoarena.serialize import (
    dedup_config_from_dict,
    dedup_config_to_dict,
    dedup_report_from_dict,
    dedup_report_from_lines,
    dedup_report_to_dict,
    dedup_report_to_lines,
    dedup_signature,
    duplicate_cluster_from_dict,
    duplicate_cluster_to_dict,
)

TEXTS = {
    "a": "Protein A increases cell growth in HeLa cells",
    "b": "Protein A increases cell growth in HeLa cells.",
    "c": "kinase K1 phosphorylates protein A",
}


def report() -> object:
    finder = DuplicateFinder(DedupConfig(method="exact"))
    return finder.report(TEXTS)


def test_configs_roundtrip() -> None:
    config = DedupConfig(
        method="minhash", threshold=0.7, shingle_unit="word", content_only=True
    )
    assert dedup_config_from_dict(dedup_config_to_dict(config)) == config


def test_clusters_keep_their_verified_similarities() -> None:
    produced = report()
    for cluster in produced.clusters:
        assert (
            duplicate_cluster_from_dict(duplicate_cluster_to_dict(cluster)) == cluster
        )


def test_reports_roundtrip() -> None:
    produced = report()
    restored = dedup_report_from_dict(dedup_report_to_dict(produced))
    assert restored == produced
    assert restored.as_dict() == produced.as_dict()


def test_report_dict_keys_are_golden() -> None:
    payload = dedup_report_to_dict(report())
    assert sorted(payload) == ["clusters", "config", "schema_version", "total"]
    assert sorted(payload["config"]) == [
        "bands",
        "content_only",
        "method",
        "min_document_frequency",
        "ngram_size",
        "num_perm",
        "seed",
        "shingle_unit",
        "threshold",
        "use_lsh",
        "word_ngram_size",
    ]


def test_lines_roundtrip_and_verify_the_header() -> None:
    produced = report()
    lines = dedup_report_to_lines(produced)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds == ["meta", "report"]
    restored = dedup_report_from_lines(lines)
    assert restored == produced
    assert json.loads(lines[0])["signature"] == dedup_signature(produced)
    assert json.loads(lines[0])["counts"] == {
        "clusters": len(produced.clusters),
        "members": produced.duplicated,
    }


def test_tampering_is_detected() -> None:
    produced = report()
    lines = dedup_report_to_lines(produced)
    payload = json.loads(lines[1])
    payload["report"]["total"] = 99  # same counts, different content
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(SchemaError, match="signature disagrees"):
        dedup_report_from_lines(lines)
    lines = dedup_report_to_lines(produced)
    payload = json.loads(lines[1])
    payload["report"]["clusters"] = []  # counts no longer match the header
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(SchemaError, match="counts disagree"):
        dedup_report_from_lines(lines)
    payload = dedup_report_to_dict(report())
    payload["config"]["threshold"] = 4.0
    with pytest.raises(SchemaError, match="above the maximum"):
        dedup_report_from_dict(payload)
    payload = dedup_report_to_dict(report())
    payload["config"]["method"] = "magic"
    with pytest.raises(ValidationError, match="unknown dedup method"):
        dedup_report_from_dict(payload)


def test_unknown_records_and_keys_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown dedup record type"):
        dedup_report_from_lines(['{"record":"cluster","members":["a","b"]}'])
    payload = dedup_report_to_dict(report())
    payload["clusters"][0]["extra"] = 1
    with pytest.raises(SchemaError, match="unknown keys"):
        dedup_report_from_dict(payload)
    with pytest.raises(SchemaError, match="no report line"):
        dedup_report_from_lines([dedup_report_to_lines(report())[0]])


def test_cluster_members_must_survive_the_minimum_size_rule() -> None:
    payload = duplicate_cluster_to_dict(report().clusters[0])
    payload["members"] = ["a"]
    with pytest.raises(SchemaError, match="at least 2"):
        duplicate_cluster_from_dict(payload)
