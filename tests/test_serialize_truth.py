"""Planted truth serialization: roundtrips, integrity and rejections."""

from __future__ import annotations

import json

import pytest

from hypoarena.errors import SchemaError, ValidationError
from hypoarena.serialize import (
    cluster_from_dict,
    cluster_to_dict,
    link_from_dict,
    link_to_dict,
    truth_counts,
    truth_from_dict,
    truth_from_lines,
    truth_signature,
    truth_to_dict,
    truth_to_lines,
)
from hypoarena.synthetic import PlantedTruth, SyntheticConfig, build_bundle


def truth() -> PlantedTruth:
    return build_bundle(SyntheticConfig(seed=61, chains=2, chain_length=3)).truth


def test_link_and_cluster_dicts_roundtrip() -> None:
    planted = truth().true_links()[0]
    assert link_from_dict(link_to_dict(planted)) == planted
    cluster = truth().clusters[0]
    assert cluster_from_dict(cluster_to_dict(cluster)) == cluster


def test_link_decoding_rejects_unknown_kinds_and_relations() -> None:
    payload = link_to_dict(truth().true_links()[0])
    payload["kind"] = "guessed"
    with pytest.raises(ValidationError, match="unknown link kind"):
        link_from_dict(payload)
    payload = link_to_dict(truth().true_links()[0])
    payload["relation"] = "teleports"
    with pytest.raises(SchemaError, match="not a valid PredictedRelation"):
        link_from_dict(payload)


def test_cluster_decoding_requires_a_three_part_key() -> None:
    payload = cluster_to_dict(truth().clusters[0])
    payload["link_key"] = ["a", "b"]
    with pytest.raises(SchemaError, match="three parts"):
        cluster_from_dict(payload)


def test_truth_dict_roundtrip_preserves_everything_but_distractors() -> None:
    original = truth()
    restored = truth_from_dict(truth_to_dict(original))
    assert restored.chains == original.chains
    assert restored.competing == original.competing
    assert restored.contradictions == original.contradictions
    assert restored.clusters == original.clusters
    assert restored.distractor_ids == original.distractor_ids


def test_truth_dict_rejects_unknown_keys_and_versions() -> None:
    payload = truth_to_dict(truth())
    payload["reviewer"] = "anon"
    with pytest.raises(SchemaError, match="unknown keys"):
        truth_from_dict(payload)
    payload = truth_to_dict(truth())
    payload["schema_version"] = "0.1"
    with pytest.raises(SchemaError, match="unsupported schema version"):
        truth_from_dict(payload)


def test_truth_lines_roundtrip_and_verify_the_header() -> None:
    original = truth()
    lines = truth_to_lines(original)
    kinds = [json.loads(line)["record"] for line in lines]
    assert kinds[0] == "meta"
    assert kinds.count("chain") == len(original.chains)
    assert kinds.count("cluster") == len(original.clusters)
    assert kinds[-1] == "distractors"
    restored = truth_from_lines(lines)
    assert restored.chains == original.chains
    assert restored.distractor_ids == original.distractor_ids
    assert truth_signature(restored) == json.loads(lines[0])["signature"]


def test_truth_line_integrity_checks_fire_on_tampering() -> None:
    original = truth()
    lines = truth_to_lines(original)
    with pytest.raises(SchemaError, match="counts disagree"):
        truth_from_lines(lines[:-2])  # drops a cluster and the distractor record
    with pytest.raises(SchemaError, match="signature disagrees"):
        truth_from_lines(lines[:-1])  # drops only the distractor record
    meta = json.loads(lines[0])
    meta["signature"] = "0" * 16
    lines[0] = json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(SchemaError, match="signature disagrees"):
        truth_from_lines(lines)


def test_unknown_truth_record_types_are_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown truth record type"):
        truth_from_lines(['{"record":"observation","value":1}'])


def test_truth_counts_match_the_records() -> None:
    original = truth()
    assert truth_counts(original) == {
        "chains": len(original.chains),
        "competing": len(original.competing),
        "contradictions": len(original.contradictions),
        "clusters": len(original.clusters),
    }
