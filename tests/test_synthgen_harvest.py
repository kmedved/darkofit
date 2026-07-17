import json

import pytest

from benchmarks.synthgen import harvest_metadata as harvest


def _entry(dataset_id, name, *, classes=0):
    qualities = {
        "NumberOfInstances": 1000,
        "NumberOfFeatures": 8,
        "NumberOfClasses": classes,
        "NumberOfSymbolicFeatures": 0,
        "NumberOfMissingValues": 0,
        "MajorityClassSize": 0,
        "MaxNominalAttDistinctValues": 0,
    }
    return {
        "did": dataset_id,
        "name": name,
        "quality": [
            {"name": key, "value": str(value)}
            for key, value in qualities.items()
        ],
    }


def test_ctr23_exclusion_manifest_is_complete_and_hashed():
    dataset_ids, names, provenance = harvest.load_ctr23_exclusions()

    assert len(dataset_ids) == len(names) == 35
    assert provenance["dataset_count"] == 35
    assert len(provenance["suite_snapshot_sha256"]) == 64
    assert len(provenance["identity_sha256"]) == 64


def test_distill_excludes_ctr23_and_tabarena_before_reduction():
    ctr23_ids, ctr23_names, _ = harvest.load_ctr23_exclusions()
    entries = [
        _entry(dataset_id, f"ctr-{dataset_id}")
        for dataset_id in sorted(ctr23_ids)
    ]
    entries.extend(
        [
            _entry(900001, "tabarena-member"),
            _entry(900002, "admissible-public-data"),
        ]
    )

    rows, observed = harvest.distill(
        entries,
        {900001},
        ctr23_ids,
        ctr23_names,
        set(),
        set(),
        set(),
    )

    assert len(rows) == 1
    assert rows[0][:3] == [1000, 7, 0]
    assert observed["drop_counts"]["ctr23"] == 35
    assert observed["drop_counts"]["tabarena"] == 1
    assert set(observed["excluded_ctr23_dataset_ids"]) == ctr23_ids
    assert observed["excluded_tabarena_dataset_ids"] == [900001]


def test_ctr23_presence_check_fails_closed():
    ctr23_ids, _, _ = harvest.load_ctr23_exclusions()
    entries = [
        _entry(dataset_id, f"ctr-{dataset_id}")
        for dataset_id in sorted(ctr23_ids)[1:]
    ]

    with pytest.raises(RuntimeError, match="absent from OpenML active listing"):
        harvest.validate_ctr23_presence(entries, ctr23_ids)


def test_snapshot_preserves_provenance():
    provenance = {
        "raw_cache_sha256": "a" * 64,
        "ctr23": {"identity_sha256": "b" * 64},
    }
    snapshot = harvest.snapshot(
        [[1000, 7, 0, 0.0, 0, 0.0, 0.0, 0, 0]],
        10,
        "test source",
        provenance,
    )

    assert snapshot["version"] == 2
    assert snapshot["provenance"] == provenance
    json.dumps(snapshot, allow_nan=False)
