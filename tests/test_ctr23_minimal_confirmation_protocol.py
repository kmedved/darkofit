"""Source-freeze tests for the minimal unseen CTR23 confirmation.

These tests inspect only committed metadata, protocol text, and runner grid
construction.  They never load a CTR23 result, prediction, target, or score.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
BENCHMARKS = ROOT / "benchmarks"
PROTOCOL_PATH = BENCHMARKS / "ctr23_minimal_confirmation_protocol.md"
SUITE_PATH = BENCHMARKS / "ctr23_suite_snapshot.json"
REGISTRY_PATH = BENCHMARKS / "ctr23_contamination_registry.json"
PARTITION_PATH = BENCHMARKS / "ctr23_partition.json"
COORDINATE_PATH = BENCHMARKS / "ctr23_minimal_confirmation_coordinates.json"
RUNNER_PATH = BENCHMARKS / "run_ctr23_minimal_confirmation.py"
CTR23_ADAPTER_PATH = BENCHMARKS / "tabarena_ctr23_adapters.py"
EXPECTED_CTR23_ADAPTER_SHA256 = (
    "9ab74de39cdcffaa5f92031519eeb68c4087c58b09aa4c1401293324d925cab3"
)

EXPECTED_TASK_IDS = (
    361236,
    361251,
    361252,
    361258,
    361268,
    361269,
    361619,
    361622,
    361623,
)
EXPECTED_LOCKBOX_TASK_IDS = (
    361247,
    361253,
    361254,
    361261,
    361264,
    361272,
    361616,
    361617,
    361618,
)
EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_COORDINATE_MANIFEST_SHA256 = (
    "6cef3b771c20440c9dad6b737797f50650d84217ee99cf8fc6fcfcbd85829c0b"
)

EXPECTED_SEMANTIC_HASHES = {
    "suite_snapshot_sha256": (
        "95bb2bb5d9c65ea21cb7642151bedb831ed67712bae28166a0bddc64670f0364"
    ),
    "contamination_registry_sha256": (
        "9bda6f8b94b71575fa8275ed724ab80976c93555d898fbec8f474fcc78c6639d"
    ),
    "partition_sha256": (
        "24e060ed3626fed23967294138d5768c3d9e7241f4ed06cf9b8180d512e81ee8"
    ),
    "registry_bundle_sha256": (
        "21980c6ddaf3f5b70e866fbcc6c59c04a98b666687234147a7cedcc0b8271516"
    ),
    "manual_evidence_sha256": (
        "66529d85f9f1caea2d04784ae6666704cb6c3b5e56e06460066a482c5358ce75"
    ),
    "declarations_sha256": (
        "bd20852afdacdbd55d20fd4adfe7331c760f651061a912fa8424f5d77675dcc9"
    ),
}
EXPECTED_FILE_HASHES = {
    "ctr23_suite_snapshot.json": (
        "ce676c7dde7576aee8c5c8f074aa76fa004746d0ed4381a89b12335282e1d33c"
    ),
    "ctr23_contamination_registry.json": (
        "002dba27713237c33f8b09de91160ad54b4f7d4c6e44b76ba5ba0e0e98c0adc2"
    ),
    "ctr23_partition.json": (
        "125e70cbe49241fb4fda1ed3f79b504f5d8d0b20b8f85d665c1cb49aa5c5fab6"
    ),
}

EXPECTED_TASK_METADATA = {
    361236: ("auction_verification", 2_043, 7, 2, False, 10),
    361251: ("grid_stability", 10_000, 12, 0, False, 10),
    361252: ("video_transcoding", 68_784, 18, 2, False, 10),
    361258: ("kin8nm", 8_192, 8, 0, False, 10),
    361268: ("fps_benchmark", 24_624, 43, 14, True, 10),
    361269: ("health_insurance", 22_272, 11, 7, False, 10),
    361619: ("student_performance_por", 649, 30, 17, False, 100),
    361622: ("cars", 804, 17, 0, False, 100),
    361623: ("space_ga", 3_107, 6, 0, False, 10),
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_coordinate_manifest() -> list[dict]:
    suite = _load(SUITE_PATH)
    task_by_id = {
        task["openml_task_id"]: task for task in suite["ctr23_tasks"]
    }
    manifest = []
    for task_id in EXPECTED_TASK_IDS:
        task = task_by_id[task_id]
        coordinates = {
            (item["repeat"], item["fold"], item["sample"]): item
            for item in task["official_splits"]["coordinates"]
        }
        for fold in EXPECTED_FOLDS:
            coordinate = coordinates[(0, fold, 0)]
            manifest.append(
                {
                    "dataset": task["normalized_name"],
                    "fold": fold,
                    "openml_task_id": task_id,
                    "repeat": 0,
                    "sample": 0,
                    "test_index_sha256": coordinate["test_index_sha256"],
                    "test_size": coordinate["test_size"],
                    "train_index_sha256": coordinate["train_index_sha256"],
                    "train_size": coordinate["train_size"],
                }
            )
    return manifest


def test_protocol_freezes_scope_arms_gates_and_terminal_state():
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    normalized_protocol = " ".join(protocol.split())

    required_literals = (
        "source-frozen confirmation protocol",
        "2026-07-16 forward-only harness amendment",
        "c3e47d5697826793c097214bb60ce68fd713c443",
        "not a replay verifier",
        "do not alter the terminal A10 decision",
        "`r0f0`, `r0f1`, and `r0f2`",
        "27 A10 + 27 M + 27 D + 9 C = 90 outer jobs",
        "90 outer jobs x 8 AutoGluon bag children = 720 selected children",
        "numpy.random.PCG64(20260719)",
        "numpy.random.PCG64(20260720)",
        "numpy.random.PCG64(20260721)",
        "exactly 10,000",
        "method=\"higher\"",
        "strictly less than 1.000",
        "less than or equal to 1.020",
        "`G[A10/M] <= 0.995`",
        "report-only target",
        "`R[d; A10/D] > 1.010`",
        "CatBoost is descriptive only",
        "`quality_only_swap_in`",
        "cross-binds every measured dispatch",
        "safe payload and completion attestation duplicate",
        "exactly four hard-coded repository-relative JSON paths",
        "Registry keys supplied by a document never confer filesystem authority",
        "exactly these seven strict JSON files",
        "`run_manifest.json`",
        "`completion_attestation.json`",
        "`analysis_payload.json`",
        "`wave_schedule.json`",
        "`preflight_report.json`",
        "`concurrency_history.json`",
        "`warmup_history.json`",
        "existence check for source `completion_attestation.json`",
        "runner-authored root statement and has no external signature",
        "forbidden_no_stat_enumerate_open_hash_or_decode",
        "runner-attested result metadata manifest",
        "45 waves",
        "The preflight is synthetic-only and non-reusable",
        "`ctr23_fit_count=0`",
        "`ctr23_time_callback_audit`",
        "`time_callback_hit=false`",
        "there is no in-place resume",
        "Fresh sequential recovery",
        "No result from that namespace may enter analysis",
        "must be Git-ignored before source provenance is collected",
        "Mandatory terminal state",
        "opening or running the nine lockbox tasks",
        "promoting an accuracy preset or changing the DarkoFit default",
        EXPECTED_COORDINATE_MANIFEST_SHA256,
    )
    for literal in required_literals:
        assert " ".join(literal.split()) in normalized_protocol

    for task_id, (name, *_metadata) in EXPECTED_TASK_METADATA.items():
        assert f"| {task_id} | `{name}` |" in protocol


def test_protocol_literal_binds_ctr23_callback_adapter_bytes():
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert _sha256_file(CTR23_ADAPTER_PATH) == EXPECTED_CTR23_ADAPTER_SHA256
    assert "`tabarena_ctr23_adapters.py` SHA-256" in protocol
    assert EXPECTED_CTR23_ADAPTER_SHA256 in protocol


def test_registry_artifacts_match_frozen_byte_and_semantic_hashes():
    for name, expected in EXPECTED_FILE_HASHES.items():
        assert _sha256_file(BENCHMARKS / name) == expected

    suite = _load(SUITE_PATH)
    registry = _load(REGISTRY_PATH)
    partition = _load(PARTITION_PATH)

    assert suite["schema_version"] == 3
    assert registry["schema_version"] == 3
    assert partition["schema_version"] == 3
    assert {
        suite["algorithm_version"],
        registry["algorithm_version"],
        partition["algorithm_version"],
    } == {"ctr23-contamination-registry-v3"}
    assert {
        suite["ctr23_suite_id"],
        registry["ctr23_suite_id"],
        partition["ctr23_suite_id"],
    } == {353}

    assert (
        suite["suite_snapshot_sha256"]
        == registry["suite_snapshot_sha256"]
        == partition["suite_snapshot_sha256"]
        == EXPECTED_SEMANTIC_HASHES["suite_snapshot_sha256"]
    )
    assert (
        registry["contamination_registry_sha256"]
        == partition["contamination_registry_sha256"]
        == EXPECTED_SEMANTIC_HASHES["contamination_registry_sha256"]
    )
    assert (
        partition["partition_sha256"]
        == EXPECTED_SEMANTIC_HASHES["partition_sha256"]
    )
    assert (
        partition["registry_bundle_sha256"]
        == EXPECTED_SEMANTIC_HASHES["registry_bundle_sha256"]
    )
    assert (
        suite["manual_evidence_sha256"]
        == registry["manual_evidence_sha256"]
        == partition["manual_evidence_sha256"]
        == EXPECTED_SEMANTIC_HASHES["manual_evidence_sha256"]
    )
    assert (
        suite["declarations_sha256"]
        == registry["declarations_sha256"]
        == EXPECTED_SEMANTIC_HASHES["declarations_sha256"]
    )


def test_confirmation_allocation_is_exact_eligible_and_disjoint_from_lockbox():
    registry = _load(REGISTRY_PATH)
    partition = _load(PARTITION_PATH)
    confirmation = tuple(partition["confirmation_task_ids"])
    lockbox = tuple(partition["lockbox_task_ids"])

    assert confirmation == EXPECTED_TASK_IDS
    assert lockbox == EXPECTED_LOCKBOX_TASK_IDS
    assert len(confirmation) == len(lockbox) == 9
    assert set(confirmation).isdisjoint(lockbox)
    assert set(confirmation) | set(lockbox) == set(
        registry["eligible_task_ids"]
    )
    assert partition["confirmation_coordinate_count"] == 270
    assert partition["lockbox_coordinate_count"] == 270
    assert partition["split_diagnostics"]["hard_constraints"] == {
        "lineage_clusters_are_atomic": True,
        "panel_task_count_difference_at_most": 1,
        "per_resampling_regime_task_count_difference_at_most": 1,
        "target_information_used": False,
    }

    entries = {item["openml_task_id"]: item for item in registry["tasks"]}
    for task_id in confirmation:
        entry = entries[task_id]
        assert entry["status"] == "eligible"
        assert entry["exclusion_reasons"] == []
        assert entry["ambiguous_matches"] == []
        assert entry["exposure_scope"] is None


def test_selected_official_coordinates_and_hash_manifest_are_exact():
    suite = _load(SUITE_PATH)
    assert suite["task_count"] == 35
    assert suite["official_coordinate_count"] == 800
    task_by_id = {
        task["openml_task_id"]: task for task in suite["ctr23_tasks"]
    }

    for task_id, expected in EXPECTED_TASK_METADATA.items():
        name, rows, features, categories, missing, coordinate_count = expected
        task = task_by_id[task_id]
        fingerprint = task["fingerprint"]
        official = task["official_splits"]
        assert (
            task["normalized_name"],
            fingerprint["n_rows"],
            fingerprint["n_features"],
            fingerprint["categorical_feature_count"],
            fingerprint["has_missing_features"],
            official["coordinate_count"],
        ) == (name, rows, features, categories, missing, coordinate_count)
        assert official["dimensions"]["folds"] == 10
        assert official["dimensions"]["samples"] == 1
        assert official["dimensions"]["repeats"] == (
            10 if task_id in {361619, 361622} else 1
        )
        assert official["integrity"] == {
            "coordinate_keys_complete": True,
            "crossvalidation_test_folds_partition_rows": True,
            "train_test_cover_all_rows_once": True,
            "train_test_disjoint": True,
        }

    manifest = _expected_coordinate_manifest()
    assert len(manifest) == 27
    assert [(row["repeat"], row["fold"], row["sample"]) for row in manifest] == [
        (0, fold, 0)
        for _task_id in EXPECTED_TASK_IDS
        for fold in EXPECTED_FOLDS
    ]
    assert [row["openml_task_id"] for row in manifest] == [
        task_id
        for task_id in EXPECTED_TASK_IDS
        for _fold in EXPECTED_FOLDS
    ]
    for row in manifest:
        assert row["train_size"] > 0 and row["test_size"] > 0
        for field in ("train_index_sha256", "test_index_sha256"):
            digest = row[field]
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")
    assert _canonical_sha256(manifest) == EXPECTED_COORDINATE_MANIFEST_SHA256

    coordinate_file = _load(COORDINATE_PATH)
    assert coordinate_file["schema_version"] == 1
    assert coordinate_file["kind"] == (
        "darkofit_ctr23_minimal_confirmation_coordinates"
    )
    assert coordinate_file["ctr23_suite_id"] == 353
    assert coordinate_file["coordinate_policy"] == {
        "repeat": 0,
        "folds": [0, 1, 2],
        "sample": 0,
        "split_indices": ["r0f0", "r0f1", "r0f2"],
    }
    assert coordinate_file["expected_task_count"] == 9
    assert coordinate_file["expected_coordinate_count"] == 27
    assert coordinate_file["source_artifacts"] == {
        "benchmarks/ctr23_suite_snapshot.json": {
            "file_sha256": EXPECTED_FILE_HASHES[
                "ctr23_suite_snapshot.json"
            ],
            "declared_suite_snapshot_sha256": EXPECTED_SEMANTIC_HASHES[
                "suite_snapshot_sha256"
            ],
        },
        "benchmarks/ctr23_contamination_registry.json": {
            "file_sha256": EXPECTED_FILE_HASHES[
                "ctr23_contamination_registry.json"
            ],
            "declared_registry_sha256": EXPECTED_SEMANTIC_HASHES[
                "contamination_registry_sha256"
            ],
        },
        "benchmarks/ctr23_partition.json": {
            "file_sha256": EXPECTED_FILE_HASHES["ctr23_partition.json"],
            "declared_partition_sha256": EXPECTED_SEMANTIC_HASHES[
                "partition_sha256"
            ],
        },
        "benchmarks/ctr23_manual_evidence_catalog.json": {
            "file_sha256": (
                "abb3dc9c875db5119b2bfbbf23d49cbb1cc2176c39d0a8237d5e01ba533941f3"
            ),
            "declared_manual_evidence_sha256": EXPECTED_SEMANTIC_HASHES[
                "manual_evidence_sha256"
            ],
        },
    }
    flattened = [
        {
            "dataset": task["dataset_name"],
            "fold": coordinate["fold"],
            "openml_task_id": task["task_id"],
            "repeat": coordinate["repeat"],
            "sample": coordinate["sample"],
            "test_index_sha256": coordinate["test_index_sha256"],
            "test_size": coordinate["test_size"],
            "train_index_sha256": coordinate["train_index_sha256"],
            "train_size": coordinate["train_size"],
        }
        for task in coordinate_file["tasks"]
        for coordinate in task["coordinates"]
    ]
    assert flattened == manifest
    assert _canonical_sha256(flattened) == EXPECTED_COORDINATE_MANIFEST_SHA256

    assert manifest[0] == {
        "dataset": "auction_verification",
        "fold": 0,
        "openml_task_id": 361236,
        "repeat": 0,
        "sample": 0,
        "test_index_sha256": (
            "73e3fd48dbf7fbe1cbba1dd2bba97cf24e87fcbf8ebf492569abe2fa272860ea"
        ),
        "test_size": 205,
        "train_index_sha256": (
            "714764e3a3ce5a03fd077cd9752f22adfd79d0af6d1054ecbe7bbc30fab29a43"
        ),
        "train_size": 1838,
    }
    assert manifest[-1] == {
        "dataset": "space_ga",
        "fold": 2,
        "openml_task_id": 361623,
        "repeat": 0,
        "sample": 0,
        "test_index_sha256": (
            "f7f893f2041f96436d54498d4d38785f9a8ce73e96c8d99e3dd1764820567bda"
        ),
        "test_size": 311,
        "train_index_sha256": (
            "12eb13d27e6a5fcb4c5901ba97ce3a602a6ae6bbbea8a5a41917704528025b92"
        ),
        "train_size": 2796,
    }


def test_runner_grid_and_coordinate_manifest_match_protocol_when_present():
    if not RUNNER_PATH.is_file():
        pytest.skip("CTR23 runner has not landed yet")

    runner = importlib.import_module(
        "benchmarks.run_ctr23_minimal_confirmation"
    )
    expected_manifest = _expected_coordinate_manifest()
    assert runner.COORDINATE_MANIFEST_SHA256 == (
        EXPECTED_COORDINATE_MANIFEST_SHA256
    )
    assert runner.expected_coordinate_manifest() == expected_manifest
    assert _canonical_sha256(runner.expected_coordinate_manifest()) == (
        EXPECTED_COORDINATE_MANIFEST_SHA256
    )

    assert runner.EXPECTED_JOBS == 90
    assert runner.EXPECTED_CHILD_FITS == 720
    grid = list(runner.expected_grid())
    assert len(grid) == len(set(grid)) == 90
    assert Counter(key[-1] for key in grid) == {
        "A10": 27,
        "M": 27,
        "D": 27,
        "C": 9,
    }

    expected_core_coordinates = {
        (
            row["dataset"],
            row["openml_task_id"],
            row["repeat"],
            row["fold"],
            row["sample"],
        )
        for row in expected_manifest
    }
    for arm in ("A10", "M", "D"):
        assert {key[:-1] for key in grid if key[-1] == arm} == (
            expected_core_coordinates
        )
    assert {key[:-1] for key in grid if key[-1] == "C"} == {
        key for key in expected_core_coordinates if key[3] == 0
    }

    children = list(runner.expected_child_grid())
    assert len(children) == len(set(children)) == 720
    assert {key[:-1] for key in children} == set(grid)
    assert Counter(key[-1] for key in children) == {index: 90 for index in range(8)}

    assert runner.protocol_sha256() == _sha256_file(PROTOCOL_PATH)


def test_analyzer_bootstrap_contract_matches_protocol_when_runner_is_present():
    if not RUNNER_PATH.is_file():
        pytest.skip("CTR23 runner has not landed yet")

    analyzer = importlib.import_module(
        "benchmarks.analyze_ctr23_minimal_confirmation"
    )
    assert analyzer.BOOTSTRAP_DRAWS == 10_000
    assert analyzer.PRIMARY_BOOTSTRAP_SEED == 20_260_719
    assert analyzer.GUARDRAIL_BOOTSTRAP_SEED == 20_260_720
    assert analyzer.CATBOOST_BOOTSTRAP_SEED == 20_260_721
    assert analyzer.QUANTILE_METHOD == "higher"
    assert analyzer.PRIMARY_RATIO_LIMIT == 1.0
    assert analyzer.PRODUCT_MAX_REGRET_LIMIT == 1.02
    assert analyzer.PRODUCT_TASK_FLAG_LIMIT == 1.01
    assert analyzer.DESIRED_PRIMARY_POINT == 0.995
