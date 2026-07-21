"""Contract and invariant tests for the Wave-3 B-archive feasibility screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import analyze_barchive_v1 as analyzer
from benchmarks import run_barchive_v1 as runner
from darkofit import DarkoRegressor
from darkofit.sklearn_api import _fit_private_ensemble_v3


ROOT = Path(__file__).resolve().parents[1]


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime() -> dict:
    environment = runner.paired._expected_environment(runner.THREADS)
    environment["NUMBA_CACHE_DIR"] = "/tmp/darkofit-barchive-test-cache"
    return {
        "ceiling": runner.THREADS,
        "current": runner.THREADS,
        "threading_layer": "omp",
        "environment": environment,
    }


def _archive_record(label: str, *, task: str, size: int) -> dict:
    return {
        "archive_bytes": size,
        "archive_sha256": _hex(f"{label}/archive"),
        "prediction_sha256": _hex(f"{label}/prediction"),
        "probability_sha256": (
            None if task == "regression" else _hex(f"{label}/probability")
        ),
        "feature_schema_sha256": _hex(f"{label}/schema"),
        "metadata_sha256": _hex(f"{label}/metadata"),
        "checks": {
            "prediction_exact": True,
            "probability_exact": True,
            "feature_schema_exact": True,
            "metadata_exact": True,
            "resave_bytes_exact": True,
        },
    }


def _row(spec: dict, provenance: str, *, effective_ratio: float) -> dict:
    case_id = spec["case_id"]
    single_bytes = 100
    combined_bytes = 550
    eligible = provenance == "numeric_target_free"
    effective_bytes = int(effective_ratio * single_bytes) if eligible else 550
    canonical = {
        "numeric_target_free_provenance": eligible,
        "array_schema_identical": eligible,
        "arrays_byte_identical": eligible,
        "headers_byte_identical": eligible,
        "eligible": eligible,
        "array_names": (sorted(runner.ALLOWED_CANONICAL_ARRAYS) if eligible else []),
        "header_fields": (
            sorted(runner.ALLOWED_CANONICAL_HEADER_FIELDS) if eligible else []
        ),
        "simulated_archive_bytes": effective_bytes if eligible else None,
    }
    components = [
        {
            "name": name,
            "section": "preprocessor",
            "present_in_all_members": True,
            "byte_identical_across_members": True,
            "member_fingerprints": [
                {
                    "sha256": _hex(f"{case_id}/{name}"),
                    "npy_bytes": 100,
                    "standalone_npz_bytes": 120,
                    "dtype": "int64",
                    "shape": [2],
                }
                for _ in range(runner.m3b.MEMBERS)
            ],
        }
        for name in sorted(runner.ALLOWED_CANONICAL_ARRAYS)
    ]
    return {
        "case_id": case_id,
        "domain": spec["domain"],
        "task": spec["task"],
        "case_sha256": _hex(f"{case_id}/case"),
        "dataset_sha256": _hex(f"{case_id}/dataset"),
        "split_sha256": _hex(f"{case_id}/split"),
        "weight_sha256": _hex(f"{case_id}/weight"),
        "implementation_path": f"/pinned/darkofit/sklearn_api.py",
        "shared_preprocessing": provenance,
        "single": _archive_record(
            f"{case_id}/single", task=spec["task"], size=single_bytes
        ),
        "combined": _archive_record(
            f"{case_id}/combined", task=spec["task"], size=combined_bytes
        ),
        "current_archive_to_single": 5.5,
        "canonical_preprocessor": canonical,
        "optimistic_all_exact_entries": {
            "array_names": sorted(runner.ALLOWED_CANONICAL_ARRAYS),
            "simulated_archive_bytes": 300,
            "includes_out_of_scope_sections": not eligible,
        },
        "components": components,
        "effective_candidate_archive_bytes": effective_bytes,
        "effective_candidate_archive_to_single": effective_bytes / single_bytes,
        "effective_uses_only_canonical_preprocessor": eligible,
        "runtime_before": _runtime(),
        "runtime_after": _runtime(),
    }


def _synthetic_evidence(tmp_path: Path, *, effective_ratio: float):
    specs = list(runner.m3b.case_specs())
    provenance = runner.expected_shared_preprocessing()
    fingerprints = {
        spec["case_id"]: {
            "case_sha256": _hex(f"{spec['case_id']}/case"),
            "dataset_sha256": _hex(f"{spec['case_id']}/dataset"),
            "split_sha256": _hex(f"{spec['case_id']}/split"),
            "weight_sha256": _hex(f"{spec['case_id']}/weight"),
        }
        for spec in specs
    }
    panel_cache = {
        "contract_path": ".cache/panel.csv",
        "bytes": 10,
        "sha256": _hex("panel"),
    }
    contract = {
        "cases": specs,
        "case_fingerprints": fingerprints,
        "expected_shared_preprocessing": provenance,
        "panel_cache": panel_cache,
        "decision_rules": runner.decision_rules(),
        "claims": runner.claim_contract(),
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    raw = {
        "schema_version": runner.SCHEMA_VERSION,
        "name": runner.CONTRACT_NAME,
        "status": "complete",
        "created_at": "2026-07-21T00:00:00+00:00",
        "contract_path": "contract.json",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "source_state": {
            "path": "/pinned",
            "head": runner.MODEL_SOURCE_HEAD,
            "status": "",
        },
        "harness_state": {
            "path": str(tmp_path),
            "head": "a" * 40,
            "status": "",
        },
        "runtime_versions": runner.FROZEN_RUNTIME,
        "panel_cache": panel_cache,
        "case_fingerprints": fingerprints,
        "rows": [
            _row(
                spec,
                provenance[spec["case_id"]],
                effective_ratio=effective_ratio,
            )
            for spec in specs
        ],
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    return contract_path, raw_path, contract, raw


def _install_synthetic_context(monkeypatch, tmp_path, contract):
    monkeypatch.setattr(analyzer, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(analyzer, "_assert_harness_provenance", lambda *_args: None)


def test_barchive_contract_keeps_m3b_boundary_and_exact_case_portfolio():
    rules = runner.decision_rules()
    provenance = runner.expected_shared_preprocessing()
    lineage = runner.m3b_r3_lineage()

    assert rules["median_effective_archive_to_single_at_most"] == 4.0
    assert rules["expected_case_count"] == len(provenance) == 13
    assert runner.execution_contract()["frozen_runtime"] == runner.FROZEN_RUNTIME
    assert sum(value == "numeric_target_free" for value in provenance.values()) == 11
    assert sum(value == "member_local" for value in provenance.values()) == 2
    assert lineage["lineage_valid"] is True
    assert lineage["combined_beats_matched_single_cases"] == 13
    assert lineage["combined_median_archive_to_single"] == pytest.approx(
        5.534767493867151
    )
    assert lineage["serialization_authorized"] is False
    assert lineage["sports_primary_scope"].startswith("player-disjoint")
    assert runner.claim_contract()["public_or_default_change_authorized"] is False


def test_barchive_roundtrip_record_checks_private_ensemble_exactness(tmp_path):
    rng = np.random.default_rng(20260721)
    X = rng.normal(size=(140, 8))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.2, size=140)
    model = _fit_private_ensemble_v3(
        DarkoRegressor(
            iterations=5,
            depth=3,
            early_stopping_rounds=2,
            random_state=4,
            n_ensembles=3,
            diagnostic_warnings="never",
        ),
        X,
        y,
        sampling="without_replacement",
        sampling_unit="rows",
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
    )

    record = runner._roundtrip_record(model, X[:25], tmp_path / "ensemble.npz")

    assert record["archive_bytes"] > 0
    assert all(record["checks"].values())
    assert record["probability_sha256"] is None


@pytest.mark.parametrize(("effective_ratio", "advances"), [(3.5, True), (4.5, False)])
def test_barchive_frozen_median_decides_without_using_member_local_deltas(
    monkeypatch, tmp_path, effective_ratio, advances
):
    contract_path, raw_path, contract, _raw = _synthetic_evidence(
        tmp_path, effective_ratio=effective_ratio
    )
    _install_synthetic_context(monkeypatch, tmp_path, contract)

    result = analyzer.build_result(raw_path, contract_path)

    assert result["case_count"] == 13
    assert result["numeric_target_free_case_count"] == 11
    assert result["member_local_case_count"] == 2
    assert result["canonical_serializer_prototype_authorized"] is advances
    assert result["serializer_retention_authorized"] is False
    assert result["fused_lane_dispatch_nominated_next"] is (not advances)
    expected_median = effective_ratio
    assert result["gate"]["median_effective_archive_to_single"] == pytest.approx(
        expected_median
    )


def test_barchive_validator_rejects_out_of_scope_effective_size(monkeypatch, tmp_path):
    contract_path, raw_path, contract, raw = _synthetic_evidence(
        tmp_path, effective_ratio=3.5
    )
    member_local = next(
        row for row in raw["rows"] if row["shared_preprocessing"] == "member_local"
    )
    member_local["effective_candidate_archive_bytes"] = 300
    member_local["effective_candidate_archive_to_single"] = 3.0
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    _install_synthetic_context(monkeypatch, tmp_path, contract)

    with pytest.raises(RuntimeError, match="effective size drifted"):
        analyzer.validate_raw(raw_path, contract_path)


def test_barchive_validator_rejects_incomplete_canonical_array_set(
    monkeypatch, tmp_path
):
    contract_path, raw_path, contract, raw = _synthetic_evidence(
        tmp_path, effective_ratio=3.5
    )
    numeric = next(
        row
        for row in raw["rows"]
        if row["shared_preprocessing"] == "numeric_target_free"
    )
    numeric["canonical_preprocessor"]["array_names"].pop()
    numeric["optimistic_all_exact_entries"]["includes_out_of_scope_sections"] = True
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    _install_synthetic_context(monkeypatch, tmp_path, contract)

    with pytest.raises(RuntimeError, match="canonical section is incomplete"):
        analyzer.validate_raw(raw_path, contract_path)


def test_barchive_result_pair_is_create_only_and_cleans_first_on_second_failure(
    monkeypatch, tmp_path
):
    output = tmp_path / "result.json"
    note = tmp_path / "result.md"
    monkeypatch.setattr(analyzer, "RESULT_PATH", output)
    monkeypatch.setattr(analyzer, "NOTE_PATH", note)
    calls = 0
    original = analyzer.paired.write_create_only

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected note publication failure")
        original(path, payload)

    monkeypatch.setattr(analyzer.paired, "write_create_only", fail_second)
    with pytest.raises(OSError, match="injected"):
        analyzer._publish_pair(
            {
                "disposition": "close",
                "case_count": 13,
                "numeric_target_free_case_count": 11,
                "member_local_case_count": 2,
                "canonical_serializer_prototype_authorized": False,
                "gate": {
                    "median_current_archive_to_single": 5.5,
                    "median_effective_archive_to_single": 4.5,
                    "median_effective_archive_to_single_at_most": 4.0,
                },
            },
            output,
            note,
        )
    assert not output.exists()
    assert not note.exists()


@pytest.mark.campaign
@pytest.mark.skipif(
    not runner.CONTRACT_PATH.exists(), reason="B-archive contract not frozen yet"
)
def test_barchive_frozen_contract_is_exact():
    contract = runner.load_contract()

    assert contract["sources"]["darkofit"] == runner.MODEL_SOURCE_HEAD
    assert contract["m3b_r3_lineage"] == runner.m3b_r3_lineage()
    assert contract["expected_shared_preprocessing"] == (
        runner.expected_shared_preprocessing()
    )


@pytest.mark.campaign
@pytest.mark.skipif(
    not analyzer.RESULT_PATH.exists(), reason="B-archive result not published yet"
)
def test_barchive_published_result_regenerates_exactly():
    stored = json.loads(analyzer.RESULT_PATH.read_text(encoding="utf-8"))
    regenerated = analyzer.build_result()
    regenerated["analyzed_at"] = stored["analyzed_at"]

    assert regenerated == stored
    assert analyzer.NOTE_PATH.read_text(encoding="utf-8") == (
        analyzer.render_note(stored)
    )
    assert stored["m3b_r3_amended"] is False
    assert stored["serializer_retention_authorized"] is False
