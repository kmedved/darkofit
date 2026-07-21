"""Outcome-blind contract tests for fused-lane dispatch infrastructure."""

from __future__ import annotations

import json
from copy import deepcopy

import numba
import numpy as np
import pytest

from benchmarks import fused_lane_dispatch_campaign as campaign
from benchmarks import freeze_fused_lane_dispatch_calibration as freezer
from benchmarks import run_fused_lane_dispatch as runner


def _digest(*parts):
    return campaign.json_sha256(list(parts))


def _calibration_rows():
    specs = campaign.calibration_specs()
    works = sorted({spec["scan_work"] for spec in specs})
    boundary = works[len(works) // 2]
    rows = []
    for spec in specs:
        ratio = 0.90 if spec["scan_work"] >= boundary else 1.01
        shared = (spec["rows"], spec["features"], spec["bins"])
        fingerprints = {
            "X": _digest(*shared, "X"),
            "grad": _digest(*shared, "grad"),
            "hess": _digest(*shared, spec["hessian"], "hess"),
            "n_bins": _digest(*shared, "n_bins"),
            "dataset_sha256": _digest(*shared, spec["hessian"], "dataset"),
        }
        repetitions = []
        for repeat in range(campaign.CALIBRATION_REPEATS):
            repetitions.append(
                {
                    "repeat": repeat,
                    "order": list(campaign.calibration_order(repeat)),
                    "fused_seconds": 1.0,
                    "unfused_seconds": ratio,
                    "exact": True,
                    "state_sha256": _digest(spec["coordinate_id"], "state"),
                    "prediction_sha256": _digest(
                        spec["coordinate_id"], "prediction"
                    ),
                    "tree_depth": 6,
                    "fused_level_count": 6,
                    "fused_opposite_level_count": 0,
                    "unfused_level_count": 6,
                    "unfused_opposite_level_count": 0,
                }
            )
        rows.append(
            {
                **spec,
                "seed": campaign.CALIBRATION_SEED,
                "warmups_per_lane": campaign.CALIBRATION_WARMUPS,
                "fingerprints": fingerprints,
                "runtime_before": {
                    "ceiling": spec["threads"],
                    "current": spec["threads"],
                },
                "runtime_after": {
                    "ceiling": spec["threads"],
                    "current": spec["threads"],
                },
                "thread_mask_restored": True,
                "repetitions": repetitions,
            }
        )
    return rows


def _validation_rows(threshold: int):
    rows = []
    for spec in campaign.validation_specs():
        expected_lane = (
            "unfused"
            if campaign.scan_work(
                spec["rows"], spec["features"], spec["threads"]
            )
            >= threshold
            else "fused"
        )
        work = campaign.scan_work(
            spec["rows"], spec["features"], spec["threads"]
        )
        for block, order in enumerate(campaign.VALIDATION_BLOCK_ORDERS):
            for arm in order:
                candidate = arm == "auto"
                ratio = 0.90 if candidate and expected_lane == "unfused" else 1.0
                resolved = expected_lane if candidate else "fused"
                reason = (
                    (
                        "at_or_above_threshold"
                        if expected_lane == "unfused"
                        else "below_threshold"
                    )
                    if candidate
                    else "user_forced_fused"
                )
                dispatch = {
                    "schema_version": 1,
                    "requested": arm,
                    "resolved": resolved,
                    "reason": reason,
                    "functional_eligible": True,
                    "automatic_eligible": True,
                    "threshold": threshold,
                    "scan_work": work,
                    "engaged": True,
                    "fused_level_count": 10 if resolved == "fused" else 0,
                    "unfused_level_count": 10 if resolved == "unfused" else 0,
                    "inputs": {
                        "platform_system": "Darwin",
                        "platform_machine": "arm64",
                        "logical_cpu_count": 14,
                        "n_rows": spec["rows"],
                        "n_active_features": spec["features"],
                        "n_threads": spec["threads"],
                        "depth": spec["depth"],
                        "max_realized_bins": spec["max_bins"] + 1,
                    },
                }
                hashes = {
                    "X": _digest(spec["cell_id"], "X"),
                    "y": _digest(spec["cell_id"], "y"),
                    "sample_weight": _digest(spec["cell_id"], "weight"),
                    "dataset_sha256": "a" * 64,
                }
                rows.append(
                    {
                        **spec,
                        "arm": arm,
                        "block": block,
                        "threshold": threshold,
                        "seed": campaign.VALIDATION_SEED,
                        "fingerprints": hashes,
                        "dataset_sha256": "a" * 64,
                        "projected_archive_sha256": "b" * 64,
                        "archive_sha256": _digest(
                            spec["cell_id"], block, arm, "archive"
                        ),
                        "prediction_sha256": "c" * 64,
                        "probability_sha256": (
                            "f" * 64
                            if spec["task"] == "binary_logloss"
                            else None
                        ),
                        "feature_importance_sha256": "d" * 64,
                        "safe_roundtrip_exact": True,
                        "resolved_lane": resolved,
                        "requested_lane": arm,
                        "dispatch_reason": reason,
                        "dispatch_metadata": dispatch,
                        "thread_mask_restored": True,
                        "thread_counts": {
                            name: spec["threads"]
                            for name in (
                                "ambient",
                                "after_warmup",
                                "after_fit",
                                "after_predict",
                                "after_roundtrip",
                            )
                        },
                        "runtime_before": {
                            "ceiling": spec["threads"],
                            "current": spec["threads"],
                        },
                        "runtime_after": {
                            "ceiling": spec["threads"],
                            "current": spec["threads"],
                        },
                        "selected_level_count": 10,
                        "opposite_level_count": 0,
                        "fit_seconds": 10.0 * ratio,
                        "tree_seconds": 5.0 * ratio,
                        "peak_rss_bytes": 1_000,
                    }
                )
    return rows


def test_frozen_grids_and_orders_match_design_contract():
    specs = campaign.calibration_specs()
    assert len(specs) == 30
    assert len({spec["coordinate_id"] for spec in specs}) == len(specs)
    assert {spec["rows"] for spec in specs} == set(campaign.CALIBRATION_ROWS)
    assert {
        (spec["features"], spec["threads"]) for spec in specs
    } == set(campaign.CALIBRATION_SHAPES)
    assert {spec["hessian"] for spec in specs} == set(
        campaign.CALIBRATION_HESSIANS
    )
    assert [campaign.calibration_order(index) for index in range(4)] == [
        ("fused", "unfused"),
        ("unfused", "fused"),
        ("fused", "unfused"),
        ("unfused", "fused"),
    ]
    assert campaign.validation_specs() == campaign.VALIDATION_CELLS
    assert campaign.VALIDATION_BLOCK_ORDERS == (
        ("fused", "auto"),
        ("auto", "fused"),
        ("fused", "auto"),
    )


def test_generators_are_deterministic_and_hessian_cases_are_distinct():
    base = {
        "coordinate_id": "test",
        "rows": 128,
        "features": 8,
        "threads": 4,
        "depth": 3,
        "bins": 16,
        "hessian": "unit",
        "scan_work": 256,
    }
    first, first_hashes = campaign.generate_calibration_case(base)
    second, second_hashes = campaign.generate_calibration_case(base)
    assert first_hashes == second_hashes
    assert all(np.array_equal(first[name], second[name]) for name in first)

    variable_spec = {**base, "hessian": "variable"}
    variable, variable_hashes = campaign.generate_calibration_case(variable_spec)
    np.testing.assert_array_equal(first["X"], variable["X"])
    np.testing.assert_array_equal(first["grad"], variable["grad"])
    assert np.all(variable["hess"] > 0.0)
    assert variable_hashes["hess"] != first_hashes["hess"]

    validation_spec = {
        "cell_id": "test_validation",
        "task": "weighted_rmse",
        "rows": 96,
        "features": 6,
        "threads": 4,
        "depth": 3,
        "max_bins": 16,
        "rounds": 2,
    }
    left, left_hashes = campaign.generate_validation_case(validation_spec)
    right, right_hashes = campaign.generate_validation_case(validation_spec)
    assert left_hashes == right_hashes
    assert all(
        (left[name] is right[name] is None)
        or np.array_equal(left[name], right[name])
        for name in left
    )


def test_calibration_analyzer_selects_mixed_qualifying_threshold():
    result = campaign.analyze_calibration(_calibration_rows())

    assert result["all_exact"] is True
    assert result["all_stable"] is True
    assert result["selected"]["threshold"] is not None
    assert result["selected"]["selected_fused_cells"] > 0
    assert result["selected"]["selected_unfused_cells"] > 0
    assert result["qualifies"] is True
    assert result["disposition"] == "freeze_threshold_before_validation"


def test_calibration_analyzer_closes_on_exactness_or_stability_failure():
    rows = _calibration_rows()
    rows[0]["repetitions"][0]["exact"] = False
    exactness = campaign.analyze_calibration(rows)
    assert exactness["qualifies"] is False
    assert exactness["disposition"] == "close_dispatch_campaign"

    rows = _calibration_rows()
    for repeat in range(3):
        rows[0]["repetitions"][repeat]["unfused_seconds"] = 2.0
    unstable = campaign.analyze_calibration(rows)
    assert unstable["all_stable"] is False
    assert unstable["qualifies"] is False

    rows = _calibration_rows()
    rows[0]["thread_mask_restored"] = False
    leaked = campaign.analyze_calibration(rows)
    assert leaked["all_exact"] is False
    assert leaked["qualifies"] is False


def test_validation_analyzer_applies_every_conjunctive_gate():
    threshold = 1_000_000
    rows = _validation_rows(threshold)
    result = campaign.analyze_validation(rows, threshold=threshold)

    assert result["all_exact"] is True
    assert result["all_stable"] is True
    assert result["mixed_dispatch"] is True
    assert result["qualifies"] is True

    broken = deepcopy(rows)
    broken[0]["prediction_sha256"] = "e" * 64
    failure = campaign.analyze_validation(broken, threshold=threshold)
    assert failure["all_exact"] is False
    assert failure["qualifies"] is False

    broken = deepcopy(rows)
    broken[0]["dispatch_metadata"]["threshold"] += 1
    failure = campaign.analyze_validation(broken, threshold=threshold)
    assert failure["all_exact"] is False
    assert failure["qualifies"] is False


def test_archive_projection_removes_only_dispatch_observability(tmp_path):
    header = {
        "params": {"depth": 4, "oblivious_kernel": "fused"},
        "auto_params": {
            "oblivious_kernel_dispatch": {"resolved": "fused"},
            "tree": {"depth": 4},
        },
        "wrapper": {"params": {"oblivious_kernel": "fused"}},
    }
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez_compressed(left, header=np.array(json.dumps(header)), values=np.arange(4))
    changed = deepcopy(header)
    changed["params"]["oblivious_kernel"] = "unfused"
    changed["auto_params"]["oblivious_kernel_dispatch"] = {
        "resolved": "unfused"
    }
    changed["wrapper"]["params"]["oblivious_kernel"] = "unfused"
    np.savez_compressed(
        right, header=np.array(json.dumps(changed)), values=np.arange(4)
    )

    assert campaign.canonical_archive_sha256(
        left, project_dispatch=True
    ) == campaign.canonical_archive_sha256(right, project_dispatch=True)
    assert campaign.canonical_archive_sha256(
        left, project_dispatch=False
    ) != campaign.canonical_archive_sha256(right, project_dispatch=False)


def test_worker_environment_is_fresh_and_thread_bounded(tmp_path):
    environment = runner.fixed_worker_environment(
        9,
        tmp_path,
    )
    assert environment["NUMBA_NUM_THREADS"] == "9"
    assert environment["OMP_THREAD_LIMIT"] == "9"
    assert environment["NUMBA_CACHE_DIR"] == str(tmp_path.resolve())
    assert "PYTHONPATH" not in environment


def test_freezer_binds_harness_runtime_and_keeps_execution_unauthorized(
    monkeypatch,
):
    source = "a" * 40
    monkeypatch.setattr(
        runner, "git_state", lambda *_args: {"head": source, "status": ""}
    )

    contract = freezer.build_contract()

    assert contract["source"] == source
    assert contract["contract_frozen"] is True
    assert contract["outcomes_opened"] is False
    assert contract["execution_authorized"] is False
    assert contract["generator"]["specs"] == list(campaign.calibration_specs())
    assert contract["execution"]["paired_repetitions"] == 7
    assert contract["downstream"]["calibration_execution_authorized"] is False
    assert contract["runtime"]["fingerprint"] == runner.runtime_fingerprint()
    assert set(contract["bound_files"]) == set(freezer.BOUND_PATHS)
    assert all(
        len(record["sha256"]) == 64
        for record in contract["bound_files"].values()
    )


def test_execution_requires_hash_bound_owner_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    contract = {
        "source": "a" * 40,
        "outputs": {"authorization": "authorization.json"},
    }
    missing = tmp_path / "authorization.json"
    with pytest.raises(RuntimeError, match="not owner-authorized"):
        runner.require_authorization(
            missing,
            contract_path=contract_path,
            contract=contract,
            phase="calibration",
        )

    authorization_path = missing
    authorization = {
        "schema_version": campaign.SCHEMA_VERSION,
        "campaign": campaign.CAMPAIGN_NAME,
        "phase": "calibration",
        "execution_authorized": True,
        "execution_contract_sha256": campaign.file_sha256(contract_path),
        "source": "a" * 40,
        "owner_decision": "explicit test authority",
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    assert runner.require_authorization(
        authorization_path,
        contract_path=contract_path,
        contract=contract,
        phase="calibration",
    ) == authorization

    copied = tmp_path / "copied-authorization.json"
    copied.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the contract"):
        runner.require_authorization(
            copied,
            contract_path=contract_path,
            contract=contract,
            phase="calibration",
        )


def test_formal_paths_are_exact_and_cannot_be_renamed_to_rerun(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    contract = {
        "outputs": {
            "raw": "benchmarks/raw.json",
            "terminal": "benchmarks/raw_terminal.json",
            "analysis": "benchmarks/analysis.json",
        }
    }
    assert runner._require_declared_path(
        tmp_path / "benchmarks" / "raw.json", contract, "raw"
    ) == (tmp_path / "benchmarks" / "raw.json").resolve()
    with pytest.raises(RuntimeError, match="does not match the contract"):
        runner._require_declared_path(
            tmp_path / "benchmarks" / "raw_rerun.json", contract, "raw"
        )


def test_execution_identity_refuses_existing_result_or_terminal(tmp_path):
    output = tmp_path / "raw.json"
    assert runner._assert_fresh_output(output) == output
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already used"):
        runner._assert_fresh_output(output)
    output.unlink()
    runner._terminal_path(output).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already used"):
        runner._assert_fresh_output(output)


def test_small_noncampaign_worker_proves_both_counters_and_exactness(monkeypatch):
    source = "a" * 40
    threads = int(numba.get_num_threads())
    if threads <= 2:
        pytest.skip("fused calibration invariant requires at least three threads")
    spec = {
        "coordinate_id": "invariant_only_not_a_campaign_cell",
        "rows": 192,
        "features": 8,
        "threads": threads,
        "depth": 3,
        "bins": 16,
        "hessian": "unit",
        "scan_work": campaign.scan_work(192, 8, threads),
    }
    state = {"head": source, "status": ""}
    monkeypatch.setattr(runner, "_activate_source", lambda *_args: state)
    monkeypatch.setattr(runner, "git_state", lambda *_args: state)
    monkeypatch.setattr(
        runner,
        "assert_worker_environment",
        lambda _threads: {
            "ceiling": threads,
            "current": threads,
            "threading_layer": "test",
            "environment": {},
        },
    )

    result = runner.calibration_worker(
        spec, source=source, source_root=runner.ROOT
    )

    assert result["thread_mask_restored"] is True
    assert len(result["repetitions"]) == campaign.CALIBRATION_REPEATS
    for repetition in result["repetitions"]:
        assert repetition["exact"] is True
        assert repetition["fused_level_count"] > 0
        assert repetition["unfused_level_count"] > 0
        assert repetition["fused_opposite_level_count"] == 0
        assert repetition["unfused_opposite_level_count"] == 0
