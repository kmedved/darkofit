"""Hostile and estimator tests for the minimal CTR23 confirmation analyzer."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import analyze_ctr23_minimal_confirmation as analysis


def _outer_rows(
    *,
    a10_over_m: float = 0.98,
    a10_over_d: float = 0.99,
) -> list[dict]:
    rows = []
    for dataset, task_id, repeat, fold, sample, arm in sorted(
        analysis.campaign.expected_grid(),
        key=lambda key: (key[1], key[3], key[5]),
    ):
        a10 = a10_over_m
        values = {
            "A10": a10,
            "M": 1.0,
            "D": a10 / a10_over_d,
            "C": 1.0,
        }
        rows.append(
            {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "sample": sample,
                "arm": arm,
                "test_rmse": values[arm],
                "val_rmse": values[arm],
                "source": analysis.campaign.expected_result_relative_path(
                    dataset, task_id, repeat, fold, sample, arm
                ),
                "num_cpus": 18,
                "num_cpus_child": 18,
                "num_gpus": 0,
                "num_gpus_child": 0,
            }
        )
    assert len(rows) == 90
    return rows


def _child_rows() -> list[dict]:
    rows = []
    for dataset, task_id, repeat, fold, sample, arm, child_fold in sorted(
        analysis.campaign.expected_child_grid(),
        key=lambda key: (key[1], key[3], key[5], key[6]),
    ):
        rows.append(
            {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "sample": sample,
                "arm": arm,
                "child_fold": child_fold,
                "source": analysis.campaign.expected_result_relative_path(
                    dataset, task_id, repeat, fold, sample, arm
                ),
                "num_cpus": 18,
                "num_gpus": 0,
                "iterations_requested": 1_000 if arm == "D" else 10_000,
                "iterations_attempted": 100,
                "rounds_completed": 100,
                "rounds_retained": 90,
                "best_iteration": 90,
                "resolved_learning_rate": (
                    0.05 if arm == "C" else (0.03 if arm == "D" else 0.1)
                ),
                "requested_tree_mode": (
                    "auto" if arm == "A10" else ("catboost" if arm == "D" else None)
                ),
                "deadline_hit": False,
                "deadline_is_soft": True,
                "time_callback_hit": False,
                "time_callback_instance_count": (
                    2 if arm == "M" else (1 if arm == "C" else 0)
                ),
                "time_callback_call_count": (
                    2 if arm == "M" else (1 if arm == "C" else 0)
                ),
                "stop_reason": "early_stopping" if arm in {"A10", "D"} else None,
                "selected_tree_mode": "catboost" if arm in {"A10", "D"} else None,
                "selected_lane": (
                    "boosting"
                    if arm in {"A10", "D"}
                    else ("constant" if arm == "M" else "cpu")
                ),
                **{
                    field: (
                        {
                            "candidate_count": 3,
                            "fitted_candidate_count": 3,
                            "candidate_order": ["catboost", "lightgbm", "hybrid"],
                            "selected_candidate_index": 0,
                            "candidates": [
                                {
                                    "candidate_index": index,
                                    "tree_mode": mode,
                                    "fitted": True,
                                    "validation_rmse": 0.9 + 0.1 * index,
                                    "deadline_hit": False,
                                    "stop_reason": "early_stopping",
                                }
                                for index, mode in enumerate(
                                    ("catboost", "lightgbm", "hybrid")
                                )
                            ],
                        }
                        if arm == "A10"
                        else None
                    )
                    for field in ("candidate_metadata", "tree_mode_selection")
                },
            }
        )
    assert len(rows) == 720
    return rows


def _paired(*, a10_over_m: float = 0.98, a10_over_d: float = 0.99):
    splits = analysis.pair_outer_rows(
        _outer_rows(a10_over_m=a10_over_m, a10_over_d=a10_over_d)
    )
    children = analysis.pair_child_rows(_child_rows())
    return splits, children


def _payload_shape_rows(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    return [{field: row.get(field) for field in fields} for row in rows]


def test_runner_exports_exact_frozen_grid_counts():
    grid = analysis._expected_grid()
    child_grid = analysis._expected_child_grid()

    assert len(grid) == 90
    assert len(child_grid) == 720
    assert {key[3] for key in grid} == {0, 1, 2}
    assert sum(key[-1] == "C" for key in grid) == 9
    assert all(key[3] == 0 for key in grid if key[-1] == "C")


def test_constant_primary_and_guardrail_ratios_are_exact_and_pass():
    splits, children = _paired(a10_over_m=0.98, a10_over_d=0.99)

    summary, per_dataset = analysis.analyze(splits, children)

    assert summary["contrasts"]["a10_over_m"]["test_rmse"]["ratio"] == pytest.approx(0.98)
    assert summary["contrasts"]["a10_over_d"]["test_rmse"]["ratio"] == pytest.approx(0.99)
    assert summary["primary_interval"]["upper_95"] == pytest.approx(0.98)
    assert summary["product_guardrail_interval"][
        "upper_95_simultaneous_max_regret"
    ] == pytest.approx(0.99)
    assert summary["confirmation_passed"] is True
    assert summary["decision"] == "confirmation_passed_clean_stop"
    assert len(per_dataset) == 45


def test_primary_gate_is_strict_at_one_and_stops_cleanly():
    splits, children = _paired(a10_over_m=1.0, a10_over_d=0.99)

    summary, _ = analysis.analyze(splits, children)

    assert summary["primary_interval"]["upper_95"] == pytest.approx(1.0)
    assert summary["gates"][
        "a10_over_m_one_sided_95_upper_strictly_below_1"
    ] is False
    assert summary["confirmation_passed"] is False
    assert summary["decision"] == "confirmation_not_established_clean_stop"
    assert "do not add folds" in summary["terminal_policy"]


def test_product_guardrail_uses_simultaneous_max_and_flags_task():
    rows = _outer_rows(a10_over_m=0.98, a10_over_d=0.99)
    flagged_task = min(row["task_id"] for row in rows)
    for row in rows:
        if row["task_id"] == flagged_task and row["arm"] == "D":
            row["test_rmse"] = 0.98 / 1.03
            row["val_rmse"] = 0.98 / 1.03
    splits = analysis.pair_outer_rows(rows)
    children = analysis.pair_child_rows(_child_rows())

    summary, _ = analysis.analyze(splits, children)

    assert summary["product_guardrail_interval"][
        "upper_95_simultaneous_max_regret"
    ] == pytest.approx(1.03)
    assert summary["gates"][
        "a10_over_d_simultaneous_max_regret_95_upper_at_most_1_02"
    ] is False
    assert [item["task_id"] for item in summary["product_task_point_flags"]] == [
        flagged_task
    ]
    assert summary["confirmation_passed"] is False


def test_equal_task_geometric_estimator_does_not_row_weight():
    rows = _outer_rows(a10_over_m=1.0, a10_over_d=0.99)
    favored_task = min(row["task_id"] for row in rows)
    for row in rows:
        if row["task_id"] == favored_task and row["arm"] == "A10":
            row["test_rmse"] = 0.5
            row["val_rmse"] = 0.5
    splits = analysis.pair_outer_rows(rows)
    children = analysis.pair_child_rows(_child_rows())

    summary, _ = analysis.analyze(splits, children)

    assert summary["contrasts"]["a10_over_m"]["test_rmse"]["ratio"] == pytest.approx(
        0.5 ** (1.0 / 9.0)
    )


def test_bootstraps_are_byte_deterministic_and_use_frozen_seeds():
    splits, children = _paired()

    first, rows_first = analysis.analyze(splits, children)
    second, rows_second = analysis.analyze(deepcopy(splits), deepcopy(children))

    assert analysis._canonical_json(first) == analysis._canonical_json(second)
    assert analysis._canonical_json(rows_first) == analysis._canonical_json(rows_second)
    assert first["primary_interval"]["seed"] == 20260719
    assert first["product_guardrail_interval"]["seed"] == 20260720
    assert first["contrasts"]["a10_over_c"]["test_rmse"][
        "descriptive_interval"
    ]["seed"] == 20260721
    assert first["primary_interval"]["quantile_method"] == "higher"


def test_catboost_is_single_fold_descriptive_and_cannot_change_decision():
    baseline, children = _paired()
    changed = deepcopy(baseline)
    for row in changed:
        if row["fold"] == 0:
            row["C_test_rmse"] *= 100.0
            row["C_val_rmse"] *= 100.0
            row["a10_over_c_test_rmse_ratio"] /= 100.0
            row["a10_over_c_val_rmse_ratio"] /= 100.0
            row["m_over_c_test_rmse_ratio"] /= 100.0
            row["m_over_c_val_rmse_ratio"] /= 100.0
            row["d_over_c_test_rmse_ratio"] /= 100.0
            row["d_over_c_val_rmse_ratio"] /= 100.0

    first, _ = analysis.analyze(baseline, children)
    second, per_dataset = analysis.analyze(changed, children)

    assert first["decision"] == second["decision"]
    assert first["gates"] == second["gates"]
    assert all(
        row["coordinate_count"] == 1
        for row in per_dataset
        if row["contrast"].endswith("over_c")
    )


@pytest.mark.parametrize("metric,bad", [("test_rmse", 0.0), ("val_rmse", math.inf)])
def test_pairing_rejects_nonpositive_or_nonfinite_rmse(metric, bad):
    rows = _outer_rows()
    rows[0][metric] = bad

    with pytest.raises(RuntimeError, match="finite and strictly positive"):
        analysis.pair_outer_rows(rows)


def test_pairing_rejects_missing_duplicate_and_misplaced_catboost_rows():
    rows = _outer_rows()
    with pytest.raises(RuntimeError, match="90 outer rows"):
        analysis.pair_outer_rows(rows[:-1])

    duplicate = deepcopy(rows)
    duplicate[-1] = deepcopy(duplicate[0])
    with pytest.raises(RuntimeError, match="duplicate|frozen 90-job grid"):
        analysis.pair_outer_rows(duplicate)

    c_index = next(index for index, row in enumerate(rows) if row["arm"] == "C")
    rows[c_index]["fold"] = 1
    with pytest.raises(RuntimeError, match="frozen 90-job grid"):
        analysis.pair_outer_rows(rows)


def test_safe_payload_validation_rejects_missing_a10_candidate():
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == "A10")
    target[analysis._candidate_metadata_field()]["candidate_order"] = [
        "catboost",
        "lightgbm",
    ]

    with pytest.raises(RuntimeError, match="all frozen candidates"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


def test_safe_payload_rejects_result_paths_swapped_between_coordinates():
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    first, second = outer[:2]
    first["source"], second["source"] = second["source"], first["source"]

    with pytest.raises(RuntimeError, match="canonical result path"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda metadata: metadata["candidates"][1].__setitem__(
                "validation_rmse", math.nan
            ),
            "finite and nonnegative",
        ),
        (
            lambda metadata: metadata["candidates"][1].__setitem__("fitted", False),
            "fitted state",
        ),
        (
            lambda metadata: metadata["candidates"][1].__setitem__(
                "deadline_hit", True
            ),
            "fitted state",
        ),
        (
            lambda metadata: metadata.__setitem__("selected_candidate_index", 2),
            "first validation argmin",
        ),
        (
            lambda metadata: (
                metadata["candidates"][0].__setitem__("validation_rmse", 1.0),
                metadata["candidates"][1].__setitem__("validation_rmse", 1.0),
                metadata.__setitem__("selected_candidate_index", 1),
            ),
            "first validation argmin",
        ),
    ],
)
def test_safe_payload_rejects_unproven_a10_candidate_states(mutation, pattern):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == "A10")
    mutation(target[analysis._candidate_metadata_field()])

    with pytest.raises(RuntimeError, match=pattern):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


def test_safe_payload_accepts_zero_candidate_validation_rmse():
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == "A10")
    target[analysis._candidate_metadata_field()]["candidates"][0][
        "validation_rmse"
    ] = 0.0

    validated_outer, validated_children = analysis._validate_payload_rows(
        {"outer_rows": outer, "child_rows": children}, artifacts
    )

    assert len(validated_outer) == 90
    assert len(validated_children) == 720


@pytest.mark.parametrize(
    "arm,instances,calls",
    [("A10", 1, 1), ("D", 1, 1), ("M", 2, 1), ("C", 0, 0)],
)
def test_safe_payload_rejects_incomplete_time_callback_coverage(
    arm, instances, calls
):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == arm)
    target["time_callback_instance_count"] = instances
    target["time_callback_call_count"] = calls

    with pytest.raises(RuntimeError, match="callback coverage"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


@pytest.mark.parametrize(
    "field,value",
    [("selected_tree_mode", "hybrid"), ("stop_reason", "iteration_limit")],
)
def test_safe_payload_binds_selected_candidate_back_to_top_level(field, value):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == "A10")
    target[field] = value
    if field == "stop_reason":
        target["iterations_attempted"] = target["iterations_requested"]
        target["rounds_completed"] = target["iterations_requested"]

    with pytest.raises(RuntimeError, match="all frozen candidates"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda row: row.__setitem__("deadline_hit", True), "deadline"),
        (lambda row: row.__setitem__("time_callback_hit", True), "time-limit"),
        (
            lambda row: row.__setitem__("time_callback_call_count", -1),
            "callback|integer",
        ),
        (lambda row: row.__setitem__("stop_reason", "time_limit"), "time-limit"),
        (lambda row: row.__setitem__("num_cpus", 17), "resource"),
    ],
)
def test_safe_payload_validation_rejects_child_failure_states(mutation, pattern):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    mutation(children[0])

    with pytest.raises(RuntimeError, match=pattern):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


def test_safe_payload_rejects_inconsistent_round_causality():
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    children[0]["best_iteration"] = children[0]["rounds_retained"] - 1

    with pytest.raises(RuntimeError, match="round counters"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


@pytest.mark.parametrize(
    "arm,reason,updates,pattern",
    [
        ("A10", "iteration_limit", {}, "iteration_limit"),
        ("D", "no_split", {}, "no_split"),
        (
            "M",
            "early_stopping",
            {"rounds_retained": 100, "best_iteration": 100},
            "ChimeraBoost",
        ),
        ("C", "early_stopping", {}, "CatBoost"),
    ],
)
def test_safe_payload_rejects_stop_reasons_not_proven_by_adapter_counters(
    arm, reason, updates, pattern
):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == arm)
    target["stop_reason"] = reason
    target.update(updates)

    with pytest.raises(RuntimeError, match=pattern):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


@pytest.mark.parametrize(
    "arm,field,value",
    [
        ("D", "iterations_requested", 10_000),
        ("D", "requested_tree_mode", "auto"),
        ("M", "resolved_learning_rate", 0.05),
        ("M", "selected_lane", "boosting"),
        ("C", "resolved_learning_rate", 0.1),
        ("C", "selected_tree_mode", "catboost"),
        ("C", "selected_lane", None),
    ],
)
def test_safe_payload_rejects_non_a10_behavior_drift(arm, field, value):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    artifacts = {row["source"]: {} for row in outer}
    target = next(row for row in children if row["arm"] == arm)
    target[field] = value

    with pytest.raises(RuntimeError, match=f"{arm} child fitted behavior changed"):
        analysis._validate_payload_rows(
            {"outer_rows": outer, "child_rows": children}, artifacts
        )


def test_detached_exports_carry_quality_only_disposition_and_no_timings():
    splits, children = _paired()
    summary, per_dataset = analysis.analyze(splits, children)

    outputs = analysis._build_output_payloads(
        splits, per_dataset, children, summary
    )

    for key in ("split_csv", "dataset_csv", "child_csv"):
        text = outputs[key].decode("utf-8")
        header = text.splitlines()[0]
        assert header.startswith(
            "execution_mode,swap_policy,timing_admissible,performance_evidence_disposition"
        )
        assert "train_time" not in header
        assert "infer_time" not in header
        assert "memory" not in header
        assert "quality_only_swap_in" in text
        assert analysis.PERFORMANCE_EVIDENCE_DISPOSITION in text


def test_analyzer_source_never_imports_or_calls_pickle():
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "pickle" not in imported
    assert not any(name.startswith("pickle") for name in called)
    assert "results.pkl" in source  # It is documented and authenticated as opaque bytes.


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(RuntimeError, match="strict finite JSON"):
        analysis._strict_json_loads(b'{"x":1,"x":2}', "fixture")
    with pytest.raises(RuntimeError, match="strict finite JSON"):
        analysis._strict_json_loads(b'{"x":NaN}', "fixture")


def test_artifact_reader_rejects_hash_size_and_symlink_attacks(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    payload = b"opaque-not-a-pickle"
    result = root / "results.pkl"
    result.write_bytes(payload)
    metadata = {"sha256": analysis._sha256(payload), "size_bytes": len(payload)}

    assert analysis._artifact_bytes(root, "results.pkl", metadata, "result") == payload
    with pytest.raises(RuntimeError, match="hash or size"):
        analysis._artifact_bytes(
            root, "results.pkl", {**metadata, "sha256": "0" * 64}, "result"
        )
    with pytest.raises(RuntimeError, match="hash or size"):
        analysis._artifact_bytes(
            root, "results.pkl", {**metadata, "size_bytes": len(payload) + 1}, "result"
        )

    outside = tmp_path / "outside.pkl"
    outside.write_bytes(payload)
    result.unlink()
    result.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symbolic-link"):
        analysis._artifact_bytes(root, "results.pkl", metadata, "result")


def test_report_is_terminal_and_does_not_claim_catboost_parity():
    splits, children = _paired()
    summary, per_dataset = analysis.analyze(splits, children)

    report = analysis.render_report(summary, per_dataset)

    assert "descriptive and have no advancement gate" in report
    assert "do not add folds" in report
    assert "no lockbox" in summary["claim_scope"]


def _write_json(path: Path, value) -> bytes:
    payload = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _singleton_metadata(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _absolute_metadata(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    source = (tmp_path / "failed-concurrent").resolve()
    destination = (tmp_path / "fresh-sequential").resolve()
    source.mkdir()
    destination.mkdir()
    source_manifest = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND,
        "created_at_utc": "2026-07-15T12:00:00+00:00",
        "output_dir": str(source),
        "protocol_sha256": analysis.campaign.protocol_sha256(),
        "frozen_protocol_sha256": analysis.campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": analysis.campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": analysis.campaign.schedule_sha256(),
        "schedule": analysis.campaign.expected_schedule(),
        "expected_jobs": 90,
        "expected_child_fits": 720,
        "time_limit_seconds": 3600.0,
        "resolved_child_num_cpus": 18,
        "execution_mode": "concurrent",
        "swap_policy": "quality_only_swap_in",
        "timing_admissible": False,
        "source_freeze": {"frozen": True},
        "source": {"source": "same"},
        "runtime": {"runtime": "same"},
        "sequential_recovery": None,
    }
    source_manifest_path = source / analysis.campaign.MANIFEST_FILENAME
    source_manifest_bytes = _write_json(source_manifest_path, source_manifest)
    marker = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_invalid_attempt",
        "invalidated_at_utc": "2026-07-15T13:00:00+00:00",
        "execution_mode": "concurrent",
        "stage": "production",
        "reuse_allowed": False,
        "recovery_policy": "fresh_sequential_namespace_from_wave_zero_only",
        "manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "error_type": "RuntimeError",
        "error": "resource barrier invalidated the complete attempt",
    }
    marker_path = source / analysis.campaign.INVALID_ATTEMPT_FILENAME
    _write_json(marker_path, marker)
    record = {
        "source_output_dir": str(source),
        "invalid_attempt_artifact": _absolute_metadata(marker_path),
        "source_manifest_artifact": _absolute_metadata(source_manifest_path),
        "reuse_policy": "no_results_reused_fresh_wave_zero",
    }
    current_manifest = {
        **source_manifest,
        "created_at_utc": "2026-07-15T14:00:00+00:00",
        "output_dir": str(destination),
        "execution_mode": "sequential_recovery",
        "sequential_recovery": record,
    }
    return source, destination, current_manifest, record


def _refresh_recovery_record(source: Path, record: dict) -> None:
    marker_path = source / analysis.campaign.INVALID_ATTEMPT_FILENAME
    source_manifest_path = source / analysis.campaign.MANIFEST_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["manifest_sha256"] = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    _write_json(marker_path, marker)
    record["invalid_attempt_artifact"] = _absolute_metadata(marker_path)
    record["source_manifest_artifact"] = _absolute_metadata(source_manifest_path)


def _stub_runner_recovery_provenance(monkeypatch, manifest: dict) -> None:
    monkeypatch.setattr(
        analysis.campaign,
        "validate_source_freeze",
        lambda: manifest["source_freeze"],
    )
    monkeypatch.setattr(
        analysis.campaign,
        "collect_source_provenance",
        lambda output_dir=None: manifest["source"],
    )
    monkeypatch.setattr(
        analysis.campaign,
        "collect_runtime_provenance",
        lambda: manifest["runtime"],
    )


def _operational_artifacts_fixture(tmp_path: Path, monkeypatch):
    from benchmarks import tabarena_comparator_warmup

    root = (tmp_path / "operational-campaign").resolve()
    ready = [
        {
            "type": "ready",
            "slot": slot,
            "pid": 30_000 + slot,
            "child_cpus": analysis.campaign.EXPECTED_CHILD_CPUS,
            "start_method": "spawn",
            "scratch_root": str(root / "worker_scratch" / f"worker-{slot}"),
        }
        for slot in range(analysis.campaign.WORKER_COUNT)
    ]
    warmup_records = []
    preflight_ready = deepcopy(ready)
    for item in preflight_ready:
        item["scratch_root"] = str(
            root / ".preflight-nonreusable" / f"worker-{item['slot']}"
        )
    preflight_warmup = []
    probes = []
    probe_release = 100_000_000
    for item in ready:
        slot = item["slot"]
        payload = {
            "darkofit": {"engine": "darkofit", "slot": slot},
            "comparators": {"engine": "comparators", "slot": slot},
        }
        record = {
            "completed_at_utc": "2026-07-16T00:00:00+00:00",
            "pid": item["pid"],
            "worker_slot": slot,
            "warmup": payload,
        }
        warmup_records.append(deepcopy(record))
        preflight_warmup.append(deepcopy(record))
        probes.append(
            {
                "worker_slot": slot,
                "pid": item["pid"],
                "behavior_sha256": hashlib.sha256(
                    analysis.campaign._canonical_json(
                        analysis.campaign._synthetic_behavior_projection(payload)
                    )
                ).hexdigest(),
                "barrier_release_monotonic_ns": probe_release,
                "started_monotonic_ns": probe_release + 100 + slot * 100,
                "ended_monotonic_ns": probe_release + 5_000 + slot * 100,
            }
        )
    probe_starts = [item["started_monotonic_ns"] for item in probes]
    probe_ends = [item["ended_monotonic_ns"] for item in probes]
    preflight = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_preflight",
        "completed_at_utc": "2026-07-16T00:00:01+00:00",
        "status": "passed",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
        "worker_ready": preflight_ready,
        "worker_warmup": preflight_warmup,
        "ctr23_fit_count": 0,
        "synthetic_probes": probes,
        "start_skew_ns": max(probe_starts) - min(probe_starts),
        "overlap_ns": max(0, min(probe_ends) - max(probe_starts)),
        "worker_restarts": False,
        "failure_count": 0,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }
    artifacts = {}
    entries = []
    previous_end = 500_000_000
    for wave in analysis.campaign.expected_schedule():
        index = wave["wave_index"]
        release = previous_end + 1_000
        reports = []
        for job in wave["jobs"]:
            slot = job["worker_slot"]
            key = analysis.campaign._key_tuple(job["key"])
            relative = analysis.campaign.expected_result_relative_path(*key)
            digest = hashlib.sha256(relative.encode()).hexdigest()
            size = 1_000 + len(artifacts)
            artifacts[relative] = {"sha256": digest, "size_bytes": size}
            started = release + 100 + slot * 100
            ended = started + 5_000 + slot * 100
            reports.append(
                {
                    "type": "result",
                    "command_id": (
                        f"production-wave-{index}-{slot}-{100_000 + index * 2 + slot}"
                    ),
                    "status": "ok",
                    "slot": slot,
                    "pid": ready[slot]["pid"],
                    "key": deepcopy(job["key"]),
                    "result_root": str(root),
                    "result_path": str((root / relative).resolve()),
                    "result_count": 1,
                    "child_count": 8,
                    "deadline_hit": False,
                    "time_callback_hit_count": 0,
                    "a10_candidate_fit_count": 24 if key[-1] == "A10" else 0,
                    "behavior_sha256": "a" * 64,
                    "result_sha256": digest,
                    "result_size_bytes": size,
                    "process_peak_rss_bytes": 10_000_000 + slot,
                    "barrier_release_monotonic_ns": release,
                    "started_monotonic_ns": started,
                    "ended_monotonic_ns": ended,
                    "start_method": "spawn",
                }
            )
        starts = [item["started_monotonic_ns"] for item in reports]
        ends = [item["ended_monotonic_ns"] for item in reports]
        entries.append(
            {
                "wave_index": index,
                "jobs": deepcopy(wave["jobs"]),
                "reports": reports,
                "swap_out_delta": 0,
                "peak_combined_rss_fraction": 0.1,
                "start_skew_ns": max(starts) - min(starts),
                "overlap_ns": max(0, min(ends) - max(starts)),
                "wave_elapsed_ns": max(ends) - min(starts),
            }
        )
        previous_end = max(ends)
    concurrency = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": "concurrent",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
        "wave_count": analysis.campaign.EXPECTED_WAVES,
        "entries": entries,
        "failure_count": 0,
        "worker_restart_count": 0,
        "recovery_mixing_count": 0,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }
    operational = {
        "preflight_report_artifact": preflight,
        "concurrency_history_artifact": concurrency,
        "warmup_history_artifact": {
            "schema_version": 1,
            "kind": analysis.campaign.CAMPAIGN_KIND + "_warmup_history",
            "execution_mode": "concurrent",
            "worker_ready": ready,
            "worker_warmup": warmup_records,
        },
    }
    manifest = {
        "output_dir": str(root),
        "execution_mode": "concurrent",
        "sequential_recovery": None,
    }
    attestation = {
        "execution_mode": "concurrent",
        "result_artifacts": artifacts,
    }
    monkeypatch.setattr(
        analysis.campaign.hardened.screen,
        "_validate_followon_warmup_history",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabarena_comparator_warmup,
        "validate_comparator_warmup_history",
        lambda *args, **kwargs: None,
    )
    return root, operational, manifest, attestation


def test_sequential_recovery_replays_exact_failed_concurrent_source(
    tmp_path, monkeypatch
):
    _, destination, manifest, _ = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)

    analysis._validate_sequential_recovery(
        manifest, campaign_root=destination
    )


def test_sequential_recovery_rejects_authenticated_foreign_source_manifest(
    tmp_path, monkeypatch
):
    source, destination, manifest, record = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)
    source_manifest_path = source / analysis.campaign.MANIFEST_FILENAME
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["source"] = {"source": "foreign"}
    _write_json(source_manifest_path, source_manifest)
    _refresh_recovery_record(source, record)

    with pytest.raises(RuntimeError, match="foreign or changed"):
        analysis._validate_sequential_recovery(
            manifest, campaign_root=destination
        )


@pytest.mark.parametrize("mutation,pattern", [
    (
        lambda source, destination, record: record["invalid_attempt_artifact"].__setitem__(
            "sha256", "0" * 64
        ),
        "hash or size",
    ),
    (
        lambda source, destination, record: (
            source / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
        ).write_text("completed"),
        "completed campaign",
    ),
    (
        lambda source, destination, record: record.__setitem__(
            "source_output_dir", str(destination)
        ),
        "overlaps",
    ),
])
def test_sequential_recovery_rejects_hostile_artifacts(
    tmp_path, monkeypatch, mutation, pattern
):
    source, destination, manifest, record = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)
    mutation(source, destination, record)

    with pytest.raises(RuntimeError, match=pattern):
        analysis._validate_sequential_recovery(
            manifest, campaign_root=destination
        )


def test_operational_validator_accepts_recomputed_concurrent_evidence(
    tmp_path, monkeypatch
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )

    analysis.campaign.validate_operational_artifacts_for_analysis(
        operational,
        manifest=manifest,
        attestation=attestation,
        output_dir=root,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["concurrency_history_artifact"]["entries"][0].__setitem__(
            "start_skew_ns",
            value["concurrency_history_artifact"]["entries"][0]["start_skew_ns"]
            + 1,
        ),
        lambda value: value["concurrency_history_artifact"]["entries"][0].__setitem__(
            "wave_elapsed_ns",
            value["concurrency_history_artifact"]["entries"][0]["wave_elapsed_ns"]
            + 1,
        ),
        lambda value: value["concurrency_history_artifact"]["entries"][0][
            "reports"
        ][1].__setitem__(
            "barrier_release_monotonic_ns",
            value["concurrency_history_artifact"]["entries"][0]["reports"][1][
                "barrier_release_monotonic_ns"
            ]
            + 1,
        ),
        lambda value: value["preflight_report_artifact"].__setitem__(
            "overlap_ns", value["preflight_report_artifact"]["overlap_ns"] + 1
        ),
        lambda value: value["warmup_history_artifact"]["worker_warmup"].__setitem__(
            1,
            deepcopy(value["warmup_history_artifact"]["worker_warmup"][0]),
        ),
    ],
)
def test_operational_validator_rejects_detached_timing_and_warmup_mutations(
    tmp_path, monkeypatch, mutation
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    mutation(operational)

    with pytest.raises(RuntimeError, match="preflight|warmup|operational wave"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


def test_operational_validator_rejects_cross_wave_barrier_overlap(
    tmp_path, monkeypatch
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    entries = operational["concurrency_history_artifact"]["entries"]
    previous_end = max(
        report["ended_monotonic_ns"] for report in entries[0]["reports"]
    )
    original_release = entries[1]["reports"][0][
        "barrier_release_monotonic_ns"
    ]
    shift = previous_end - original_release
    for report in entries[1]["reports"]:
        report["barrier_release_monotonic_ns"] += shift
        report["started_monotonic_ns"] += shift
        report["ended_monotonic_ns"] += shift

    with pytest.raises(RuntimeError, match="precedes its release"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


def _campaign_fixture(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "campaign"
    root.mkdir()
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    result_artifacts = {}
    source_map = {}
    for row in outer:
        relative = row["source"]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"opaque-invalid-pickle-" + relative.encode()
        path.write_bytes(payload)
        result_artifacts[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        source_map[
            (row["dataset"], row["task_id"], row["repeat"], row["fold"], row["sample"], row["arm"])
        ] = relative
    for row in children:
        row["source"] = source_map[
            (row["dataset"], row["task_id"], row["repeat"], row["fold"], row["sample"], row["arm"])
        ]

    schedule_bytes = _write_json(
        root / analysis.campaign.SCHEDULE_FILENAME,
        analysis.campaign.expected_schedule(),
    )
    for name, value in (
        (analysis.campaign.PREFLIGHT_REPORT_FILENAME, {"status": "passed"}),
        (analysis.campaign.CONCURRENCY_HISTORY_FILENAME, {"waves": 45}),
        (analysis.campaign.WARMUP_HISTORY_FILENAME, {"workers": 2}),
    ):
        _write_json(root / name, value)

    source_freeze = {"verified": "registry"}
    source = {"verified": "sources"}
    runtime = {"verified": "runtime"}
    manifest = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND,
        "created_at_utc": "2026-07-15T12:00:00Z",
        "output_dir": str(root.resolve()),
        "protocol_sha256": analysis.campaign.protocol_sha256(),
        "frozen_protocol_sha256": analysis.campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": analysis.campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": analysis.campaign.schedule_sha256(),
        "schedule": analysis.campaign.expected_schedule(),
        "expected_jobs": 90,
        "expected_child_fits": 720,
        "time_limit_seconds": 3600.0,
        "resolved_child_num_cpus": 18,
        "execution_mode": "concurrent",
        "swap_policy": "quality_only_swap_in",
        "timing_admissible": False,
        "source_freeze": source_freeze,
        "source": source,
        "runtime": runtime,
        "sequential_recovery": None,
    }
    manifest_bytes = _write_json(root / analysis.campaign.MANIFEST_FILENAME, manifest)
    payload = {
        "schema_version": 1,
        "kind": analysis.campaign.PAYLOAD_KIND,
        "protocol_sha256": analysis.campaign.protocol_sha256(),
        "frozen_protocol_sha256": analysis.campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": analysis.campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": analysis.campaign.schedule_sha256(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_artifacts_sha256": analysis._sha256(
            analysis._canonical_json(result_artifacts)
        ),
        "swap_policy": "quality_only_swap_in",
        "timing_admissible": False,
        "outer_rows": outer,
        "child_rows": children,
    }
    _write_json(root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME, payload)
    validation = {
        "result_count": 90,
        "child_fit_count": 720,
        "failure_count": 0,
    }
    attestation = {
        "schema_version": 1,
        "kind": analysis.campaign.COMPLETION_KIND,
        "completed_at_utc": "2026-07-15T13:00:00Z",
        "pid": 123,
        "execution_mode": "concurrent",
        "swap_policy": "quality_only_swap_in",
        "timing_admissible": False,
        "protocol_sha256": analysis.campaign.protocol_sha256(),
        "frozen_protocol_sha256": analysis.campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": analysis.campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": analysis.campaign.schedule_sha256(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_count": 90,
        "expected_result_count": 90,
        "expected_child_fits": 720,
        "result_artifacts": result_artifacts,
        "analysis_payload_artifact": _singleton_metadata(
            root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME, root
        ),
        "schedule_artifact": {
            "path": analysis.campaign.SCHEDULE_FILENAME,
            "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
            "size_bytes": len(schedule_bytes),
        },
        "preflight_report_artifact": _singleton_metadata(
            root / analysis.campaign.PREFLIGHT_REPORT_FILENAME, root
        ),
        "concurrency_history_artifact": _singleton_metadata(
            root / analysis.campaign.CONCURRENCY_HISTORY_FILENAME, root
        ),
        "warmup_history_artifact": _singleton_metadata(
            root / analysis.campaign.WARMUP_HISTORY_FILENAME, root
        ),
        "validation": validation,
    }
    _write_json(root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME, attestation)

    monkeypatch.setattr(analysis.campaign, "validate_source_freeze", lambda: source_freeze)
    monkeypatch.setattr(
        analysis.campaign,
        "collect_source_provenance",
        lambda output_dir=None: source,
    )
    monkeypatch.setattr(analysis.campaign, "collect_runtime_provenance", lambda: runtime)
    monkeypatch.setattr(
        analysis.campaign,
        "validate_operational_artifacts_for_analysis",
        lambda value, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        analysis.campaign,
        "validate_completion_for_analysis",
        lambda value, **kwargs: None,
        raising=False,
    )
    return root


def test_full_integrity_accepts_opaque_nonpickle_results_and_exact_hash_chain(
    tmp_path, monkeypatch
):
    root = _campaign_fixture(tmp_path, monkeypatch)

    manifest, attestation, payload, digests, protected = (
        analysis.verify_campaign_integrity(root)
    )

    assert manifest["expected_jobs"] == attestation["result_count"] == 90
    assert len(payload["outer_rows"]) == 90
    assert len(payload["child_rows"]) == 720
    assert len(digests["manifest_sha256"]) == 64
    assert len(protected) == 97


@pytest.mark.parametrize("target", ["result", "payload", "schedule", "manifest"])
def test_full_integrity_rejects_hash_chain_tampering(tmp_path, monkeypatch, target):
    root = _campaign_fixture(tmp_path, monkeypatch)
    if target == "result":
        path = next((root / "experiments").rglob("results.pkl"))
    elif target == "payload":
        path = root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME
    elif target == "schedule":
        path = root / analysis.campaign.SCHEDULE_FILENAME
    else:
        path = root / analysis.campaign.MANIFEST_FILENAME
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="hash|attestation|manifest"):
        analysis.verify_campaign_integrity(root)


def test_full_integrity_rejects_source_freeze_revalidation_failure(
    tmp_path, monkeypatch
):
    root = _campaign_fixture(tmp_path, monkeypatch)

    def reject(*_args, **_kwargs):
        raise RuntimeError("source artifact hash changed")

    monkeypatch.setattr(analysis.campaign, "validate_source_freeze", reject)
    with pytest.raises(RuntimeError, match="source artifact hash changed"):
        analysis.verify_campaign_integrity(root)


def test_main_atomically_publishes_byte_deterministic_outputs(tmp_path, monkeypatch):
    root = _campaign_fixture(tmp_path, monkeypatch)

    assert analysis.main(["--input-dir", str(root)]) == 0
    first = {
        name: (root / name).read_bytes() for name in analysis.OUTPUT_NAMES
    }
    assert analysis.main(["--input-dir", str(root)]) == 0
    second = {
        name: (root / name).read_bytes() for name in analysis.OUTPUT_NAMES
    }

    assert first == second
    summary = json.loads(first["summary.json"])
    assert summary["integrity"]["raw_results_deserialized_by_analyzer"] is False
    assert summary["timing_admissible"] is False
