"""Hostile and estimator tests for the minimal CTR23 confirmation analyzer."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks import analyze_ctr23_minimal_confirmation as analysis


def _install_raw_path_tripwire(monkeypatch) -> list[Path]:
    touched: list[Path] = []
    for method_name in (
        "resolve",
        "stat",
        "lstat",
        "exists",
        "is_file",
        "open",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
    ):
        original = getattr(Path, method_name)

        def guarded(
            path,
            *args,
            _original=original,
            _method_name=method_name,
            **kwargs,
        ):
            folded = {part.casefold() for part in path.parts}
            if "experiments" in folded or "results.pkl" in folded:
                touched.append(path)
                raise AssertionError(f"raw path {_method_name} touched {path}")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)
    return touched


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
    assert "_observed_result_paths" not in source
    assert "os.walk" not in source
    assert "results.pkl" in source  # The prohibition is explicit in the module contract.


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(RuntimeError, match="strict finite JSON"):
        analysis._strict_json_loads(b'{"x":1,"x":2}', "fixture")
    with pytest.raises(RuntimeError, match="strict finite JSON"):
        analysis._strict_json_loads(b'{"x":NaN}', "fixture")


def test_campaign_json_reader_enforces_allowlist_hash_size_and_symlinks(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    payload = b'{}\n'
    result = root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME
    result.write_bytes(payload)
    metadata = {"sha256": analysis._sha256(payload), "size_bytes": len(payload)}

    assert analysis._campaign_json_artifact_bytes(
        root, analysis.campaign.ANALYSIS_PAYLOAD_FILENAME, metadata, "payload"
    ) == payload
    with pytest.raises(RuntimeError, match="allowlist"):
        analysis._campaign_json_artifact_bytes(
            root, "results.pkl", metadata, "raw result"
        )
    with pytest.raises(RuntimeError, match="hash or size"):
        analysis._campaign_json_artifact_bytes(
            root,
            analysis.campaign.ANALYSIS_PAYLOAD_FILENAME,
            {**metadata, "sha256": "0" * 64},
            "payload",
        )
    with pytest.raises(RuntimeError, match="hash or size"):
        analysis._campaign_json_artifact_bytes(
            root,
            analysis.campaign.ANALYSIS_PAYLOAD_FILENAME,
            {**metadata, "size_bytes": len(payload) + 1},
            "payload",
        )

    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    result.unlink()
    result.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symbolic-link"):
        analysis._campaign_json_artifact_bytes(
            root, analysis.campaign.ANALYSIS_PAYLOAD_FILENAME, metadata, "payload"
        )


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


def _recovery_failure_swap_telemetry() -> dict:
    session = {
        "sample_count": 2,
        "samples": [
            {
                "monotonic_ns": 100,
                "swap_in_bytes": 10,
                "swap_out_bytes": 20,
            },
            {
                "monotonic_ns": 200,
                "swap_in_bytes": 15,
                "swap_out_bytes": 20,
            },
        ],
        "swap_in_delta": 5,
        "swap_out_delta": 0,
    }
    return {
        "capture_status": "captured",
        "teardown_confirmed": True,
        "post_teardown_sample_recorded": True,
        "worker_session_swap_telemetry": session,
        "swap_in_bytes": 5,
        "swap_out_bytes": 0,
        "diagnostic": None,
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
        "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_invalid_attempt",
        "invalidated_at_utc": "2026-07-15T13:00:00+00:00",
        "execution_mode": "concurrent",
        "stage": "production",
        "reuse_allowed": False,
        "recovery_policy": "fresh_sequential_namespace_from_wave_zero_only",
        "manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "worker_shutdown_confirmed": True,
        "failure_swap_telemetry": _recovery_failure_swap_telemetry(),
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
    preflight_session = {
        "sample_count": 6,
        "samples": [
            {"monotonic_ns": 1, "swap_in_bytes": 100, "swap_out_bytes": 200},
            {"monotonic_ns": 2, "swap_in_bytes": 101, "swap_out_bytes": 200},
            {"monotonic_ns": 3, "swap_in_bytes": 102, "swap_out_bytes": 200},
            {"monotonic_ns": 90_000_000, "swap_in_bytes": 110, "swap_out_bytes": 200},
            {"monotonic_ns": 120_000_000, "swap_in_bytes": 125, "swap_out_bytes": 200},
            {"monotonic_ns": 130_000_000, "swap_in_bytes": 130, "swap_out_bytes": 200},
        ],
        "swap_in_delta": 30,
        "swap_out_delta": 0,
    }
    preflight_dispatch = {
        "sample_count": 2,
        "samples": [
            {"monotonic_ns": 95_000_000, "swap_in_bytes": 112, "swap_out_bytes": 200},
            {"monotonic_ns": 110_000_000, "swap_in_bytes": 120, "swap_out_bytes": 200},
        ],
        "swap_in_delta": 8,
        "swap_out_delta": 0,
        "barrier_release_monotonic_ns": probe_release,
    }
    preflight_measured = {
        "start_sample_index": 3,
        "end_sample_index": 4,
        "sample_count": 2,
        "swap_in_delta": 15,
        "swap_out_delta": 0,
        "dispatches": [
            {
                "label": "preflight-synthetic-probe",
                "sample_index": 4,
                "resource_first_monotonic_ns": 95_000_000,
                "resource_last_monotonic_ns": 110_000_000,
                "resource_first_swap_in_bytes": 112,
                "resource_last_swap_in_bytes": 120,
                "resource_first_swap_out_bytes": 200,
                "resource_last_swap_out_bytes": 200,
                "barrier_release_monotonic_ns": probe_release,
                "max_report_end_monotonic_ns": max(probe_ends),
            }
        ],
    }
    preflight = {
        "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
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
        "worker_session_swap_telemetry": preflight_session,
        "measured_phase_swap_window": preflight_measured,
        "synthetic_dispatch_telemetry": preflight_dispatch,
        "swap_in_bytes": 30,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }
    artifacts = {}
    entries = []
    previous_end = 500_000_000
    production_samples = [
        {"monotonic_ns": 400_000_000, "swap_in_bytes": 1_000, "swap_out_bytes": 2_000},
        {"monotonic_ns": 450_000_000, "swap_in_bytes": 1_001, "swap_out_bytes": 2_000},
        {"monotonic_ns": 500_000_000, "swap_in_bytes": 1_010, "swap_out_bytes": 2_000},
    ]
    measured_dispatches = []
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
        swap_start_index = 2 + index
        swap_end_index = swap_start_index + 1
        prior_swap_in = production_samples[-1]["swap_in_bytes"]
        resource_first_ns = release - 100
        resource_last_ns = max(ends) + 100
        checkpoint_ns = resource_last_ns + 100
        production_samples.append(
            {
                "monotonic_ns": checkpoint_ns,
                "swap_in_bytes": prior_swap_in + 3,
                "swap_out_bytes": 2_000,
            }
        )
        measured_dispatches.append(
            {
                "label": f"production-wave-{index}",
                "sample_index": swap_end_index,
                "resource_first_monotonic_ns": resource_first_ns,
                "resource_last_monotonic_ns": resource_last_ns,
                "resource_first_swap_in_bytes": prior_swap_in + 1,
                "resource_last_swap_in_bytes": prior_swap_in + 2,
                "resource_first_swap_out_bytes": 2_000,
                "resource_last_swap_out_bytes": 2_000,
                "barrier_release_monotonic_ns": release,
                "max_report_end_monotonic_ns": max(ends),
            }
        )
        entries.append(
            {
                "wave_index": index,
                "jobs": deepcopy(wave["jobs"]),
                "reports": reports,
                "swap_start_sample_index": swap_start_index,
                "swap_end_sample_index": swap_end_index,
                "swap_in_delta": 3,
                "swap_out_delta": 0,
                "peak_combined_rss_fraction": 0.1,
                "start_skew_ns": max(starts) - min(starts),
                "overlap_ns": max(0, min(ends) - max(starts)),
                "wave_elapsed_ns": max(ends) - min(starts),
            }
        )
        previous_end = max(ends)
    measured_swap = {
        "start_sample_index": 2,
        "end_sample_index": 47,
        "sample_count": 46,
        "swap_in_delta": 135,
        "swap_out_delta": 0,
        "dispatches": measured_dispatches,
    }
    production_samples.append(
        {
            "monotonic_ns": production_samples[-1]["monotonic_ns"] + 1_000,
            "swap_in_bytes": production_samples[-1]["swap_in_bytes"] + 1,
            "swap_out_bytes": 2_000,
        }
    )
    production_session = {
        "sample_count": len(production_samples),
        "samples": production_samples,
        "swap_in_delta": production_samples[-1]["swap_in_bytes"] - 1_000,
        "swap_out_delta": 0,
    }
    concurrency = {
        "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": "concurrent",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
        "wave_count": analysis.campaign.EXPECTED_WAVES,
        "entries": entries,
        "failure_count": 0,
        "worker_restart_count": 0,
        "recovery_mixing_count": 0,
        "worker_session_swap_telemetry": production_session,
        "measured_phase_swap_window": measured_swap,
        "swap_dispatch_count": 45,
        "swap_in_bytes": production_session["swap_in_delta"],
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }
    operational = {
        "preflight_report_artifact": preflight,
        "concurrency_history_artifact": concurrency,
        "warmup_history_artifact": {
            "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
            "kind": analysis.campaign.CAMPAIGN_KIND + "_warmup_history",
            "execution_mode": "concurrent",
            "worker_ready": ready,
            "worker_warmup": warmup_records,
        },
    }
    manifest = {
        "output_dir": str(root),
        "execution_mode": "concurrent",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
        "sequential_recovery": None,
    }
    swap_audit = analysis.campaign.build_swap_audit(
        preflight, concurrency, execution_mode="concurrent"
    )
    attestation = {
        "execution_mode": "concurrent",
        "result_artifacts": artifacts,
        "raw_result_verification": {
            "authority": "runner",
            "count": 90,
            "method": "sha256_size_and_safe_extraction",
            "analyzer_access": "forbidden",
        },
        "analysis_boundary": analysis.campaign.analysis_boundary(),
        "swap_audit": swap_audit,
    }
    monkeypatch.setattr(
        analysis.campaign.hardened.screen,
        "_validate_followon_warmup_history",
        lambda *args, **kwargs: None,
    )
    module = ModuleType("benchmarks.tabarena_comparator_warmup")
    module.validate_comparator_warmup_history = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return root, operational, manifest, attestation


def test_sequential_recovery_replays_exact_failed_concurrent_source(
    tmp_path, monkeypatch
):
    _, destination, manifest, _ = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)

    analysis._validate_sequential_recovery(
        manifest, campaign_root=destination
    )


def test_sequential_recovery_reads_only_attested_json_and_completion_absence(
    tmp_path, monkeypatch,
):
    source, destination, manifest, _ = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)
    reads: list[Path] = []
    absence_checks: list[Path] = []
    original_read = analysis._read_stable_regular
    original_exists = analysis._exists_including_broken_symlink

    def tracked_read(path, field):
        reads.append(path)
        return original_read(path, field)

    def tracked_exists(path, field):
        absence_checks.append(path)
        return original_exists(path, field)

    monkeypatch.setattr(analysis, "_read_stable_regular", tracked_read)
    monkeypatch.setattr(
        analysis, "_exists_including_broken_symlink", tracked_exists
    )

    analysis._validate_sequential_recovery(
        manifest, campaign_root=destination
    )

    assert reads == [
        source / analysis.campaign.INVALID_ATTEMPT_FILENAME,
        source / analysis.campaign.MANIFEST_FILENAME,
    ]
    assert absence_checks == [
        source / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    ]


def test_recovery_git_ignore_gate_is_case_insensitive(tmp_path, monkeypatch):
    repository = tmp_path / "Repository"
    repository.mkdir()
    monkeypatch.setattr(analysis.campaign, "REPOSITORY_ROOT", repository)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return analysis.subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(analysis.subprocess, "run", fake_run)
    variant = repository.with_name(repository.name.swapcase()) / ".cache" / "run"

    analysis._validate_recovery_namespace_policy(variant)

    assert len(calls) == 1
    assert calls[0][0][-1] == str(Path(".cache/run"))


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
        "unsafe",
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


@pytest.mark.parametrize(
    "raw_suffix",
    [
        ("ExPeRiMeNtS", "hostile", "ReSuLtS.PkL"),
        ("ReSuLtS.PkL", "failed-concurrent"),
    ],
)
def test_sequential_recovery_rejects_case_variant_raw_path_before_access(
    tmp_path, monkeypatch, raw_suffix,
):
    _, destination, manifest, record = _recovery_fixture(tmp_path)
    record["source_output_dir"] = str(destination.joinpath(*raw_suffix))
    touched = _install_raw_path_tripwire(monkeypatch)

    with pytest.raises(RuntimeError, match="unsafe"):
        analysis._validate_sequential_recovery(
            manifest, campaign_root=destination
        )
    assert touched == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda marker: marker.__setitem__("schema_version", 2.0),
        lambda marker: marker.__setitem__("worker_shutdown_confirmed", False),
        lambda marker: marker["failure_swap_telemetry"].__setitem__(
            "teardown_confirmed", False
        ),
        lambda marker: marker["failure_swap_telemetry"].__setitem__(
            "post_teardown_sample_recorded", False
        ),
        lambda marker: marker.__setitem__("error_type", "x" * 257),
        lambda marker: marker.__setitem__("error", "x" * 4_097),
        lambda marker: (
            marker["failure_swap_telemetry"][
                "worker_session_swap_telemetry"
            ]["samples"][0].update(
                {"swap_in_bytes": -1, "swap_out_bytes": -1}
            ),
            marker["failure_swap_telemetry"][
                "worker_session_swap_telemetry"
            ].update({"swap_in_delta": 16, "swap_out_delta": 21}),
            marker["failure_swap_telemetry"].update(
                {"swap_in_bytes": 16, "swap_out_bytes": 21}
            ),
        ),
    ],
)
def test_sequential_recovery_rejects_noncanonical_failure_marker(
    tmp_path, monkeypatch, mutation,
):
    source, destination, manifest, record = _recovery_fixture(tmp_path)
    _stub_runner_recovery_provenance(monkeypatch, manifest)
    marker_path = source / analysis.campaign.INVALID_ATTEMPT_FILENAME
    marker = json.loads(marker_path.read_text())
    mutation(marker)
    _write_json(marker_path, marker)
    _refresh_recovery_record(source, record)

    with pytest.raises(RuntimeError, match="marker|swap telemetry|swap counters"):
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


def test_operational_validator_rejects_boolean_swap_audit_substitution(
    tmp_path, monkeypatch,
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    attestation["swap_audit"]["preflight"][
        "worker_lifecycle_swap_out_bytes"
    ] = False

    with pytest.raises(RuntimeError, match="detaches operational evidence"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


def test_operational_validator_rejects_raw_result_count_coercion(
    tmp_path, monkeypatch,
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    attestation["raw_result_verification"]["count"] = 90.0

    with pytest.raises(RuntimeError, match="raw result count"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["preflight_report_artifact"].__setitem__(
            "failure_count", False
        ),
        lambda value: value["preflight_report_artifact"].__setitem__(
            "ctr23_fit_count", False
        ),
        lambda value: value["preflight_report_artifact"].__setitem__(
            "peak_combined_rss_fraction", "0.1"
        ),
        lambda value: value["concurrency_history_artifact"].__setitem__(
            "worker_restart_count", False
        ),
        lambda value: value["concurrency_history_artifact"].__setitem__(
            "recovery_mixing_count", False
        ),
        lambda value: value["concurrency_history_artifact"].__setitem__(
            "peak_combined_rss_fraction", "0.1"
        ),
    ],
)
def test_operational_validator_rejects_header_type_coercions(
    tmp_path, monkeypatch, mutation,
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    mutation(operational)

    with pytest.raises(RuntimeError, match="operational artifacts"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


def test_operational_validator_never_touches_runner_owned_raw_results(
    tmp_path, monkeypatch
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )

    for method_name in ("resolve", "stat", "lstat", "exists", "is_file", "open", "read_bytes"):
        original = getattr(Path, method_name)

        def guarded(path, *args, _original=original, _name=method_name, **kwargs):
            if path.name == "results.pkl":
                raise AssertionError(f"raw result {_name} is forbidden")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)

    analysis.campaign.validate_operational_artifacts_for_analysis(
        operational,
        manifest=manifest,
        attestation=attestation,
        output_dir=root,
    )


def test_operational_validator_rejects_hostile_scratch_before_raw_access(
    tmp_path, monkeypatch,
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    operational["warmup_history_artifact"]["worker_ready"][0][
        "scratch_root"
    ] = str(root / "ExPeRiMeNtS" / "hostile" / "ReSuLtS.PkL")
    touched = _install_raw_path_tripwire(monkeypatch)

    with pytest.raises(RuntimeError, match="warmup worker readiness"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )
    assert touched == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["preflight_report_artifact"].__setitem__(
            "swap_in_bytes", value["preflight_report_artifact"]["swap_in_bytes"] + 1
        ),
        lambda value: value["concurrency_history_artifact"].__setitem__(
            "swap_in_bytes",
            value["concurrency_history_artifact"]["swap_in_bytes"] + 1,
        ),
        lambda value: value["concurrency_history_artifact"]["entries"][0].__setitem__(
            "swap_in_delta",
            value["concurrency_history_artifact"]["entries"][0]["swap_in_delta"]
            + 1,
        ),
        lambda value: value["concurrency_history_artifact"][
            "worker_session_swap_telemetry"
        ].__setitem__(
            "swap_in_delta",
            value["concurrency_history_artifact"][
                "worker_session_swap_telemetry"
            ]["swap_in_delta"]
            + 1,
        ),
        lambda value: value["concurrency_history_artifact"][
            "measured_phase_swap_window"
        ].__setitem__(
            "swap_in_delta",
            value["concurrency_history_artifact"]["measured_phase_swap_window"][
                "swap_in_delta"
            ]
            + 1,
        ),
    ],
)
def test_operational_validator_rejects_detached_swap_in_evidence(
    tmp_path, monkeypatch, mutation
):
    root, operational, manifest, attestation = _operational_artifacts_fixture(
        tmp_path, monkeypatch
    )
    mutation(operational)

    with pytest.raises(RuntimeError, match="swap"):
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

    with pytest.raises(RuntimeError, match="precedes its release|swap dispatch binding"):
        analysis.campaign.validate_operational_artifacts_for_analysis(
            operational,
            manifest=manifest,
            attestation=attestation,
            output_dir=root,
        )


def _swap_audit_fixture() -> dict:
    waves = [
        {
            "wave_index": index,
            "dispatch_count": 1,
            "swap_in_bytes": 5 if index == 0 else 0,
            "swap_out_bytes": 0,
        }
        for index in range(45)
    ]
    return {
        "policy": "quality_only_swap_in",
        "preflight": {
            "worker_lifecycle_swap_in_bytes": 3,
            "worker_lifecycle_swap_out_bytes": 0,
            "measured_phase_swap_in_bytes": 2,
            "measured_phase_swap_out_bytes": 0,
            "measured_dispatch_count": 1,
        },
        "production": {
            "worker_lifecycle_swap_in_bytes": 7,
            "worker_lifecycle_swap_out_bytes": 0,
            "measured_phase_swap_in_bytes": 5,
            "measured_phase_swap_out_bytes": 0,
            "measured_dispatch_count": 45,
            "wave_count": 45,
            "waves": waves,
        },
    }


def _completion_validation_fixture(swap_audit: dict) -> dict:
    return {
        "result_count": 90,
        "child_fit_count": 720,
        "a10_candidate_fit_count": 648,
        "failure_count": 0,
        "imputation_count": 0,
        "deadline_hit_count": 0,
        "time_callback_hit_count": 0,
        "worker_failure_count": 0,
        "recovery_mixing_count": 0,
        "swap_in_audit_evidence_retained": True,
        "preflight_swap_in_bytes": swap_audit["preflight"][
            "worker_lifecycle_swap_in_bytes"
        ],
        "production_swap_in_bytes": swap_audit["production"][
            "worker_lifecycle_swap_in_bytes"
        ],
        "swap_dispatch_count": 45,
        "swap_wave_count": 45,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
        "unresolved_comparator_stop_count": 288,
        "resource_allocation": {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        },
    }


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("swap_in_audit_evidence_retained", False),
        ("preflight_swap_in_bytes", 4),
        ("production_swap_in_bytes", 8),
        ("swap_dispatch_count", 44),
        ("swap_wave_count", 44),
        ("swap_out_bytes", 1),
        ("swap_out_bytes", False),
        ("failure_count", False),
    ],
)
def test_completion_validation_binds_swap_audit_end_to_end(field, bad_value):
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    swap_audit = _swap_audit_fixture()
    validation = _completion_validation_fixture(swap_audit)
    manifest = {
        "execution_mode": "concurrent",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
    }

    analysis.campaign.validate_completion_for_analysis(
        validation,
        manifest=manifest,
        outer_rows=outer,
        child_rows=children,
        swap_audit=swap_audit,
    )
    validation[field] = bad_value
    with pytest.raises(RuntimeError, match="completion validation"):
        analysis.campaign.validate_completion_for_analysis(
            validation,
            manifest=manifest,
            outer_rows=outer,
            child_rows=children,
            swap_audit=swap_audit,
        )


def test_completion_validation_rejects_boolean_zero_in_audit_and_resources():
    outer = _payload_shape_rows(_outer_rows(), analysis.campaign.OUTER_PAYLOAD_FIELDS)
    children = _payload_shape_rows(
        _child_rows(), analysis.campaign.CHILD_PAYLOAD_FIELDS
    )
    manifest = {
        "execution_mode": "concurrent",
        "swap_policy": analysis.campaign.SWAP_POLICY,
        "timing_admissible": False,
    }
    for mutate in (
        lambda validation, audit: validation.__setitem__(
            "preflight_swap_in_bytes", False
        ),
        lambda validation, audit: audit["preflight"].__setitem__(
            "worker_lifecycle_swap_in_bytes", False
        ),
        lambda validation, audit: validation["resource_allocation"].__setitem__(
            "num_gpus", False
        ),
    ):
        swap_audit = _swap_audit_fixture()
        swap_audit["preflight"]["worker_lifecycle_swap_in_bytes"] = 0
        validation = _completion_validation_fixture(swap_audit)
        mutate(validation, swap_audit)
        with pytest.raises(RuntimeError, match="completion validation"):
            analysis.campaign.validate_completion_for_analysis(
                validation,
                manifest=manifest,
                outer_rows=outer,
                child_rows=children,
                swap_audit=swap_audit,
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
        payload = b"opaque-invalid-pickle-" + relative.encode()
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
    swap_audit = _swap_audit_fixture()
    boundary = analysis.campaign.analysis_boundary()
    payload = {
        "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
        "kind": analysis.campaign.PAYLOAD_KIND,
        "protocol_sha256": analysis.campaign.protocol_sha256(),
        "frozen_protocol_sha256": analysis.campaign.frozen_protocol_sha256(),
        "coordinate_manifest_sha256": analysis.campaign.COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": analysis.campaign.schedule_sha256(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_artifacts_sha256": analysis._sha256(
            analysis._canonical_json(result_artifacts)
        ),
        "analysis_boundary_sha256": analysis._sha256(
            analysis._canonical_json(boundary)
        ),
        "swap_policy": "quality_only_swap_in",
        "timing_admissible": False,
        "swap_audit": swap_audit,
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
        "schema_version": analysis.campaign.HARNESS_SCHEMA_VERSION,
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
        "raw_result_verification": {
            "authority": "runner",
            "count": 90,
            "method": "sha256_size_and_safe_extraction",
            "analyzer_access": "forbidden",
        },
        "analysis_boundary": boundary,
        "swap_audit": swap_audit,
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
    monkeypatch.setattr(
        analysis.campaign,
        "build_swap_audit",
        lambda *args, **kwargs: deepcopy(swap_audit),
        raising=False,
    )
    return root


def test_full_integrity_accepts_runner_attestation_without_raw_result_files(
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
    assert len(protected) == 7


@pytest.mark.parametrize(
    "parts",
    [
        ("ExPeRiMeNtS", "hostile", "ReSuLtS.PkL"),
        ("ReSuLtS.PkL", ".."),
        ("ReSuLtS.PkL", "child"),
    ],
)
def test_full_integrity_rejects_raw_cli_path_before_filesystem_access(
    tmp_path, monkeypatch, parts,
):
    raw_path = tmp_path.joinpath(*parts)
    touched = _install_raw_path_tripwire(monkeypatch)

    with pytest.raises(RuntimeError, match="campaign directory path is unsafe"):
        analysis.verify_campaign_integrity(raw_path)
    assert touched == []


def test_full_integrity_allows_unrelated_experiments_ancestor(
    tmp_path, monkeypatch,
):
    parent = tmp_path / "experiments"
    parent.mkdir()
    root = _campaign_fixture(parent, monkeypatch)

    manifest, *_ = analysis.verify_campaign_integrity(root)

    assert manifest["output_dir"] == str(root.resolve())


def test_full_integrity_reads_exact_json_allowlist_and_never_raw_results(
    tmp_path, monkeypatch
):
    root = _campaign_fixture(tmp_path, monkeypatch).resolve()
    observed = []
    original = analysis._read_stable_regular

    def guarded_read(path, field):
        resolved = path.resolve()
        if root == resolved or root in resolved.parents:
            relative = str(resolved.relative_to(root))
            if relative.endswith("results.pkl"):
                raise AssertionError("analyzer attempted to read a raw result")
            observed.append(relative)
        return original(path, field)

    monkeypatch.setattr(analysis, "_read_stable_regular", guarded_read)

    analysis.verify_campaign_integrity(root)

    assert len(observed) == len(set(observed)) == 7
    assert set(observed) == set(analysis.campaign.ANALYZER_CAMPAIGN_JSON_FILENAMES)


def test_full_integrity_rejects_hostile_manifest_path_before_raw_access(
    tmp_path, monkeypatch,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    manifest_path = root / analysis.campaign.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["output_dir"] = str(
        root / "ExPeRiMeNtS" / "hostile" / "ReSuLtS.PkL"
    )
    _write_json(manifest_path, manifest)
    touched = _install_raw_path_tripwire(monkeypatch)

    with pytest.raises(RuntimeError, match="run manifest"):
        analysis.verify_campaign_integrity(root)
    assert touched == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.__setitem__("schema_version", True),
        lambda manifest: manifest.__setitem__("expected_jobs", 90.0),
        lambda manifest: manifest.__setitem__("time_limit_seconds", 3_600),
        lambda manifest: manifest["schedule"][0]["jobs"][0].__setitem__(
            "worker_slot", False
        ),
    ],
)
def test_full_integrity_rejects_manifest_type_coercions(
    tmp_path, monkeypatch, mutation,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    manifest_path = root / analysis.campaign.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="run manifest"):
        analysis.verify_campaign_integrity(root)


@pytest.mark.parametrize("target", ["payload", "schedule", "manifest"])
def test_full_integrity_rejects_hash_chain_tampering(tmp_path, monkeypatch, target):
    root = _campaign_fixture(tmp_path, monkeypatch)
    if target == "payload":
        path = root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME
    elif target == "schedule":
        path = root / analysis.campaign.SCHEDULE_FILENAME
    else:
        path = root / analysis.campaign.MANIFEST_FILENAME
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="hash|attestation|manifest"):
        analysis.verify_campaign_integrity(root)


def test_full_integrity_rejects_detached_swap_audit(tmp_path, monkeypatch):
    root = _campaign_fixture(tmp_path, monkeypatch)
    payload_path = root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME
    payload = json.loads(payload_path.read_text())
    payload["swap_audit"]["production"]["worker_lifecycle_swap_in_bytes"] += 1
    _write_json(payload_path, payload)
    attestation_path = root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    attestation = json.loads(attestation_path.read_text())
    attestation["analysis_payload_artifact"] = _singleton_metadata(
        payload_path, root
    )
    _write_json(attestation_path, attestation)

    with pytest.raises(RuntimeError, match="swap audit|safe analysis payload"):
        analysis.verify_campaign_integrity(root)


def test_full_integrity_rejects_payload_schema_type_coercion(
    tmp_path, monkeypatch,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    payload_path = root / analysis.campaign.ANALYSIS_PAYLOAD_FILENAME
    payload = json.loads(payload_path.read_text())
    payload["schema_version"] = 2.0
    _write_json(payload_path, payload)
    attestation_path = root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    attestation = json.loads(attestation_path.read_text())
    attestation["analysis_payload_artifact"] = _singleton_metadata(
        payload_path, root
    )
    _write_json(attestation_path, attestation)

    with pytest.raises(RuntimeError, match="safe analysis payload"):
        analysis.verify_campaign_integrity(root)


def test_full_integrity_rejects_raw_result_count_type_coercion(
    tmp_path, monkeypatch,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    attestation_path = root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    attestation = json.loads(attestation_path.read_text())
    attestation["raw_result_verification"]["count"] = 90.0
    _write_json(attestation_path, attestation)

    with pytest.raises(RuntimeError, match="raw result count"):
        analysis.verify_campaign_integrity(root)


def test_full_integrity_rejects_boolean_boundary_substitution(
    tmp_path, monkeypatch,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    attestation_path = root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    attestation = json.loads(attestation_path.read_text())
    attestation["analysis_boundary"]["schema_version"] = True
    _write_json(attestation_path, attestation)

    with pytest.raises(RuntimeError, match="analysis boundary"):
        analysis.verify_campaign_integrity(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2.0),
        ("completed_at_utc", ""),
        ("pid", True),
        ("result_count", 90.0),
        ("expected_result_count", 90.0),
        ("expected_child_fits", 720.0),
    ],
)
def test_full_integrity_rejects_attestation_type_coercions(
    tmp_path, monkeypatch, field, value,
):
    root = _campaign_fixture(tmp_path, monkeypatch)
    attestation_path = root / analysis.campaign.COMPLETION_ATTESTATION_FILENAME
    attestation = json.loads(attestation_path.read_text())
    attestation[field] = value
    _write_json(attestation_path, attestation)

    with pytest.raises(RuntimeError, match="completion attestation"):
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
    assert summary["integrity"]["swap_in_audit_evidence_retained"] is True
    assert summary["integrity"]["preflight_worker_lifecycle_swap_in_bytes"] == 3
    assert summary["integrity"]["production_worker_lifecycle_swap_in_bytes"] == 7
    assert summary["integrity"]["production_dispatches_with_swap_in_telemetry"] == 45
    assert summary["integrity"]["raw_results_read_by_analyzer"] is False
    assert summary["integrity"]["raw_results_deserialized_by_analyzer"] is False
    assert summary["timing_admissible"] is False
