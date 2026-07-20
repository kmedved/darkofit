"""Contract and fail-closed checks for standing M5/M6 infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchmark_adapters import RevisionSpec, standing_slice_specs  # noqa: E402
from bench_compare_revisions import (  # noqa: E402
    _ordered_variants,
    _standing_order_index,
)
from run_standing_evidence import (  # noqa: E402
    summarize_pairs,
    validate_rows,
    validate_source_contract,
)
from standing_evidence import (  # noqa: E402
    M5_SENTINEL_DOMAINS,
    M6_DATASETS,
    M6_SMOKE_DATASETS,
    contract_payload,
    m6_expected_grid,
)


def _row(
    identity,
    *,
    control: Path,
    candidate: Path,
    primary_value: float = 1.0,
):
    variant, dataset, size, seed, weight_mode = identity
    task_by_dataset = {
        "friedman_numeric": "regression",
        "numeric_binary": "binary",
        "categorical_binary": "binary",
    }
    source = control if variant == "control_default" else candidate
    task = task_by_dataset[dataset]
    row = {
        "status": "ok",
        "error": "",
        "variant": variant,
        "revision_path": str(source.resolve()),
        "use_defaults": "True",
        "dataset": dataset,
        "task": task,
        "size": size,
        "seed": str(seed),
        "weight_mode": weight_mode,
        "fit_seconds": "2.0",
        "predict_seconds": "0.2",
        "primary_metric": (
            "weighted_rmse"
            if task == "regression" and weight_mode == "stress"
            else (
                "rmse"
                if task == "regression"
                else (
                    "weighted_log_loss"
                    if weight_mode == "stress"
                    else "log_loss"
                )
            )
        ),
        "primary_value": str(primary_value),
    }
    if task == "regression":
        rmse = primary_value if weight_mode == "none" else 1.0
        weighted_rmse = (
            primary_value if weight_mode == "stress" else rmse
        )
        row.update(
            rmse=str(rmse),
            mae="0.8",
            r2="0.2",
            weighted_rmse=str(weighted_rmse),
            weighted_mae="0.8",
            weighted_r2="0.2",
        )
    else:
        log_loss = primary_value if weight_mode == "none" else 0.5
        weighted_log_loss = (
            primary_value if weight_mode == "stress" else log_loss
        )
        row.update(
            accuracy="0.7",
            f1_macro="0.6",
            log_loss=str(log_loss),
            brier="0.3",
            weighted_accuracy="0.7",
            weighted_f1_macro="0.6",
            weighted_log_loss=str(weighted_log_loss),
            weighted_brier="0.3",
        )
    return row


def _smoke_rows(control: Path, candidate: Path):
    return [
        _row(identity, control=control, candidate=candidate)
        for identity in m6_expected_grid(smoke=True)
    ]


def test_standing_slice_specs_are_named_public_default_arms():
    specs = standing_slice_specs("/control", "/candidate")

    assert [spec.label for spec in specs] == [
        "control_default",
        "candidate_default",
    ]
    assert [spec.path for spec in specs] == ["/control", "/candidate"]
    assert all(spec.use_defaults for spec in specs)
    assert all(spec.tree_mode is None for spec in specs)


def test_standing_order_rotates_only_for_m6_cells():
    variants = [RevisionSpec("control", "/c"), RevisionSpec("candidate", "/n")]

    assert [
        variant.label
        for variant in _ordered_variants(
            variants, policy_suite="standing-slice", cell_index=0
        )
    ] == ["control", "candidate"]
    assert [
        variant.label
        for variant in _ordered_variants(
            variants, policy_suite="standing-slice", cell_index=1
        )
    ] == ["candidate", "control"]
    assert [
        variant.label
        for variant in _ordered_variants(
            variants, policy_suite="revision", cell_index=1
        )
    ] == ["control", "candidate"]


def test_standing_order_is_balanced_within_each_weight_stratum():
    variants = [RevisionSpec("control", "/c"), RevisionSpec("candidate", "/n")]

    orders = {
        weight_index: [
            [
                variant.label
                for variant in _ordered_variants(
                    variants,
                    policy_suite="standing-slice",
                    cell_index=_standing_order_index(
                        block_index,
                        weight_index,
                    ),
                )
            ]
            for block_index in range(4)
        ]
        for weight_index in range(2)
    }

    assert orders[0] == [
        ["control", "candidate"],
        ["candidate", "control"],
        ["control", "candidate"],
        ["candidate", "control"],
    ]
    assert orders[1] == [
        ["candidate", "control"],
        ["control", "candidate"],
        ["candidate", "control"],
        ["control", "candidate"],
    ]


def test_standing_contract_covers_classification_and_weighted_domains():
    domain_ids = [domain.id for domain in M5_SENTINEL_DOMAINS]
    tasks = {domain.task for domain in M5_SENTINEL_DOMAINS}

    assert len(domain_ids) == len(set(domain_ids))
    assert {"binary", "multiclass"} <= tasks
    weighted_tasks = {
        domain.task for domain in M5_SENTINEL_DOMAINS if domain.weighted
    }
    assert {"regression", "binary"} <= weighted_tasks
    assert set(M6_SMOKE_DATASETS) < set(M6_DATASETS)
    payload = contract_payload()
    assert payload["contract_version"].endswith("-v1")
    assert payload["m6"]["contract_frozen"] is False
    assert payload["m6"]["backtest_complete"] is False


def test_m6_grid_is_complete_and_unique():
    full = m6_expected_grid()
    smoke = m6_expected_grid(smoke=True)

    assert len(full) == 120
    assert len(set(full)) == 120
    assert len(smoke) == 12
    assert len(set(smoke)) == 12


def test_source_contract_requires_a_same_clean_checkout_for_null_smoke():
    harness = {
        "path": "/harness",
        "head": "h",
        "tree": "tree-h",
        "branch": "main",
        "clean": True,
        "status": [],
    }
    source = {
        "path": "/repo",
        "head": "a",
        "tree": "b",
        "branch": "main",
        "clean": True,
        "status": [],
    }

    validate_source_contract(harness, source, dict(source), smoke=True)

    with pytest.raises(RuntimeError, match="same checkout"):
        validate_source_contract(
            harness,
            source,
            {**source, "path": "/other"},
            smoke=True,
        )
    with pytest.raises(RuntimeError, match="clean committed"):
        validate_source_contract(
            harness,
            {**source, "clean": False, "status": ["M file"]},
            {**source, "clean": False, "status": ["M file"]},
            smoke=True,
        )
    with pytest.raises(RuntimeError, match="harness checkout"):
        validate_source_contract(
            {**harness, "clean": False, "status": ["M runner"]},
            source,
            dict(source),
            smoke=True,
        )


def test_source_contract_requires_clean_distinct_trees_for_full_m6():
    harness = {
        "path": "/harness",
        "head": "h",
        "tree": "tree-h",
        "branch": "main",
        "clean": True,
        "status": [],
    }
    control = {
        "path": "/control",
        "head": "a",
        "tree": "tree-a",
        "branch": "main",
        "clean": True,
        "status": [],
    }
    candidate = {
        **control,
        "path": "/candidate",
        "head": "b",
        "tree": "tree-b",
    }

    validate_source_contract(harness, control, candidate, smoke=False)

    with pytest.raises(RuntimeError, match="clean committed"):
        validate_source_contract(
            harness,
            control,
            {**candidate, "clean": False, "status": ["M file"]},
            smoke=False,
        )
    with pytest.raises(RuntimeError, match="distinct"):
        validate_source_contract(
            harness,
            control,
            {**candidate, "tree": control["tree"]},
            smoke=False,
        )


def test_validate_rows_accepts_complete_smoke_grid(tmp_path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    rows = _smoke_rows(control, candidate)

    result = validate_rows(
        rows,
        smoke=True,
        control=control,
        candidate=candidate,
    )

    assert result == {
        "expected_rows": 12,
        "actual_rows": 12,
        "all_rows_ok": True,
        "grid_complete": True,
    }


@pytest.mark.parametrize("failure", ["missing", "duplicate", "error"])
def test_validate_rows_rejects_broken_smoke_grid(tmp_path, failure):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    rows = _smoke_rows(control, candidate)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[0]["status"] = "error"
        rows[0]["error"] = "boom"

    with pytest.raises(RuntimeError):
        validate_rows(
            rows,
            smoke=True,
            control=control,
            candidate=candidate,
        )


def test_validate_rows_rejects_nonfinite_secondary_metric(tmp_path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    rows = _smoke_rows(control, candidate)
    rows[0]["r2"] = "nan"

    with pytest.raises(RuntimeError, match="non-finite metric 'r2'"):
        validate_rows(
            rows,
            smoke=True,
            control=control,
            candidate=candidate,
        )


def test_validate_rows_rejects_primary_metric_value_disagreement(tmp_path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    rows = _smoke_rows(control, candidate)
    rows[0]["rmse"] = "1.1"

    with pytest.raises(RuntimeError, match="primary value disagrees"):
        validate_rows(
            rows,
            smoke=True,
            control=control,
            candidate=candidate,
        )


def test_pair_summary_uses_matched_candidate_over_control_ratios(tmp_path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    rows = _smoke_rows(control, candidate)
    for row in rows:
        if row["variant"] == "candidate_default":
            row["primary_value"] = "0.9"
            row["fit_seconds"] = "1.5"
            row["predict_seconds"] = "0.1"

    summary = summarize_pairs(rows)

    assert summary["paired_cells"] == 6
    assert summary["candidate_loss_wins"] == 6
    assert summary["candidate_loss_ties"] == 0
    assert summary["candidate_loss_losses"] == 0
    assert summary["candidate_over_control_primary_loss_ratio"]["median"] == 0.9
    assert summary["candidate_over_control_fit_seconds_ratio"]["median"] == 0.75
    assert (
        summary["candidate_over_control_predict_seconds_ratio"]["median"]
        == 0.5
    )
