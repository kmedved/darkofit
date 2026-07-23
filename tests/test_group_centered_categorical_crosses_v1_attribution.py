from __future__ import annotations

import copy
import json

import pytest

from benchmarks import run_group_centered_categorical_crosses_v1_attribution as runner


def test_manifest_binds_the_two_spent_m2_datasets_and_three_coordinates():
    manifest = runner.build_manifest()

    assert manifest["status"] == "ready"
    assert manifest["planned_workers"] == 6
    assert manifest["model"]["iterations"] == 1_000
    assert manifest["model"]["thread_count"] == 14
    assert {
        (row["dataset"], row["task_id"])
        for row in manifest["coordinates"]
    } == {
        ("diamonds", 363631),
        ("healthcare_insurance_expenses", 363675),
    }
    assert [
        (row["repeat"], row["fold"])
        for row in manifest["coordinates"][:3]
    ] == [(0, 0), (1, 1), (2, 2)]


def _rows(
    *,
    automatic=(0.99, 1.0, 1.01),
    forced=(0.98, 0.99, 1.0),
    selected=(True, False, True),
    ineligible_datasets=(),
):
    rows = []
    for dataset, task_id in runner.DATASETS:
        for index, (coordinate, repeat, fold) in enumerate(runner.COORDINATES):
            pairs = [[1, 0]]
            eligible = dataset not in ineligible_datasets
            selector = (
                {
                    "eligible": True,
                    "selected": selected[index],
                    "pairs": pairs,
                }
                if eligible
                else {
                    "eligible": False,
                    "selected": False,
                    "pairs": [],
                    "reason": "below_min_samples",
                }
            )
            rows.append(
                {
                    "schema_version": 1,
                    "attribution_id": runner.ATTRIBUTION_ID,
                    "status": "ok",
                    "dataset": dataset,
                    "task_id": task_id,
                    "coordinate": coordinate,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": repeat * 1_000 + fold,
                    "arms": {
                        "constant": {
                            "rmse": 10.0,
                            "fitted_pairs": [],
                            "selector": None,
                        },
                        "automatic": {
                            "rmse": (
                                10.0 * automatic[index] if eligible else 10.0
                            ),
                            "fitted_pairs": (
                                pairs if eligible and selected[index] else []
                            ),
                            "selector": selector,
                        },
                        "forced": {
                            "rmse": 10.0 * forced[index],
                            "fitted_pairs": pairs,
                            "selector": None,
                        },
                    },
                    "forced_pair_source": (
                        "automatic_selector"
                        if eligible
                        else "constant_full_train_importance"
                    ),
                }
            )
    return rows


def test_analyzer_reports_automatic_and_forced_value_and_calibration_findings():
    result = runner.analyze(_rows())

    assert result["integrity"] == {
        "passes": True,
        "workers": 6,
        "arm_rows": 18,
    }
    assert result["gates"]["passes"] is True
    assert result["engagement"]["selected_coordinates"] == 4
    assert len(result["engagement"]["calibration_findings"]) == 2
    assert result["disposition"] == "attribution_supports_opt_in_product_path"


def test_analyzer_rejects_harm_and_incomplete_or_drifted_rows():
    harmed = runner.analyze(
        _rows(automatic=(0.99, 1.0, 1.03), forced=(0.98, 0.99, 1.0))
    )
    assert harmed["gates"]["passes"] is False
    assert harmed["disposition"] == "attribution_does_not_support_automatic_path"

    with pytest.raises(RuntimeError, match="census"):
        runner.analyze(_rows()[:-1])

    drifted = copy.deepcopy(_rows())
    drifted[0]["arms"]["forced"]["fitted_pairs"] = [[2, 0]]
    with pytest.raises(RuntimeError, match="provenance"):
        runner.analyze(drifted)

    wrong_split = copy.deepcopy(_rows())
    wrong_split[0]["repeat"] = 2
    with pytest.raises(RuntimeError, match="worker row"):
        runner.analyze(wrong_split)

    nonfinite = copy.deepcopy(_rows())
    nonfinite[0]["arms"]["automatic"]["rmse"] = float("nan")
    with pytest.raises(RuntimeError, match="provenance"):
        runner.analyze(nonfinite)


def test_analyzer_reports_small_dataset_coverage_gap_without_losing_forced_probe():
    result = runner.analyze(
        _rows(ineligible_datasets={"healthcare_insurance_expenses"})
    )

    assert result["gates"]["automatic_not_worse_each_dataset"] is True
    assert result["gates"]["automatic_worst_coordinate_at_most_1_02"] is True
    assert result["gates"]["automatic_eligible_all_coordinates"] is False
    assert result["gates"]["passes"] is False
    assert result["engagement"]["eligible_coordinates"] == 3
    assert result["engagement"]["ineligible_coordinates"] == 3
    assert {
        finding["reason"]
        for finding in result["engagement"]["calibration_findings"]
        if finding["dataset"] == "healthcare_insurance_expenses"
    } == {"automatic_ineligible_with_forced_value_left"}
    assert result["disposition"] == "attribution_requires_selector_successor"


def test_output_paths_reject_the_source_tree(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        runner.output_paths(runner.ROOT / "benchmarks" / "not-allowed")

    paths = runner.output_paths(tmp_path / "catcross")
    assert set(paths) == {"launch", "raw", "result"}


def test_execute_rejects_any_manifest_drift_before_launch(tmp_path):
    manifest = runner.build_manifest()
    manifest["coordinates"][0]["seed"] += 1
    manifest_path = tmp_path / "drifted-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest is invalid"):
        runner.execute(
            manifest_path=manifest_path,
            source=tmp_path / "unused-source",
            prefix=tmp_path / "unused-output",
        )

    assert not (tmp_path / "unused-output_launch.json").exists()
