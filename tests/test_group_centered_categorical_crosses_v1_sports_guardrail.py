from __future__ import annotations

import copy
import hashlib

import pytest

from benchmarks import (
    run_group_centered_categorical_crosses_v1_sports_guardrail as runner,
)


def _manifest():
    coordinates = [
        {
            "coordinate": f"fold_{fold}",
            "kind": "fold",
            "fold": fold,
        }
        for fold in range(10)
    ]
    coordinates.append(
        {"coordinate": "held_team", "kind": "held_team", "fold": None}
    )
    return {
        "coordinates": coordinates,
        "checks": {
            "fold_geomean_at_most": 1.0,
            "worst_fold_at_most": 1.02,
            "held_team_at_most": 1.0,
            "cold_player_at_most": 1.0,
            "eligible_all_coordinates": True,
        },
    }


def _metric(rmse, *, rows=100, prediction="control"):
    return {
        "rows": rows,
        "rmse": rmse,
        "r2": 0.5 - (rmse - 10.0) / 10.0,
        "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
    }


def _rows(*, fold_ratio=0.99, held_ratio=0.99, cold_ratio=0.98):
    rows = []
    for spec in _manifest()["coordinates"]:
        for arm in runner.ARMS:
            selected = arm == "automatic"
            ratio = 1.0
            if selected:
                ratio = (
                    fold_ratio if spec["kind"] == "fold" else held_ratio
                )
            metrics = {
                "all": _metric(
                    10.0 * ratio,
                    prediction=f"{spec['coordinate']}-{arm}",
                )
            }
            if spec["kind"] == "held_team":
                metrics.update(
                    {
                        "seen_player": _metric(
                            10.0 * ratio,
                            rows=80,
                            prediction=f"seen-{arm}",
                        ),
                        "cold_player": _metric(
                            10.0 * (cold_ratio if selected else 1.0),
                            rows=20,
                            prediction=f"cold-{arm}",
                        ),
                    }
                )
            rows.append(
                {
                    "schema_version": 1,
                    "guardrail_id": runner.GUARDRAIL_ID,
                    "status": "ok",
                    "coordinate": spec["coordinate"],
                    "kind": spec["kind"],
                    "fold": spec["fold"],
                    "arm": arm,
                    "fit_seconds": 2.0 if selected else 1.0,
                    "predict_seconds": 0.2 if selected else 0.1,
                    "prediction_sha256": hashlib.sha256(
                        f"{spec['coordinate']}-{arm}".encode()
                    ).hexdigest(),
                    "source": {
                        "head": runner.CANDIDATE_HEAD,
                        "clean": True,
                    },
                    "feature_count": 19,
                    "cat_features": list(runner.CAT_COLUMNS),
                    "metrics": metrics,
                    "selector": (
                        {
                            "eligible": True,
                            "selected": True,
                            "pairs": [[0, 15]],
                        }
                        if selected
                        else None
                    ),
                    "fitted_pairs": [[0, 15]] if selected else [],
                    "resolved_threads": runner.THREADS,
                    "ambient_thread_restored": True,
                }
            )
    return rows


def test_analyzer_accepts_positive_cold_player_guardrail():
    result = runner.analyze(_rows(), _manifest())

    assert result["integrity"]["workers"] == 22
    assert result["quality"]["fold_equal_geomean_rmse_ratio"] == pytest.approx(
        0.99
    )
    assert result["quality"]["held_team_views"]["cold_player"][
        "rmse_ratio"
    ] == pytest.approx(0.98)
    assert result["engagement"] == {
        "eligible_coordinates": 11,
        "selected_coordinates": 11,
        "total_coordinates": 11,
        "held_team_selected": True,
    }
    assert result["gates"]["passes"] is True
    assert result["disposition"] == "sports_guardrail_supports_scoped_opt_in"


def test_analyzer_rejects_cold_player_harm_and_integrity_drift():
    harmed = runner.analyze(_rows(cold_ratio=1.01), _manifest())
    assert harmed["gates"]["cold_player_at_most_1_0"] is False
    assert harmed["gates"]["passes"] is False
    assert harmed["disposition"] == "sports_guardrail_does_not_support_opt_in"

    missing = _rows()[:-1]
    with pytest.raises(RuntimeError, match="census"):
        runner.analyze(missing, _manifest())

    drifted = copy.deepcopy(_rows())
    drifted[0]["resolved_threads"] = 2
    with pytest.raises(RuntimeError, match="worker row"):
        runner.analyze(drifted, _manifest())


def test_analyzer_requires_exact_fallback_when_selector_declines():
    rows = _rows()
    automatic = next(
        row
        for row in rows
        if row["coordinate"] == "fold_0" and row["arm"] == "automatic"
    )
    control = next(
        row
        for row in rows
        if row["coordinate"] == "fold_0" and row["arm"] == "control"
    )
    automatic["selector"]["selected"] = False
    automatic["fitted_pairs"] = []
    automatic["metrics"] = copy.deepcopy(control["metrics"])
    automatic["prediction_sha256"] = control["prediction_sha256"]

    result = runner.analyze(rows, _manifest())
    assert result["engagement"]["selected_coordinates"] == 10

    automatic["prediction_sha256"] = hashlib.sha256(b"drifted").hexdigest()
    with pytest.raises(RuntimeError, match="fall back exactly"):
        runner.analyze(rows, _manifest())


def test_output_paths_reject_source_tree(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        runner.output_paths(runner.ROOT / "benchmarks" / "not-allowed")
    assert set(runner.output_paths(tmp_path / "sports")) == {
        "launch",
        "raw",
        "result",
    }
