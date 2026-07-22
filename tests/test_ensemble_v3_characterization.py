from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import analyze_ensemble_v3_characterization as analysis  # noqa: E402
import run_ensemble_v3_characterization as campaign  # noqa: E402
from darkofit import DarkoRegressor  # noqa: E402
from darkofit.sklearn_api import (  # noqa: E402
    _fit_ensemble_v3_release_candidate,
    _fit_private_ensemble_v3,
)


READOUT = BENCH / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"


def test_frozen_grid_is_balanced_and_exact():
    assert campaign.execution_spec()["threads"] == 14
    assert campaign.BATCH_SIZES == (8192, 65536, 524288, 2000000)
    assert len(campaign.CASES) == 4
    for case_id in campaign.CASES:
        positions = {arm: [] for arm in campaign.ARMS}
        for block in range(campaign.BLOCKS):
            order = campaign.order_for(case_id, block)
            assert set(order) == set(campaign.ARMS)
            for position, arm in enumerate(order):
                positions[arm].append(position)
        assert all(sorted(value) == [0, 1, 2] for value in positions.values())


def test_quality_analysis_reproduces_readout_and_declared_uncertainty():
    result = analysis.analyze_quality(json.loads(READOUT.read_text()))
    assert result["wins_vs_single"] == result["case_count"] == 13
    assert result["all_case_geometric_mean"] == pytest.approx(0.9655130355873516)
    assert result["sports_geometric_mean"] == pytest.approx(0.9610769432150231)
    assert result["general_geometric_mean"] == pytest.approx(0.9755692524591789)
    assert set(result["sports"]["leave_one_season_out"]) == {"2014", "2015", "2016"}
    assert len(result["general"]["leave_one_case_out"]) == 4
    assert result["sports"]["cluster_bootstrap"]["draws"] == 100_000


def test_release_candidate_matches_historical_combined_mechanics():
    rng = np.random.default_rng(20260721)
    X = rng.normal(size=(180, 6))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(0, 0.1, len(X))
    params = {
        "iterations": 5,
        "early_stopping_rounds": 2,
        "random_state": 4,
        "thread_count": 2,
        "n_ensembles": 8,
        "diagnostic_warnings": "never",
    }
    historical = _fit_private_ensemble_v3(
        DarkoRegressor(**params),
        X,
        y,
        sampling="without_replacement",
        sampling_unit="rows",
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
    )
    candidate = _fit_ensemble_v3_release_candidate(
        DarkoRegressor(**params), X, y
    )
    np.testing.assert_array_equal(historical.predict(X), candidate.predict(X))
    assert historical.ensemble_metadata_["sampling"] == candidate.ensemble_metadata_["sampling"]
    assert historical.ensemble_metadata_["member_policy"] == candidate.ensemble_metadata_["member_policy"]
    assert historical.ensemble_metadata_["sample_fraction"] == candidate.ensemble_metadata_["sample_fraction"] == 0.8
    assert candidate.ensemble_metadata_["recipe_contract"] == "ensemble-v3-public-contract-v1"


def _raw_fixture():
    rows = []
    for block in range(campaign.BLOCKS):
        for case_id in campaign.CASES:
            for position, arm in enumerate(campaign.order_for(case_id, block)):
                scale = {
                    campaign.DARKO_SINGLE: 1.0,
                    campaign.DARKO_V3: 4.0,
                    campaign.CHIMERA_SINGLE: 0.8,
                }[arm]
                rows.append(
                    {
                        "block": block,
                        "position": position,
                        "case_id": case_id,
                        "arm": arm,
                        "fit_seconds": scale,
                        "fit_rss": {
                            "scope": "worker_plus_recursive_children",
                            "start_bytes": 100,
                            "peak_bytes": int(1000 * scale),
                            "peak_delta_bytes": int(100 * scale),
                            "errors": [],
                        },
                        "archive": {
                            "format": "darkofit_safe_npz" if arm != campaign.CHIMERA_SINGLE else "python_pickle_telemetry",
                            "bytes": int(1000 * scale),
                            "roundtrip_exact": True if arm != campaign.CHIMERA_SINGLE else None,
                        },
                        "predictions": {
                            str(size): {
                                "seconds_per_call": 0.001 * scale,
                                "rows_per_second": size / (0.001 * scale),
                                "interval_seconds": 1.0,
                                "minimum_interval_met": True,
                                "calls": 2,
                            }
                            for size in campaign.BATCH_SIZES
                        },
                    }
                )
    return rows


def test_resource_and_prediction_analysis_use_paired_complete_grid():
    rows = _raw_fixture()
    resources = analysis.analyze_resources(rows)
    assert resources["fit_seconds"]["equal_case_geometric_mean_ratio"] == pytest.approx(4.0)
    assert resources["safe_npz_archive_bytes"]["equal_case_geometric_mean_ratio"] == pytest.approx(4.0)
    prediction = analysis.analyze_prediction(rows)
    assert prediction["all_intervals_meet_minimum"] is True
    assert prediction["aggregate"]["darkofit_single_over_chimeraboost_0_18_single"]["equal_coordinate_geometric_mean_ratio"] == pytest.approx(1.25)
    assert prediction["aggregate"]["darkofit_ensemble_v3_over_darkofit_single"]["equal_coordinate_geometric_mean_ratio"] == pytest.approx(4.0)


def test_analysis_rejects_duplicate_or_incomplete_grid():
    rows = _raw_fixture()
    duplicate = [*rows, copy.deepcopy(rows[0])]
    with pytest.raises(RuntimeError, match="duplicate"):
        analysis._row_index(duplicate)
    with pytest.raises(RuntimeError, match="exact grid"):
        analysis.analyze_prediction(rows[:-1])


def test_protocol_preserves_scope_and_claim_boundaries():
    text = campaign.PROTOCOL_PATH.read_text()
    for token in (
        campaign.CONTRACT_ID,
        campaign.DARKOFIT_HEAD,
        campaign.CHIMERABOOST_HEAD,
        "100,000",
        "leave-one-case-out",
        "process-tree",
        "2,000,000",
        "not M2, M4",
        "not an optimized default",
    ):
        assert token in text
