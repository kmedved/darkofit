from __future__ import annotations

import json

from benchmarks import run_large_n_engine as experiment


def _result(arm, rows, fit, rmse, rss, fingerprint):
    return {
        "arm": arm,
        "train_rows": rows,
        "fit_seconds": fit,
        "predict_seconds": 0.1,
        "rmse": rmse,
        "peak_rss_bytes": rss,
        "behavior_fingerprint_sha256": fingerprint,
        "metadata": {
            "fused_engagement_count": (
                1800 if arm == experiment.DARKO else 0
            )
        },
    }


def _results(darko_ratio=0.70, rmse_ratio=1.001):
    results = []
    for block in range(3):
        for rows in experiment.TRAIN_ROWS:
            results.extend(
                (
                    _result(
                        experiment.DARKO,
                        rows,
                        darko_ratio,
                        rmse_ratio,
                        90,
                        f"darko-{rows}",
                    ),
                    _result(
                        experiment.CHIMERA,
                        rows,
                        1.0,
                        1.0,
                        100,
                        f"chimera-{rows}",
                    ),
                )
            )
    return results


def test_analysis_certifies_fast_quality_neutral_lane():
    analysis = experiment.analyze(_results())

    assert analysis["passes_all_gates"] is True
    assert analysis["fit_geomean_speedup"] > 1.30
    assert analysis["recommendation"] == (
        "certify_large_n_engine_advantage"
    )


def test_analysis_rejects_speed_or_quality_failure():
    slow = experiment.analyze(_results(darko_ratio=0.80))
    regret = experiment.analyze(_results(rmse_ratio=1.003))

    assert slow["gates"]["fit_geomean_at_most_1_over_1_30"] is False
    assert regret["gates"]["quality_noninferior"] is False
    assert slow["recommendation"] == (
        "do_not_claim_large_n_engine_advantage"
    )
    assert regret["recommendation"] == (
        "do_not_claim_large_n_engine_advantage"
    )


def test_behavior_metadata_excludes_only_observational_timing():
    metadata = {
        "fitted_tree_count": 300,
        "final_fit": {
            "stop_reason": "iteration_limit",
            "phase_seconds": {"tree_build": 1.2},
        },
        "selection_fit": None,
    }

    behavior = experiment._behavior_metadata(metadata)

    assert behavior["final_fit"]["phase_seconds"] is None
    assert behavior["final_fit"]["stop_reason"] == "iteration_limit"
    assert metadata["final_fit"]["phase_seconds"] == {"tree_build": 1.2}
    json.dumps(behavior, allow_nan=False)
