from __future__ import annotations

import random

import pytest

from benchmarks import run_m1_q0_wave1 as experiment


def _m1_row(arm, rows, block, fit, rmse=1.0):
    return {
        "arm": arm,
        "train_rows": rows,
        "block": block,
        "position": experiment.M1_BLOCK_ORDERS[block].index(arm),
        "fit_seconds": fit,
        "predict_seconds": 0.1,
        "rmse": rmse,
        "peak_rss_bytes": 100,
        "data_probe_sha256": f"data-{rows}",
        "worker_stderr": None,
        "behavior_fingerprint_sha256": f"{arm}-{rows}",
        "serialization": {"common_pickle_bytes": 1_000},
        "metadata": {
            "fitted_tree_count": experiment.M1_ITERATIONS,
            "resolved_thread_count": experiment.THREADS,
            "resolved_depth": 6,
            "resolved_learning_rate": 0.1,
            "fused_engagement_count": 1 if arm == experiment.DARKO else 0,
            "tree_mode": "catboost",
            "histogram_dtype": "float64",
            "linear_leaves_active": False,
            "bin_sample_count": 200_000,
            "quantize_gradients": arm == experiment.CHIMERA_QUANTIZED,
            "linear_leaves_selected": False,
            "cross_features_selected": False,
        },
    }


def _m1_results(
    *,
    darko_fit=0.60,
    quantized_fit=0.70,
    float_fit=1.0,
    quantized_rmse=1.0,
    float_rmse=1.0,
):
    fits = {
        experiment.DARKO: darko_fit,
        experiment.CHIMERA_QUANTIZED: quantized_fit,
        experiment.CHIMERA_FLOAT: float_fit,
    }
    rmses = {
        experiment.DARKO: 1.0,
        experiment.CHIMERA_QUANTIZED: quantized_rmse,
        experiment.CHIMERA_FLOAT: float_rmse,
    }
    results = []
    for block, order in enumerate(experiment.M1_BLOCK_ORDERS):
        for rows in experiment.TRAIN_ROWS:
            for arm in order:
                results.append(
                    _m1_row(
                        arm,
                        rows,
                        block,
                        fits[arm],
                        rmse=rmses[arm],
                    )
                )
    return results


def _components(mode, fused_share):
    production = mode == experiment.Q0_PRODUCTION
    return {
        "fused_histogram_split": {
            "calls": 240 if production else 0,
            "seconds": fused_share if production else 0.0,
        },
        "histogram_construction": {
            "calls": 0 if production else 240,
            "seconds": 0.0 if production else 0.55,
        },
        "split_search": {
            "calls": 0 if production else 240,
            "seconds": 0.0 if production else 0.20,
        },
        "sibling_subtraction": {"calls": 0, "seconds": 0.0},
        "leaf_values": {"calls": 40, "seconds": 0.02},
        "leaf_routing": {"calls": 240, "seconds": 0.03},
    }


def _q0_row(mode, rows, block, fused_share):
    production = mode == experiment.Q0_PRODUCTION
    return {
        "mode": mode,
        "train_rows": rows,
        "block": block,
        "position": experiment.Q0_BLOCK_ORDERS[block].index(mode),
        "fit_seconds": 1.0 if production else 1.2,
        "predict_seconds": 0.1,
        "peak_rss_bytes": 100,
        "data_probe_sha256": f"data-{rows}",
        "worker_stderr": None,
        "behavior_fingerprint_sha256": f"behavior-{rows}",
        "fused_engagement_count": 240 if production else 0,
        "components": _components(mode, fused_share),
        "phase_seconds": {
            "preprocess": 0.1,
            "grad_hess": 0.05,
            "tree_build": 0.8,
            "train_update": 0.05,
            "validation_predict": 0.0,
            "loss_eval": 0.0,
        },
        "metadata": {
            "fitted_tree_count": experiment.Q0_ITERATIONS,
            "resolved_thread_count": experiment.THREADS,
            "resolved_depth": 6,
            "resolved_learning_rate": 0.1,
            "tree_mode": "catboost",
            "histogram_dtype": "float64",
            "linear_leaves_active": False,
            "bin_sample_count": 200_000,
        },
        "profile_accounting": {
            "timed_tree_components_seconds": (
                fused_share + 0.05 if production else 0.85
            ),
            "tree_build_seconds": 0.8 if production else 0.9,
        },
    }


def _q0_results(fused_share=0.50):
    results = []
    for block, order in enumerate(experiment.Q0_BLOCK_ORDERS):
        for rows in experiment.TRAIN_ROWS:
            for mode in order:
                results.append(
                    _q0_row(mode, rows, block, fused_share)
                )
    return results


def test_m1_uses_all_six_primary_arm_permutations():
    assert len(experiment.M1_BLOCK_ORDERS) == 6
    assert set(experiment.M1_BLOCK_ORDERS) == set(
        __import__("itertools").permutations(experiment.M1_ARMS)
    )


def test_m1_analysis_reports_position_and_material_donor_signal():
    results = _m1_results()
    random.Random(4).shuffle(results)

    analysis = experiment.analyze_m1(results)

    assert analysis["integrity"]["all_behavior_stable"] is True
    assert analysis["descriptive_verdicts"][
        "darkofit_faster_than_current_quantized_chimera"
    ] is True
    assert analysis["descriptive_verdicts"][
        "material_quantization_donor_signal"
    ] is True
    assert analysis["g_m_input"] == "material_quantization_donor_signal"
    assert analysis["certification_or_default_change_authorized"] is False


def test_m1_material_donor_signal_fails_speed_or_quality_guard():
    slow = experiment.analyze_m1(_m1_results(quantized_fit=0.95))
    regret = experiment.analyze_m1(
        _m1_results(quantized_rmse=1.003)
    )

    assert slow["descriptive_verdicts"][
        "material_quantization_donor_signal"
    ] is False
    assert regret["descriptive_verdicts"][
        "quantized_float_quality_neutral"
    ] is False
    assert regret["g_m_input"] == "no_material_quantization_donor_signal"


def test_m1_rejects_duplicate_coordinate_and_missing_stderr():
    duplicate = _m1_results()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(RuntimeError, match="duplicate coordinate"):
        experiment.analyze_m1(duplicate)

    missing_stderr = _m1_results()
    del missing_stderr[-1]["worker_stderr"]
    with pytest.raises(RuntimeError, match="stderr record is missing"):
        experiment.analyze_m1(missing_stderr)


def test_q0_analysis_uses_only_production_share_for_funding_projection():
    analysis = experiment.analyze_q0(_q0_results(fused_share=0.50))

    assert analysis["integrity"]["passed"] is True
    assert analysis["speed_budget"]["equal_share_required_at_prior"] == (
        pytest.approx(0.43333333333333335)
    )
    assert analysis["projection"][
        "conservative_projection_clears_budget"
    ] is True
    assert analysis["profile_supports_q_funding"] is True
    assert analysis["disposition"] == (
        "eligible_for_g_m_quantization_funding_decision"
    )
    assert analysis["prototype_or_public_change_authorized"] is False


def test_q0_closes_before_prototype_when_projection_misses_budget():
    analysis = experiment.analyze_q0(_q0_results(fused_share=0.30))

    assert analysis["integrity"]["passed"] is True
    assert analysis["projection"][
        "conservative_projection_clears_budget"
    ] is False
    assert analysis["profile_supports_q_funding"] is False
    assert analysis["disposition"] == "close_quantization_before_prototype"


def test_q0_integrity_failure_is_inconclusive_not_a_speed_failure():
    results = _q0_results()
    results[-1]["worker_stderr"] = "unexpected warning"

    analysis = experiment.analyze_q0(results)

    assert analysis["integrity"]["passed"] is False
    assert analysis["profile_supports_q_funding"] is False
    assert analysis["disposition"] == (
        "inconclusive_profile_integrity_failure"
    )


def test_wave1_data_generator_is_deterministic_and_disjoint():
    first = experiment._data(100, 20)
    second = experiment._data(100, 20)

    assert first[-1] == second[-1]
    assert first[0].shape == (100, experiment.FEATURES)
    assert first[2].shape == (20, experiment.FEATURES)
    assert first[0][-1] is not first[2][0]
    assert (first[0] == second[0]).all()
    assert (first[3] == second[3]).all()


def test_wave1_source_pins_and_speed_budget_are_explicit():
    protocol = experiment.PROTOCOL.read_text()

    assert experiment.DARKO_SOURCE_HEAD in protocol
    assert experiment.CHIMERA_SOURCE_HEAD in protocol
    assert "10% lower" in protocol
    assert "0.433333" in protocol
    assert "unfused reference" in protocol
    assert experiment.Q_REQUIRED_EQUAL_SHARE == pytest.approx(
        0.43333333333333335
    )
