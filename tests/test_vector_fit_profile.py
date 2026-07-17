import hashlib
import json
from pathlib import Path

from benchmarks import run_vector_fit_profile as experiment


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "vector_fit_profile.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "e1ae6facac2a9c465fe0bbde6b99c7c649b3652e9600777935238ca92f8b876f"
)


def _row(case, fit, tree, grad, prediction="same"):
    phases = {phase: 0.0 for phase in experiment.PHASES}
    phases["tree_build"] = tree
    phases["grad_hess"] = grad
    return {
        "case": case,
        "fit_seconds": fit,
        "seconds_per_round": fit / experiment.ITERATIONS,
        "phase_seconds": phases,
        "prediction_sha256": prediction,
        "fitted_round_count": experiment.ITERATIONS,
        "fitted_component_tree_count": experiment.ITERATIONS,
        "selected_tree_mode": "catboost",
        "loss": "test",
        "peak_rss_bytes": 100,
    }


def test_analysis_selects_largest_noncontrol_phase():
    results = []
    for case in experiment.CASES:
        for _ in range(experiment.REPETITIONS):
            if case == "student_t_lightgbm":
                results.append(_row(case, 10.0, 1.0, 9.0))
            else:
                results.append(_row(case, 10.0, 8.0, 2.0))

    analysis = experiment.analyze(results)

    assert analysis["summaries"]["scalar_rmse_catboost"]["largest_phase"] == (
        "tree_build"
    )
    assert analysis["selected_opportunity"] == {
        "case": "student_t_lightgbm",
        "phase": "grad_hess",
        "share_of_attributed": 0.9,
    }
    assert analysis["recommendation"] == (
        "profile_grad_hess_inside_student_t_lightgbm_before_e1"
    )


def test_analysis_rejects_changed_predictions_or_missing_workers():
    results = []
    for case in experiment.CASES:
        for repeat in range(experiment.REPETITIONS):
            results.append(
                _row(
                    case,
                    10.0,
                    8.0,
                    2.0,
                    prediction=(
                        f"changed-{repeat}"
                        if case == "binary_catboost"
                        else "same"
                    ),
                )
            )

    try:
        experiment.analyze(results)
    except RuntimeError as exc:
        assert "predictions changed" in str(exc)
    else:
        raise AssertionError("changed profile predictions were accepted")

    complete = []
    for case in experiment.CASES:
        for _ in range(experiment.REPETITIONS):
            complete.append(_row(case, 10.0, 8.0, 2.0))
    removed = False
    results = []
    for row in complete:
        if row["case"] == "gaussian_lightgbm" and not removed:
            removed = True
            continue
        results.append(row)
    try:
        experiment.analyze(results)
    except RuntimeError as exc:
        assert "does not have" in str(exc)
    else:
        raise AssertionError("incomplete profile was accepted")


def test_recorded_profile_is_clean_bound_and_tree_build_limited():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["source"]["clean"] is True
    assert artifact["source"]["head"] == (
        "447190e044269d3c3f5e3f34c3d5c00cedd90d3c"
    )
    assert artifact["protocol"]["sha256"] == hashlib.sha256(
        experiment.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["runner_sha256"] == hashlib.sha256(
        Path(experiment.__file__).read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["lockbox_data_used"] is False
    assert artifact["analysis"]["selected_opportunity"]["phase"] == (
        "tree_build"
    )
    assert artifact["analysis"]["selected_opportunity"]["case"] == (
        "gaussian_lightgbm"
    )
    for summary in artifact["analysis"]["summaries"].values():
        assert summary["largest_phase"] == "tree_build"
        assert summary["fit_iqr_over_median"] < 0.05
