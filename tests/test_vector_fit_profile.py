from benchmarks import run_vector_fit_profile as experiment


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
