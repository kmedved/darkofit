from benchmarks import run_automatic_linear_selector_v3_protein as runner


def _rows(*, automatic_ratios=(0.95, 0.98, 1.0), selected=(True, True, False)):
    rows = []
    for cell in runner.base.expected_ordered_grid():
        coordinate = int(cell["coordinate"])
        arm = cell["arm"]
        constant_rmse = 2.0 + coordinate
        if arm == "constant":
            rmse = constant_rmse
            digest = f"constant-{coordinate}"
            selector = None
        elif arm == "explicit_linear":
            rmse = constant_rmse * (0.94 + 0.01 * coordinate)
            digest = f"linear-{coordinate}"
            selector = None
        else:
            rmse = constant_rmse * automatic_ratios[coordinate]
            chosen = selected[coordinate]
            digest = (
                f"linear-{coordinate}" if chosen else f"constant-{coordinate}"
            )
            selector = {
                "version": 2,
                "eligible": True,
                "minimum_gain_z": 2.0,
                "resolved_linear_leaves": chosen,
                "reason": (
                    "selected_linear" if chosen else "gain_not_above_noise"
                ),
                "relative_validation_improvement": 0.02,
                "paired_mse_gain_z": 2.5 if chosen else 1.5,
            }
        rows.append({
            **cell,
            "fingerprints": {"split": coordinate},
            "environment": runner.base.WORKER_ENVIRONMENT,
            "warnings": [],
            "fit_rss": {"errors": []},
            "numba_threads_before_fit": runner.base.THREADS,
            "numba_threads_after_fit": runner.base.THREADS,
            "numba_threads_after_predict": runner.base.THREADS,
            "numba_threads_after_timing": runner.base.THREADS,
            "test_rmse": rmse,
            "prediction_sha256": digest,
            "core_booster_state_sha256": digest,
            "selector": selector,
        })
    return rows


def test_analyzer_advances_clear_exact_automatic_gain():
    result = runner.analyze_rows(_rows())
    assert result["disposition"] == "ready_for_holdout_ship_check"
    assert result["selected_coordinate_count"] == 2
    assert result["all_resolved_arms_exact"] is True
    assert result["worst_coordinate_ratio"] == 1.0


def test_analyzer_keeps_opt_in_on_coordinate_harm_or_state_mismatch():
    harmed = runner.analyze_rows(
        _rows(automatic_ratios=(0.95, 0.98, 1.001))
    )
    assert harmed["disposition"] == "keep_opt_in"

    rows = _rows()
    automatic = next(
        row for row in rows
        if row["coordinate"] == 0 and row["arm"] == "automatic"
    )
    automatic["core_booster_state_sha256"] = "wrong"
    mismatched = runner.analyze_rows(rows)
    assert mismatched["disposition"] == "keep_opt_in"
    assert mismatched["all_resolved_arms_exact"] is False
