import copy

import pytest

from benchmarks import run_basketball_warmup as experiment


def _summary(median, iqr_fraction=0.05):
    return {
        "values_seconds": [median] * experiment.TIMING_BLOCKS,
        "minimum_seconds": median,
        "median_seconds": median,
        "maximum_seconds": median,
        "iqr_seconds": median * iqr_fraction,
        "iqr_fraction": iqr_fraction,
    }


def _passing_summaries():
    return {
        "first_fit": {
            experiment.CONTROL: _summary(3.0),
            experiment.CANDIDATE: _summary(1.5),
        },
        "first_predict": {
            experiment.CONTROL: _summary(0.1),
            experiment.CANDIDATE: _summary(0.01),
        },
        "warmup": {
            experiment.CANDIDATE: _summary(5.0),
        },
        "import": {
            experiment.CONTROL: _summary(0.5),
            experiment.CANDIDATE: _summary(0.5),
        },
    }


def _analyze(summaries):
    return experiment.analyze(
        summaries=summaries,
        paired_fit_ratio=_summary(0.5),
        all_outputs_exact=True,
        behavior_fingerprints={"one"},
        import_cold=True,
        state_clean=True,
        caches_isolated=True,
    )


def test_schedule_is_six_reciprocal_blocks():
    assert experiment.schedule() == (
        (experiment.CONTROL, experiment.CANDIDATE),
        (experiment.CANDIDATE, experiment.CONTROL),
        (experiment.CONTROL, experiment.CANDIDATE),
        (experiment.CANDIDATE, experiment.CONTROL),
        (experiment.CONTROL, experiment.CANDIDATE),
        (experiment.CANDIDATE, experiment.CONTROL),
    )


def test_timing_summary_is_fail_closed():
    summary = experiment.timing_summary([1, 1, 1, 1, 1, 1])
    assert summary["median_seconds"] == 1.0
    assert summary["iqr_fraction"] == 0.0
    with pytest.raises(RuntimeError, match="exactly 6"):
        experiment.timing_summary([1, 1])
    with pytest.raises(RuntimeError, match="positive and finite"):
        experiment.timing_summary([1, 1, 1, 1, 1, 0])


def test_passing_analysis_ships_only_explicit_warmup():
    decision = _analyze(_passing_summaries())
    assert decision["passed"] is True
    assert decision["recommendation"] == "ship_explicit_warmup"
    assert decision["model_default_change_authorized"] is False
    assert decision["hidden_import_warmup_authorized"] is False
    assert all(decision["gates"].values())


@pytest.mark.parametrize(
    ("section", "arm", "field", "value", "failed_gate"),
    [
        ("first_fit", experiment.CANDIDATE, "median_seconds", 2.2, "first_fit_speedup"),
        (
            "first_predict",
            experiment.CANDIDATE,
            "median_seconds",
            0.03,
            "first_predict_speedup",
        ),
        (
            "first_fit",
            experiment.CONTROL,
            "iqr_fraction",
            0.26,
            "first_fit_stable",
        ),
        (
            "first_predict",
            experiment.CANDIDATE,
            "iqr_fraction",
            0.51,
            "candidate_predict_stable",
        ),
        (
            "warmup",
            experiment.CANDIDATE,
            "maximum_seconds",
            15.1,
            "warmup_within_budget",
        ),
    ],
)
def test_analysis_rejects_each_timing_boundary(
    section, arm, field, value, failed_gate
):
    summaries = copy.deepcopy(_passing_summaries())
    summaries[section][arm][field] = value
    decision = _analyze(summaries)
    assert decision["passed"] is False
    assert decision["gates"][failed_gate] is False
    assert decision["recommendation"] == (
        "close_warmup_attempt_without_threshold_changes"
    )


def test_analysis_rejects_behavior_or_state_drift():
    kwargs = {
        "summaries": _passing_summaries(),
        "paired_fit_ratio": _summary(0.5),
        "all_outputs_exact": True,
        "behavior_fingerprints": {"one"},
        "import_cold": True,
        "state_clean": True,
        "caches_isolated": True,
    }
    for key, value, gate in (
        ("all_outputs_exact", False, "outputs_array_exact"),
        ("behavior_fingerprints", {"one", "two"}, "model_behavior_repeat_exact"),
        ("import_cold", False, "ordinary_import_stays_cold"),
        ("state_clean", False, "caller_state_preserved"),
        ("caches_isolated", False, "fresh_caches_isolated"),
    ):
        current = dict(kwargs)
        current[key] = value
        decision = experiment.analyze(**current)
        assert decision["passed"] is False
        assert decision["gates"][gate] is False
