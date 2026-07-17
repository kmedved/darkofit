from __future__ import annotations

import json

import pytest

from benchmarks import run_basketball_random_strength as experiment


def _scores(mean: float) -> list[float]:
    return [mean] * experiment.creator.N_SPLITS


def _result(
    config: str,
    *,
    mean: float,
    held: float = 0.2,
    seen: float = 0.2,
    cold: float = 0.1,
) -> dict:
    return {
        "config": config,
        "mean_r2": mean,
        "fold_scores": _scores(mean),
        "holdout": {
            "scores": {
                "overlap_exposed_team_holdout": {"r2": held},
                "seen_player_subset": {"r2": seen},
                "cold_player_subset": {"r2": cold},
            }
        },
    }


def test_quality_gate_requires_broad_material_gain():
    control = _result(
        experiment.CONTROL,
        mean=experiment.EXPECTED_CONTROL_MEAN_R2,
    )
    passing = _result(
        "candidate",
        mean=experiment.EXPECTED_CONTROL_MEAN_R2 + 0.002,
        held=0.201,
        seen=0.201,
        cold=0.101,
    )
    decision = experiment.analyze_quality(control, passing)
    assert decision["passes_quality_gates"]
    assert decision["fold_wins"] == 10

    cold_regression = _result(
        "candidate",
        mean=experiment.EXPECTED_CONTROL_MEAN_R2 + 0.003,
        held=0.201,
        cold=0.099,
    )
    decision = experiment.analyze_quality(control, cold_regression)
    assert not decision["passes_quality_gates"]
    assert not decision["quality_gates"]["cold_player_no_regression"]


def test_quality_gate_rejects_control_drift():
    control = _result(experiment.CONTROL, mean=0.0)
    candidate = _result("candidate", mean=0.01)
    with pytest.raises(RuntimeError, match="control no longer reproduces"):
        experiment.analyze_quality(control, candidate)


def test_parse_args_rejects_nonpositive_threads():
    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "0"])


def test_parent_refuses_existing_output(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("{}")
    args = experiment.parse_args(["--output", str(output)])
    with pytest.raises(RuntimeError, match="refusing to replace"):
        experiment.run_parent(args)


def test_worker_process_requires_one_result(monkeypatch, tmp_path):
    class Completed:
        returncode = 0
        stdout = experiment.WORKER_RESULT_PREFIX + json.dumps({"ok": True})
        stderr = ""

    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: Completed())
    args = experiment.parse_args(
        ["--output", str(tmp_path / "out.json"), "--threads", "2"]
    )
    assert experiment._run_worker_process(args, experiment.CONTROL)["ok"]
