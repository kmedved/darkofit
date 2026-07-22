from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks import check_t7b_automatic_l2_invariants as invariants


def _arm(marker="same"):
    return {
        "cases": {
            case: {
                "prediction_sha256": marker,
                "logical_booster_sha256": marker,
                "resolved_l2_leaf_reg": 3.0,
            }
            for case in invariants.NOOP_CASES
        }
    }


def test_noop_analysis_requires_prediction_and_state_exactness():
    result = invariants.analyze(_arm(), _arm())

    assert result["all_noop_cases_exact"] is True
    assert set(result["comparisons"]) == set(invariants.NOOP_CASES)

    candidate = _arm()
    candidate["cases"]["catboost_mae_auto"]["prediction_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="catboost_mae_auto"):
        invariants.analyze(_arm(), candidate)

    candidate = _arm()
    candidate["cases"]["hybrid_rmse_auto"]["logical_booster_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="hybrid_rmse_auto"):
        invariants.analyze(_arm(), candidate)


def test_main_parse_requires_complete_paths():
    with pytest.raises(SystemExit):
        invariants.parse_args([])

    parsed = invariants.parse_args(
        ["--control", "/control", "--candidate", "/candidate", "--output", "/out"]
    )
    assert parsed.worker is None


def test_output_is_external_and_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        invariants.campaign,
        "validate_sources",
        lambda *_: {"control": {}, "candidate": {}, "harness": {}},
    )
    monkeypatch.setattr(
        invariants,
        "_run_worker",
        lambda source, cache: _arm(),
    )
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        control=tmp_path / "control",
        candidate=tmp_path / "candidate",
        output=output,
    )

    assert invariants.run(args) == output
    with pytest.raises(FileExistsError):
        invariants.run(args)
    with pytest.raises(ValueError, match="outside"):
        invariants.run(
            SimpleNamespace(
                control=tmp_path / "control",
                candidate=tmp_path / "candidate",
                output=invariants.ROOT / "inside.json",
            )
        )
