from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks import check_t7b_automatic_depth_invariants as invariants


EXPECTED = {
    "default_low_density": (4, 200.0, 8, 25.0, "low_density"),
    "default_middle_density": (6, 200.0, 2, 100.0, "middle_density"),
    "default_high_density": (8, 2_500.0, 1, 2_500.0, "high_density"),
}


def _arm(*, candidate: bool, marker: str = "same"):
    cases = {
        case: {
            "prediction_sha256": marker,
            "logical_booster_sha256": marker,
            "requested_depth": 6,
            "resolved_depth": 6,
            "resolved_l2_leaf_reg": 3.0,
            "auto_structure": {},
        }
        for case in invariants.NOOP_CASES
    }
    for case, values in EXPECTED.items():
        expected_depth, n_eff, n_features, density, branch = values
        depth = expected_depth if candidate else 6
        cases[case] = {
            "prediction_sha256": f"{case}-{depth}",
            "logical_booster_sha256": f"{case}-{depth}",
            "requested_depth": None,
            "resolved_depth": depth,
            "resolved_l2_leaf_reg": 3.0,
            "auto_structure": {
                "resolved": {
                    "depth": {
                        "input": None,
                        "resolved": depth,
                        "source": "auto" if candidate else "default",
                    }
                },
                "candidates": {
                    "depth": {
                        "rule": invariants.DEPTH_RULE,
                        "branch": branch,
                        "n_eff": n_eff,
                        "input_feature_count": n_features,
                        "effective_rows_per_feature": density,
                        "low_threshold": 100.0,
                        "high_threshold": 2_500.0,
                    }
                },
            },
        }
    return {"cases": cases}


def test_analysis_requires_noop_exactness_and_all_three_depth_branches():
    result = invariants.analyze(
        _arm(candidate=False), _arm(candidate=True)
    )

    assert result["all_noop_cases_exact"] is True
    assert result["all_depth_branches_engaged"] is True
    assert set(result["comparisons"]) == set(invariants.NOOP_CASES)
    assert set(result["engagement"]) == set(invariants.ENGAGED_CASES)

    candidate = _arm(candidate=True)
    candidate["cases"]["catboost_mae_default"][
        "prediction_sha256"
    ] = "changed"
    with pytest.raises(RuntimeError, match="catboost_mae_default"):
        invariants.analyze(_arm(candidate=False), candidate)

    candidate = _arm(candidate=True)
    candidate["cases"]["default_high_density"]["resolved_depth"] = 7
    with pytest.raises(RuntimeError, match="default_high_density"):
        invariants.analyze(_arm(candidate=False), candidate)


def test_main_parse_requires_complete_paths():
    with pytest.raises(SystemExit):
        invariants.parse_args([])

    parsed = invariants.parse_args(
        [
            "--control",
            "/control",
            "--candidate",
            "/candidate",
            "--output",
            "/out",
        ]
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
        lambda source, cache: _arm(candidate=source.name == "candidate"),
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
