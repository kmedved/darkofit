from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("test_m3a_runner", "benchmarks/run_m3a_wave1.py")
analyzer = _load("test_m3a_analyzer", "benchmarks/analyze_m3a_wave1.py")
freezer = _load("test_m3a_freezer", "benchmarks/freeze_m3a_wave1.py")


def test_m3a_creator_folds_are_deterministic_and_partition_rows():
    first = runner.creator_fold_records(211, 2014)
    second = runner.creator_fold_records(211, 2014)

    assert first == second
    assert len(first) == 10
    assert sorted(
        index for fold in first for index in fold["test_indices"]
    ) == list(range(211))
    for fold in first:
        assert set(fold["train_indices"]).isdisjoint(fold["test_indices"])


def test_m3a_creator_fold_seed_changes_by_season():
    left = runner.creator_fold_records(200, 2014)
    right = runner.creator_fold_records(200, 2015)

    assert runner._json_sha256(left) != runner._json_sha256(right)


def test_m3a_overlap_disclosures_match_group_usage():
    assert runner._uses_groups(runner.DARKO_SINGLE)
    assert runner._uses_groups(runner.DARKO_GROUP5)
    assert runner._uses_groups(runner.DARKO_GROUP8)
    assert runner._uses_groups(runner.CHIMERA_SINGLE)
    assert runner._uses_groups(runner.CHIMERA_ENSEMBLE8)
    assert not runner._uses_groups(runner.DARKO_ROW5)
    assert not runner._uses_groups(runner.DARKO_ROW8)


def test_m3a_general_slice_is_fixed_to_shipped_row_ensemble_context():
    assert runner.GENERAL_ARMS == (
        runner.DARKO_SINGLE,
        runner.DARKO_ROW8,
        runner.CHIMERA_SINGLE,
        runner.CHIMERA_ENSEMBLE8,
    )
    assert runner.DARKO_GROUP8 not in runner.GENERAL_ARMS


def test_m3a_season_cluster_summary_uses_three_season_units():
    rows = [
        {"season": season, "target": target, "ratio": ratio}
        for season, ratio in ((2014, 0.98), (2015, 1.0), (2016, 1.02))
        for target in ("a", "b", "c")
    ]

    result = analyzer.season_cluster_summary(
        rows, seed=20260720, resamples=10_000
    )

    assert result["cell_geometric_mean"] == pytest.approx(
        (0.98 * 1.0 * 1.02) ** (1.0 / 3.0)
    )
    assert result["season_ratios"] == pytest.approx(
        {"2014": 0.98, "2015": 1.0, "2016": 1.02}
    )
    assert result["cluster_bootstrap"]["clusters"] == 3
    assert result["cluster_bootstrap"]["descriptive_only"] is True


def test_m3a_season_cluster_summary_rejects_pseudoreplicated_shape():
    rows = [
        {"season": 2014, "target": "a", "ratio": 1.0},
        {"season": 2015, "target": "a", "ratio": 1.0},
        {"season": 2016, "target": "a", "ratio": 1.0},
    ]

    with pytest.raises(RuntimeError, match="three targets x seasons"):
        analyzer.season_cluster_summary(rows, seed=0, resamples=100)


def test_m3a_survival_checks_are_conjunctive():
    contract = {
        "survival_gates": {
            "player_geomean_at_most": 0.995,
            "player_cluster_p95_at_most": 1.0,
            "held_geomean_at_most": 1.005,
            "cold_geomean_at_most": 1.005,
            "worst_season_at_most": 1.01,
            "worst_player_cell_at_most": 1.03,
            "fit_ratio_at_most": 9.0,
            "predict_ratio_at_most": 9.0,
            "model_bytes_ratio_at_most": 9.0,
            "peak_rss_ratio_at_most": 4.0,
        }
    }
    pair = {
        "views": {
            "player_disjoint": {
                "cell_geometric_mean": 0.994,
                "cell_max": 1.01,
                "season_ratios": {
                    "2014": 0.99,
                    "2015": 0.995,
                    "2016": 1.001,
                },
                "cluster_bootstrap": {"p95": 0.999},
            },
            "held_team": {"cell_geometric_mean": 1.0},
            "cold_player": {"cell_geometric_mean": 1.0},
        }
    }
    costs = {
        "fit_seconds": {"ratio": 8.0},
        "predict_seconds": {"ratio": 8.0},
        "held_median_model_bytes": {"ratio": 8.0},
        "aggregate_peak_rss_bytes": {"ratio": 3.0},
    }

    passing = analyzer._checks(pair, costs, contract, True)
    assert all(record["passed"] for record in passing.values())

    failing = analyzer._checks(pair, costs, contract, False)
    assert failing["integrity"]["passed"] is False
    assert not all(record["passed"] for record in failing.values())


def test_m3a_contract_builder_contains_medium_general_cells_and_mixed_verdict():
    contract = freezer.build_contract()

    assert contract["general"]["size"] == "medium"
    assert contract["general"]["rows"] == 10_000
    assert contract["general"]["seeds"] == [0, 1]
    assert contract["execution"]["quality_first"] is True
    assert len(contract["execution"]["orders"]["primary-repeats"]) == 2
    assert contract["claims"]["m6_ranking_authorized"] is False


def test_m3a_contract_creator_hashes_match_runner():
    contract = freezer.build_contract()
    for season, record in contract["creator_folds"]["seasons"].items():
        folds = runner.creator_fold_records(record["rows"], int(season))
        assert runner._json_sha256(folds) == record["sha256"]


def test_m3a_atomic_create_refuses_overwrite(tmp_path):
    output = tmp_path / "artifact.json"
    runner._atomic_create(output, b"one")

    with pytest.raises(FileExistsError):
        runner._atomic_create(output, b"two")
    assert output.read_bytes() == b"one"


def test_m3a_all_public_arm_plumbing_smoke(monkeypatch):
    darko_source = runner.DEFAULT_DARKO_SOURCE
    chimera_source = runner.DEFAULT_CHIMERA_SOURCE
    if not darko_source.is_dir() or not chimera_source.is_dir():
        pytest.skip("exact M3a source trees are unavailable")
    for package, source in (
        ("darkofit", darko_source),
        ("chimeraboost", chimera_source),
    ):
        module_file = getattr(sys.modules.get(package), "__file__", None)
        if module_file is not None and not runner._path_is_under(
            Path(module_file).resolve(), source
        ):
            pytest.skip(
                f"{package} already imported outside the pinned M3a source "
                "tree; this in-process smoke needs a fresh interpreter"
            )

    original = runner._build_estimator

    def small_estimator(arm, darkofit_source, chimeraboost_source):
        model, source = original(
            arm, darkofit_source, chimeraboost_source
        )
        if arm.startswith("darkofit_"):
            model.set_params(iterations=3, early_stopping=False)
        else:
            model.set_params(n_estimators=3, early_stopping=False)
            if model.n_ensembles:
                # The managed test sandbox blocks loky's semaphore limit
                # query. Formal M3a workers retain the public -1 scheduler.
                model.set_params(ensemble_n_jobs=1)
        return model, source

    monkeypatch.setattr(runner, "_build_estimator", small_estimator)
    rng = np.random.default_rng(20260720)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = pd.Series(
        X["a"].to_numpy() - 0.5 * X["b"].to_numpy()
        + rng.normal(0.0, 0.1, len(X))
    )
    groups = np.repeat(np.arange(30), 4)

    for arm in runner.ALL_ARMS:
        fitted = runner._fit_predict(
            arm,
            X.iloc[:100],
            y.iloc[:100],
            X.iloc[100:],
            darkofit_source=darko_source,
            chimeraboost_source=chimera_source,
            groups=groups[:100] if runner._uses_groups(arm) else None,
        )
        assert fitted["prediction"].shape == (20,)
        assert np.isfinite(fitted["prediction"]).all()
        assert fitted["model_bytes"] > 0
        assert fitted["fit_metadata"]["member_count"] in {1, 5, 8}


def test_m3a_frozen_contract_loads_when_present():
    path = ROOT / "benchmarks" / "m3a_wave1_contract.json"
    if not path.exists():
        pytest.skip("contract is created after the pre-freeze source commit")

    contract = runner.load_contract(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert contract == raw
    assert contract["outcomes_opened"] is False


def test_published_m3a_archive_has_exact_unique_grids_and_no_repeat_branch():
    contract_path = ROOT / "benchmarks" / "m3a_wave1_contract.json"
    artifact_path = ROOT / "benchmarks" / "m3a_wave1.json"
    contract = json.loads(contract_path.read_text())
    artifact = json.loads(artifact_path.read_text())

    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == (
        artifact["contract"]["sha256"]
    )
    for name in ("runner", "analyzer", "freezer"):
        record = contract["bound_files"][name]
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == (
            record["sha256"]
        )

    expected_sports = {
        (season, target)
        for season in contract["sports_panel"]["seasons"]
        for target in contract["sports_panel"]["targets"]
    }
    expected_general = {
        (dataset, contract["general"]["size"], seed)
        for dataset in contract["general"]["datasets"]
        for seed in contract["general"]["seeds"]
    }
    for key, phase_name in (
        ("primary_quality", "primary-quality"),
        ("diagnostics", "diagnostics"),
    ):
        phase = artifact["raw_phases"][key]
        expected_arms = tuple(
            arm
            for order in contract["execution"]["orders"][phase_name]
            for arm in order
        )
        assert tuple(row["arm"] for row in phase["results"]) == expected_arms
        for row in phase["results"]:
            sports = [
                (int(cell["season"]), str(cell["target"]))
                for cell in row["sports_cells"]
            ]
            assert len(sports) == len(set(sports))
            assert set(sports) == expected_sports
            general = [
                (cell["dataset"], cell["size"], int(cell["seed"]))
                for cell in row["general_cells"]
            ]
            assert len(general) == len(set(general))
            if row["arm"] in contract["general"]["arms"]:
                assert set(general) == expected_general
            else:
                assert general == []

    assert artifact["analysis"]["primary_decision"]["survives"] is False
    assert artifact["raw_phases"]["primary_repeats"] is None
    assert artifact["analysis"]["primary_timing"]["repeat_series_run"] is False
