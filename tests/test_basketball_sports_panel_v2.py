from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_basketball_sports_panel_v2 as analyzer
from benchmarks import build_basketball_sports_panel_v2 as panel
from benchmarks import run_basketball_sports_panel_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
RAW_ARTIFACT = ROOT / "benchmarks" / "basketball_sports_panel_v2_raw.json"
RESULT_ARTIFACT = ROOT / "benchmarks" / "basketball_sports_panel_v2_result.json"
REPORT_ARTIFACT = ROOT / "benchmarks" / "basketball_sports_panel_v2_result.md"


def _raw_like_frame() -> pd.DataFrame:
    rows = []
    for season in panel.SEASONS:
        for team_number in range(30):
            for game in range(2):
                value = float(team_number + game + 1)
                rows.append(
                    {
                        "Date": f"{season - 1}-11-{game + 1:02d}",
                        "Age": "25-183",
                        "Tm": f"T{team_number:02d}",
                        "GS": float(game == 0),
                        "Minutes": 260.0 + game,
                        "TS.": value / 100.0,
                        "eFG.": value / 101.0,
                        "ORB.": value / 10.0,
                        "DRB.": value / 9.0,
                        "TRB.": value / 8.0,
                        "AST.": value / 7.0,
                        "STL.": value / 6.0,
                        "BLK.": value / 5.0,
                        "TOV.": value / 4.0,
                        "USG.": value / 3.0,
                        "ORtg": 100.0 + value,
                        "DRtg": 110.0 - value,
                        "GmSc": value,
                        "BPM": value - 5.0,
                        "bref_id": f"p{team_number:02d}",
                        "Player": f"Player {team_number:02d}",
                        "year": season,
                    }
                )
    return pd.DataFrame(rows, columns=panel.RAW_COLUMNS)


def test_v2_panel_uses_only_fresh_seasons_and_middle_team_third():
    prepared = panel.prepare_panel(_raw_like_frame())
    assert tuple(sorted(prepared["year"].unique())) == panel.SEASONS
    for season in panel.SEASONS:
        assert panel.held_teams(prepared, season) == tuple(
            f"T{value:02d}" for value in range(10, 20)
        )


def test_v2_split_manifest_is_player_disjoint_and_partitions_rows():
    prepared = panel.prepare_panel(_raw_like_frame())
    manifest = panel.split_manifest(prepared)
    for season in panel.SEASONS:
        record = manifest["seasons"][str(season)]
        assert record["primary_rows"] == 20
        assert record["held_team_rows"] == 10
        assert record["seen_player_rows"] == 0
        assert record["cold_player_rows"] == 10
        observed = []
        primary = prepared.loc[
            (prepared["year"] == season) & ~prepared["Tm"].isin(record["held_teams"])
        ].reset_index(drop=True)
        groups = primary["bref_id"].astype(str).to_numpy()
        for fold in record["folds"]:
            train = np.asarray(fold["train_indices"], dtype=np.int64)
            test = np.asarray(fold["test_indices"], dtype=np.int64)
            assert set(groups[train]).isdisjoint(groups[test])
            observed.extend(test.tolist())
        assert sorted(observed) == list(range(len(primary)))


def test_v2_power_analysis_is_deterministic_and_sufficient():
    first = panel.power_analysis()
    second = panel.power_analysis()
    assert first == second
    assert first["cells"] == 9
    assert first["bootstrap_resamples"] == 2_000
    assert first["source_geometric_mean_rmse_ratio"] == pytest.approx(
        0.9967283642622231
    )
    assert 0.84 <= first["pass_probability"] <= 0.87
    assert first["passes"] is True


def _result(arm: str, ratio: float) -> dict:
    def score() -> dict:
        return {"rows": 10, "rmse": ratio, "r2": 0.0}

    cells = []
    for season in panel.SEASONS:
        for target in panel.TARGET_COLUMNS:
            cells.append(
                {
                    "season": season,
                    "target": target,
                    "primary": score(),
                    "guardrail": {
                        "scores": {
                            "held_team": score(),
                            "seen_player": score(),
                            "cold_player": score(),
                        }
                    },
                }
            )
    return {"arm": arm, "cells": cells}


def test_v2_quality_rule_credits_safe_aggregate_improvement():
    comparison = analyzer._quality_comparison(
        _result("candidate", 0.995),
        _result("control", 1.0),
    )
    assert comparison["aggregate_rmse_ratio"] == pytest.approx(0.995)
    assert comparison["bootstrap_95_upper"] == pytest.approx(0.995)
    assert comparison["passes_quality"] is True


def test_v2_quality_rule_rejects_single_lineage_harm():
    candidate = _result("candidate", 0.99)
    candidate["cells"][0]["primary"]["rmse"] = 1.03
    comparison = analyzer._quality_comparison(
        candidate,
        _result("control", 1.0),
    )
    assert comparison["worst_lineage_ratio"] == pytest.approx(1.03)
    assert comparison["primary_gates"]["worst_lineage_at_most_1_020"] is False
    assert comparison["passes_quality"] is False


def test_v2_analyzer_requires_distinct_raw_and_output_paths(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    with pytest.raises(RuntimeError, match="must be distinct"):
        analyzer._validate_paths(raw, raw, tmp_path / "report.md")


def test_v2_analyzer_cli_imports_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(Path(analyzer.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_v2_analyzer_publishes_outputs_as_a_rollback_safe_pair(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    original_link = analyzer.os.link
    calls = 0

    def fail_second_link(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-output failure")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(analyzer.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected"):
        analyzer._atomic_create_many(
            {
                output: b"json",
                report: b"markdown",
            }
        )
    assert not output.exists()
    assert not report.exists()
    assert list(tmp_path.iterdir()) == []


def test_v2_analyzer_pair_is_create_only(tmp_path):
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing existing"):
        analyzer._atomic_create_many(
            {
                output: b"replacement",
                report: b"markdown",
            }
        )
    assert output.read_bytes() == b"existing"
    assert not report.exists()


def test_v2_analyzer_pair_publishes_successfully(tmp_path):
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    analyzer._atomic_create_many(
        {
            output: b"json",
            report: b"markdown",
        }
    )
    assert output.read_bytes() == b"json"
    assert report.read_bytes() == b"markdown"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "result.json",
        "result.md",
    ]


def test_v2_analyzer_close_cleanup_does_not_report_failed_commit(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result.json"
    original_close = analyzer.os.close
    failed_descriptor = None

    def fail_first_close(descriptor):
        nonlocal failed_descriptor
        if failed_descriptor is None:
            failed_descriptor = descriptor
            raise OSError("injected close cleanup failure")
        return original_close(descriptor)

    monkeypatch.setattr(analyzer.os, "close", fail_first_close)
    analyzer._atomic_create_many({output: b"json"})
    assert output.read_bytes() == b"json"
    assert failed_descriptor is not None
    original_close(failed_descriptor)


def test_v2_analyzer_rollback_removes_owned_nested_directories(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "json" / "nested" / "result.json"
    report = tmp_path / "markdown" / "nested" / "result.md"
    original_link = analyzer.os.link
    calls = 0

    def fail_second_link(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected nested-output failure")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(analyzer.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected nested-output failure"):
        analyzer._atomic_create_many(
            {
                output: b"json",
                report: b"markdown",
            }
        )
    assert list(tmp_path.iterdir()) == []


def test_v2_analyzer_rolls_back_partial_directory_creation(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "first" / "second" / "result.json"
    original_mkdir = Path.mkdir
    calls = 0

    def fail_second_directory(path, *args, **kwargs):
        nonlocal calls
        if tmp_path in path.parents:
            calls += 1
            if calls == 2:
                raise OSError("injected directory-creation failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_second_directory)
    with pytest.raises(OSError, match="injected directory-creation failure"):
        analyzer._atomic_create_many({output: b"json"})
    assert list(tmp_path.iterdir()) == []


def test_v2_analyzer_rollback_preserves_another_writers_replacement(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    original_link = analyzer.os.link
    calls = 0

    def replace_first_then_fail(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            output.unlink()
            output.write_bytes(b"replacement")
            raise OSError("injected second-output failure")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(analyzer.os, "link", replace_first_then_fail)
    with pytest.raises(OSError, match="injected"):
        analyzer._atomic_create_many(
            {
                output: b"json",
                report: b"markdown",
            }
        )
    assert output.read_bytes() == b"replacement"
    assert not report.exists()


def test_v2_analyzer_rolls_back_pair_after_temp_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result.json"
    report = tmp_path / "result.md"
    original_unlink = analyzer.os.unlink
    failed = False

    def fail_first_temp_cleanup(path, *args, **kwargs):
        nonlocal failed
        if str(path).endswith(".tmp") and not failed:
            failed = True
            raise OSError("injected temp cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(analyzer.os, "unlink", fail_first_temp_cleanup)
    with pytest.raises(OSError, match="injected temp cleanup failure"):
        analyzer._atomic_create_many(
            {
                output: b"json",
                report: b"markdown",
            }
        )
    assert not output.exists()
    assert not report.exists()
    assert list(tmp_path.iterdir()) == []


def test_v2_analyzer_rejects_symlink_output_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink analyzer output"):
        analyzer._atomic_create_many(
            {linked / "missing" / "result.json": b"json"}
        )
    assert list(real.iterdir()) == []


def test_v2_analyzer_detects_symlink_parent_swap_race(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    output = parent / "result.json"
    original_link = analyzer.os.link

    def swap_parent_then_link(source, destination, **kwargs):
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(analyzer.os, "link", swap_parent_then_link)
    with pytest.raises(RuntimeError, match="symlink analyzer output"):
        analyzer._atomic_create_many({output: b"json"})
    assert not output.exists()
    assert list(moved.iterdir()) == []


def test_v2_analyzer_pair_rejects_real_parent_replacement(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    output = parent / "result.json"
    report = parent / "result.md"
    original_link = analyzer.os.link

    def replace_parent_then_link(source, destination, **kwargs):
        parent.rename(moved)
        parent.mkdir()
        output.write_bytes(b"replacement json")
        report.write_bytes(b"replacement markdown")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(analyzer.os, "link", replace_parent_then_link)
    with pytest.raises(RuntimeError, match="analyzer output parent changed"):
        analyzer._atomic_create_many(
            {
                output: b"json",
                report: b"markdown",
            }
        )
    assert output.read_bytes() == b"replacement json"
    assert report.read_bytes() == b"replacement markdown"
    assert sorted(path.name for path in parent.iterdir()) == [
        "result.json",
        "result.md",
    ]
    assert list(moved.iterdir()) == []


@pytest.mark.parametrize("system_root", [Path("/tmp"), Path("/var/tmp")])
def test_v2_analyzer_allows_immutable_system_symlink_root(system_root):
    if not system_root.is_dir():
        pytest.skip(f"{system_root} is unavailable")
    directory = Path(
        analyzer.tempfile.mkdtemp(
            prefix="darkofit-sports-analyzer-",
            dir=system_root,
        )
    )
    output = directory / "result.json"
    try:
        analyzer._atomic_create_many({output: b"json"})
        assert output.read_bytes() == b"json"
    finally:
        output.unlink(missing_ok=True)
        directory.rmdir()


def test_v2_analyzer_rejects_unknown_schema():
    with pytest.raises(RuntimeError, match="unknown schema"):
        analyzer._canonical_results(
            {
                "schema_version": 2,
                "name": "darkofit_basketball_sports_panel_raw_v2",
            }
        )


def test_v2_analyzer_recomputes_behavior_fingerprints():
    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    arm = runner.CONTROL
    raw["behavior_fingerprints"][arm] = "forged"
    for record in raw["repeats"]:
        if record["arm"] == arm:
            record["result"]["behavior_fingerprint_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="fingerprint payload changed"):
        analyzer._canonical_results(raw)


def test_v2_analyzer_binds_raw_provenance_to_frozen_inputs():
    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    raw["runner"]["sha256"] = "forged"
    with pytest.raises(RuntimeError, match="runner hash changed"):
        analyzer.analyze(raw, "forged-raw-hash")


def test_v2_analyzer_rejects_a_different_well_formed_raw_hash():
    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="content hash changed"):
        analyzer.analyze(raw, "0" * 64)


def test_v2_analyzer_rejects_forged_source_and_fold_ledgers():
    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    raw["source"]["chimeraboost"]["head"] = "forged"
    with pytest.raises(RuntimeError, match="ChimeraBoost source ledger"):
        analyzer.analyze(raw, "forged-raw-hash")

    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    for record in raw["repeats"]:
        fold = record["result"]["cells"][0]["folds"][0]
        fold["train_indices"].append(fold["test_indices"][0])
        fingerprint = analyzer.harness.behavior_fingerprint(
            runner._behavior_payload(record["result"])
        )
        record["result"]["behavior_fingerprint_sha256"] = fingerprint
        raw["behavior_fingerprints"][record["arm"]] = fingerprint
    with pytest.raises(RuntimeError, match="fold ledger"):
        analyzer.analyze(raw, "forged-raw-hash")


def test_v2_analyzer_rejects_invalid_score_and_timing_domains():
    candidate = _result("candidate", 1.0)
    control = _result("control", 1.0)
    candidate["cells"][0]["guardrail"]["scores"]["seen_player"]["rmse"] = -1.0
    control["cells"][0]["guardrail"]["scores"]["seen_player"]["rmse"] = -1.0
    with pytest.raises(RuntimeError, match="finite and positive"):
        analyzer._quality_comparison(candidate, control)

    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    for record in raw["repeats"]:
        record["result"]["total_fit_seconds"] *= -1.0
    with pytest.raises(RuntimeError, match="finite and positive"):
        analyzer._timing_analysis(raw)

    candidate = _result("candidate", 1.0)
    control = _result("control", 1.0)
    candidate["cells"][0]["primary"]["rmse"] = "1.0"
    with pytest.raises(RuntimeError, match="finite and positive"):
        analyzer._quality_comparison(candidate, control)

    candidate = _result("candidate", 1.0)
    control = _result("control", 1.0)
    candidate["cells"][0]["primary"]["r2"] = 1.01
    with pytest.raises(RuntimeError, match="at most 1"):
        analyzer._quality_comparison(candidate, control)

    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    raw["repeats"][0]["block"] = "0"
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        analyzer._canonical_results(raw)

    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    for record in raw["repeats"]:
        record["result"]["total_fit_seconds"] = str(
            record["result"]["total_fit_seconds"]
        )
    with pytest.raises(RuntimeError, match="finite and positive"):
        analyzer._timing_analysis(raw)


def test_v2_analyzer_rejects_malformed_nested_guardrail_ledgers():
    candidate = _result("candidate", 1.0)
    control = _result("control", 1.0)
    candidate["cells"][0]["guardrail"] = []
    with pytest.raises(RuntimeError, match="guardrail ledger is invalid"):
        analyzer._quality_comparison(candidate, control)

    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    manifest = json.loads(
        (
            ROOT / raw["panel_manifest"]["path"]
        ).read_text(encoding="utf-8")
    )
    raw["repeats"][0]["result"]["cells"][0]["guardrail"] = []
    with pytest.raises(RuntimeError, match="score-row ledger changed"):
        analyzer._validate_manifest_evidence(raw, manifest)


def test_v2_recorded_artifacts_reproduce_end_to_end(assert_analysis_equal):
    raw_sha256 = analyzer._sha256(RAW_ARTIFACT)
    raw = json.loads(RAW_ARTIFACT.read_text(encoding="utf-8"))
    stored = json.loads(RESULT_ARTIFACT.read_text(encoding="utf-8"))
    regenerated = analyzer.analyze(raw, raw_sha256)
    manifest_path = ROOT / raw["panel_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stored["raw"]["sha256"] == raw_sha256
    assert raw["runner"]["sha256"] == analyzer._sha256(
        ROOT / raw["runner"]["path"]
    )
    assert raw["protocol"]["sha256"] == analyzer._sha256(
        ROOT / raw["protocol"]["path"]
    )
    assert raw["panel_manifest"]["file_sha256"] == analyzer._sha256(
        manifest_path
    )
    assert manifest["builder"]["sha256"] == analyzer._sha256(
        ROOT / manifest["builder"]["path"]
    )
    assert manifest["builder"]["shared_builder_sha256"] == analyzer._sha256(
        ROOT / manifest["builder"]["shared_builder_path"]
    )
    assert_analysis_equal(stored, regenerated)
    assert analyzer.render_report(stored) == REPORT_ARTIFACT.read_text(
        encoding="utf-8"
    )


def test_v2_runner_freezes_public_row_oob_ensemble():
    model = runner.build_estimator(
        runner.CANDIDATE,
        runner.DEFAULT_CHIMERABOOST_REPO,
    )
    assert model.n_ensembles == 5
    assert model.ensemble_bootstrap == "rows"
    assert model.ensemble_shared_preprocessing is True
    assert model.random_state == 4
    assert model.thread_count == 18


def test_v2_ratio_summary_uses_seconds_scale_paired_ratios():
    summary = analyzer._ratio_summary([2.0, 2.1, 1.9])
    assert math.isclose(summary["median"], 2.0)
    assert summary["stable"] is True
