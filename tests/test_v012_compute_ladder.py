"""Tests for the light-regime v0.12 release compute ladder."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks import analyze_v012_compute_ladder as analysis
from benchmarks import run_v011_compute_ladder as historical_legacy
from benchmarks import run_v012_compute_ladder as campaign


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/run_v012_compute_ladder.py",
        "benchmarks/analyze_v012_compute_ladder.py",
    ],
)
def test_v012_compute_ladder_clis_bootstrap_from_isolated_invocations(
    relative, tmp_path
):
    result = subprocess.run(
        [sys.executable, "-I", str(campaign.ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v012_grid_is_complete_unique_and_position_balanced():
    grid = campaign.expected_ordered_grid()
    assert campaign.EXPECTED_COORDINATES == 39
    assert campaign.EXPECTED_WORKERS == 234
    assert len(grid) == len(set(grid)) == 234
    assert {(row[1], row[2]) for row in grid} == {(0, 0), (1, 1), (2, 2)}
    assert {row[3] for row in grid} == set(campaign.ARM_SPECS)
    audit = campaign.position_audit()
    assert all(sum(values) == 39 for values in audit.values())
    assert all(max(values) - min(values) == 1 for values in audit.values())
    assert campaign.ordered_grid_sha256() == (
        "7e4a658be81c5645ebaa0364ebef449ab8101e965cbb170c83c6337ba5cba946"
    )


def test_v012_public_points_and_source_pins_are_exact():
    assert campaign.DARKOFIT_VERSION == "0.12.0"
    assert campaign.DARKOFIT_COMMIT == (
        "a9eb4dbbf8af0e6db42e9ace433e7a267c80fca7"
    )
    assert campaign.CHIMERABOOST_VERSION == "0.23.0"
    assert campaign.CHIMERABOOST_COMMIT == (
        "6667843b8970454b0f582ffd1ab2be033989c578"
    )
    assert campaign.ARM_SPECS[campaign.DARKO_DEFAULT]["config"] == {}
    assert campaign.ARM_SPECS[campaign.DARKO_ACCURACY]["config"] == {
        "preset": "accuracy"
    }
    assert campaign.ARM_SPECS[campaign.DARKO_ENSEMBLE]["config"] == {
        "ensemble_mode": "v3",
        "n_ensembles": 8,
    }
    assert campaign.ARM_SPECS[campaign.CHIMERA_DEFAULT]["config"] == {}
    assert campaign.ARM_SPECS[campaign.CHIMERA_ACCURACY]["config"] == {"depth": 10}
    assert campaign.ARM_SPECS[campaign.CHIMERA_ENSEMBLE]["config"] == {
        "n_ensembles": 8
    }


def test_v012_legacy_worker_adapter_is_fully_repointed():
    assert campaign.legacy.DARKOFIT_VERSION == campaign.DARKOFIT_VERSION
    assert campaign.legacy.DARKOFIT_COMMIT == campaign.DARKOFIT_COMMIT
    assert campaign.legacy.CHIMERABOOST_VERSION == campaign.CHIMERABOOST_VERSION
    assert campaign.legacy.CHIMERABOOST_COMMIT == campaign.CHIMERABOOST_COMMIT
    assert campaign.legacy.ARM_SPECS is campaign.ARM_SPECS
    assert campaign.legacy.WORKER_ENVIRONMENT == campaign.WORKER_ENVIRONMENT
    assert campaign.legacy.WORKER_PREFIX == campaign.WORKER_PREFIX


def test_v012_adapters_do_not_mutate_historical_modules():
    assert historical_legacy.DARKOFIT_VERSION == "0.11.0"
    assert historical_legacy.CHIMERABOOST_VERSION == "0.20.0"
    from benchmarks import analyze_v011_compute_ladder as historical_analysis

    assert historical_analysis.campaign is historical_legacy
    assert analysis.legacy_analysis is not historical_analysis
    assert not hasattr(campaign, "CONTRACT_ID")
    assert analysis.legacy_analysis.campaign.CONTRACT_ID == campaign.RUN_ID
    assert analysis.PLANNED_ANALYZER_SHA256 == (
        "0cc073c3cb9493a6f6a32b2e2be85d942c318f56a4535ec2b8f720efa49cdbb2"
    )


def test_v012_worker_command_invokes_current_runner_without_retired_contract():
    args = Namespace(
        darkofit_source=Path("/tmp/darko"),
        chimeraboost_source=Path("/tmp/chimera"),
        tabarena_source=Path("/tmp/tabarena"),
    )
    command = campaign._worker_command(
        args,
        worker_index=0,
        arm=campaign.expected_ordered_grid()[0][3],
        parent_pid=123,
    )
    assert Path(command[1]).name == "run_v012_compute_ladder.py"
    assert "--contract" not in command
    assert command[command.index("--worker-index") + 1] == "0"


def test_v012_internal_worker_arguments_are_all_or_none():
    with pytest.raises(SystemExit):
        campaign.parse_args(["--worker-index", "0"])
    args = campaign.parse_args(
        [
            "--worker-index",
            "0",
            "--arm",
            campaign.expected_ordered_grid()[0][3],
            "--parent-pid",
            "123",
            "--worker-started-at",
            "2026-07-24T00:00:00+00:00",
        ]
    )
    assert args.worker_index == 0


def test_v012_protocol_names_current_scope_and_light_rerun_rule():
    text = campaign.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "DarkoFit `v0.12.0`" in text
    assert "ChimeraBoost `v0.23.0`" in text
    assert "39 coordinates × 6 arms = 234 fresh worker processes" in text
    assert "14 total CPU threads" in text
    assert "not a tuning panel" in text
    assert "not a tuning or shipping" not in text
    assert "TabArena placement" in text
    assert "A harness bug may be fixed and the benchmark rerun" in text
    assert "one-shot" not in text.lower()
    assert "contract" not in text.lower()


def test_v012_worker_kind_adapter_preserves_current_kind(monkeypatch):
    captured = {}

    def fake_verify(raw, **kwargs):
        captured.update(raw)
        return dict(raw), 42

    monkeypatch.setattr(analysis.legacy_analysis, "_verify_worker", fake_verify)
    normalized, parent = analysis._verify_worker(
        {"kind": "v012_compute_ladder_worker"},
        expected_index=0,
        parent_pid=None,
        manifest={},
    )
    assert captured["kind"] == "v011_compute_ladder_worker"
    assert normalized["kind"] == "v012_compute_ladder_worker"
    assert parent == 42


def test_v012_report_relabels_legacy_template(monkeypatch):
    template = "\n".join(
        [
            "# DarkoFit v0.11 release compute-ladder scoreboard",
            "",
            "Status: **spent, descriptive release evidence; no policy "
            "advancement is authorized.**",
            "",
            "pinned ChimeraBoost v0.20 default",
            "the predeclared point-estimate readout",
        ]
    )
    monkeypatch.setattr(
        analysis.legacy_analysis,
        "render_report",
        lambda summary: template,
    )
    report = analysis.render_report({})
    assert "DarkoFit v0.12" in report
    assert "ChimeraBoost v0.23" in report
    assert "not a tuning or shipping gate" in report
    assert "fixed-protocol point-estimate readout" in report
    assert "spent" not in report
    assert "The completed 234-worker measurement was not rerun or changed" in report
