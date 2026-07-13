import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import benchmarks.analyze_tabarena_regression_cap_horizon as cap_analysis
from benchmarks.analyze_tabarena_regression_cap_horizon import (
    REQUIRED_FIT_METADATA,
    REQUIRED_REFIT_PARAMS,
    _atomic_write_bytes,
    _atomic_write_group,
    _assert_campaign_snapshot_unchanged,
    _canonical_json,
    _canonical_output_targets,
    _current_runtime_provenance,
    _protected_campaign_paths,
    _verify_repository_source,
    _verify_runtime_provenance,
    _verify_history_artifact,
    analyze_paired_rows,
    exact_one_sided_sign_test_pvalue,
    hierarchical_bootstrap_log_ratios,
    hierarchical_point_log_ratio,
    load_safe_rows,
    parse_result_record,
    render_markdown_report,
    verify_campaign_integrity,
)
from benchmarks.run_tabarena_regression_cap_horizon import (
    ANALYSIS_PAYLOAD_FILENAME,
    COMPLETION_ATTESTATION_FILENAME,
    HORIZON_ARMS,
    MANIFEST_FILENAME,
    SOURCE_FILES,
    expected_ag_ensemble_config,
    expected_child_hyperparameters,
    expected_fit_kwargs_extra,
    expected_resolved_method_hyperparameters,
)


def _split_panel(*, test_ratios=None, infer_ratio=1.05):
    test_ratios = test_ratios or {f"dataset_{index:02d}": 0.99 for index in range(13)}
    rows = []
    for dataset, test_ratio in test_ratios.items():
        for repeat in range(3):
            for fold in range(3):
                row = {"dataset": dataset, "repeat": repeat, "fold": fold}
                for metric, ratio in {
                    "test_rmse": test_ratio,
                    "val_rmse": 1.0,
                    "train_time_s": 1.5,
                    "infer_time_s": infer_ratio,
                    "peak_memory_bytes": 1.05,
                }.items():
                    row[f"{metric}_ratio"] = ratio
                    row[f"{metric}_log_ratio"] = math.log(ratio)
                    row[f"{metric}_pct"] = 100.0 * (ratio - 1.0)
                rows.append(row)
    return rows


def _child_pairs(split_rows, *, cap_reason="iteration_limit", long_reason="early_stopping"):
    rows = []
    for split in split_rows:
        row = {
            "dataset": split["dataset"],
            "repeat": split["repeat"],
            "fold": split["fold"],
            "cap1000_hit_cap": cap_reason == "iteration_limit",
            "cap10000_completed_over_1000": True,
        }
        for arm, reason, completed in (
            ("cap1000", cap_reason, 1_000),
            ("cap10000", long_reason, 1_100),
        ):
            row[f"{arm}_stop_reason"] = reason
            row[f"{arm}_best_iteration"] = completed - 10
            row[f"{arm}_rounds_completed"] = completed
            row[f"{arm}_resolved_learning_rate"] = 0.1
            row[f"{arm}_selected_tree_mode"] = "catboost"
            row[f"{arm}_selected_lane"] = "boosting"
        rows.append(row)
    return rows


def _refit_params(best):
    return {
        "iterations": best,
        "learning_rate": 0.1,
        "tree_mode": "catboost",
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
        "depth": 6,
        "num_leaves": None,
        "l2_leaf_reg": 3.0,
        "min_child_samples": 20,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }


def _ag_args_fit():
    return {
        "max_memory_usage_ratio": 1.0,
        "max_time_limit_ratio": 1.0,
        "max_time_limit": None,
        "min_time_limit": 0,
    }


def _fit_metadata(*, requested=1_000, completed=900, best=850):
    return {
        "iterations_requested": requested,
        "iterations_attempted": completed,
        "rounds_completed": completed,
        "rounds_retained": best,
        "best_iteration": best,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "early_stopping_rounds": 50,
        "stop_reason": "early_stopping",
        "wall_clock_limit_seconds": 100.0,
        "wall_clock_safety_margin_seconds": 5.0,
        "wall_clock_effective_seconds": 95.0,
        "wall_clock_elapsed_seconds": 10.0,
        "deadline_hit": False,
        "deadline_is_soft": True,
    }


def _result_record(arm="cap1000"):
    requested = HORIZON_ARMS[arm]["iterations"]
    completed = 900 if arm == "cap1000" else 1_100
    best = 850 if arm == "cap1000" else 1_050
    children = {}
    for index in range(1, 9):
        name = f"S1F{index}"
        children[name] = {
            "name": name,
            "model_type": "DarkoFitModel",
            "is_valid": True,
            "can_infer": True,
            "hyperparameters": expected_child_hyperparameters(arm, index - 1),
            "hyperparameters_user": dict(HORIZON_ARMS[arm]),
            "num_cpus": 18,
            "num_gpus": 0,
            "problem_type": "regression",
            "eval_metric": "root_mean_squared_error",
            "stopping_metric": "root_mean_squared_error",
            "val_in_fit": True,
            "unlabeled_in_fit": False,
            "ag_args_fit": _ag_args_fit(),
            "hyperparameters_fit": _refit_params(best),
            "darkofit_fit": _fit_metadata(
                requested=requested, completed=completed, best=best
            ),
        }
    return {
        "problem_type": "regression",
        "metric": "rmse",
        "metric_error": 1.0,
        "metric_error_val": 1.1,
        "time_train_s": 2.0,
        "time_infer_s": 0.1,
        "memory_usage": {"peak_mem_cpu": 1_000_000},
        "task_metadata": {
            "name": "toy",
            "tid": 7,
            "repeat": 0,
            "fold": 0,
            "split_idx": 0,
        },
        "framework": f"DarkoFit_c1_{arm}_horizon_BAG_L1",
        "experiment_metadata": {
            "experiment_cls": "OOFExperimentRunner",
            "method_cls": "AGSingleBagWrapper",
        },
        "method_metadata": {
            "hyperparameters": expected_resolved_method_hyperparameters(arm),
            "fit_kwargs_extra": expected_fit_kwargs_extra(18),
            "init_kwargs_extra": {},
            "model_cls": "DarkoFitModel",
            "model_type": "DARKO",
            "name_prefix": "DarkoFit",
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
            "fit_metadata": {
                "num_cpus": 18,
                "num_gpus": 0,
                "val_in_fit": False,
                "unlabeled_in_fit": False,
            },
            "model_hyperparameters": {
                **HORIZON_ARMS[arm],
                "ag_args": {"name_suffix": f"_c1_{arm}_horizon"},
                "ag_args_ensemble": expected_ag_ensemble_config(),
            },
            "info": {
                "is_valid": True,
                "can_infer": True,
                "model_type": "StackerEnsembleModel",
                "num_cpus": 18,
                "num_gpus": 0,
                "problem_type": "regression",
                "eval_metric": "root_mean_squared_error",
                "stopping_metric": "root_mean_squared_error",
                "val_in_fit": False,
                "unlabeled_in_fit": False,
                "bagged_info": {
                    "num_child_models": 8,
                    "child_model_type": "DarkoFitModel",
                    "child_model_names": [f"S1F{index}" for index in range(1, 9)],
                    "_n_repeats": 1,
                    "_k_per_n_repeat": [8],
                    "_random_state": 1,
                    "bagged_mode": True,
                    "child_hyperparameters_user": dict(HORIZON_ARMS[arm]),
                    "child_hyperparameters": expected_child_hyperparameters(arm, 0),
                    "child_ag_args_fit": _ag_args_fit(),
                    "child_hyperparameters_fit": _refit_params(best),
                },
                "children_info": children,
            },
        },
    }


def _safe_payload():
    outer_rows = []
    child_rows = []
    for arm in HORIZON_ARMS:
        outer, children = parse_result_record(
            _result_record(arm),
            source=f"experiments/{arm}/results.pkl",
            task_split_counts={"toy": (7, 1)},
        )
        outer.update(
            {
                "imputed": False,
                "experiment_cls": "OOFExperimentRunner",
                "method_cls": "AGSingleBagWrapper",
                "outer_model_type": "StackerEnsembleModel",
                "ag_ensemble": expected_ag_ensemble_config(),
            }
        )
        for child in children:
            child["refit_params"] = _refit_params(child["best_iteration"])
        outer_rows.append(outer)
        child_rows.extend(children)
    return {"outer_rows": outer_rows, "child_rows": child_rows}


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frozen_source_repository(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    for relative in SOURCE_FILES:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen bytes for {relative}\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "DarkoFit Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "frozen source")

    files = {}
    for relative in SOURCE_FILES:
        payload = (repository / relative).read_bytes()
        files[str(relative)] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob": _git(repository, "hash-object", str(repository / relative)),
        }
    source = {
        "repository": str(repository.resolve()),
        "git_head": _git(repository, "rev-parse", "HEAD"),
        "git_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "relevant_status": "",
        "files": files,
        "darkofit_import": {},
        "tabarena": {},
    }
    output_dir = repository / "generated-results"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text("generated\n", encoding="utf-8")
    return repository, output_dir, source


def _output_boundary_fixture(tmp_path):
    input_dir = tmp_path / "campaign"
    input_dir.mkdir()
    manifest = input_dir / MANIFEST_FILENAME
    attestation_path = input_dir / COMPLETION_ATTESTATION_FILENAME
    analysis_payload = input_dir / ANALYSIS_PAYLOAD_FILENAME
    warmup_history = input_dir / "warmup_history.json"
    resume_history = input_dir / "resume_history.json"
    result = input_dir / "experiments" / "job" / "results.pkl"
    result.parent.mkdir(parents=True)
    for path, payload in (
        (manifest, b"manifest"),
        (attestation_path, b"attestation"),
        (analysis_payload, b"analysis"),
        (warmup_history, b"warmup"),
        (resume_history, b"resume"),
        (result, b"result"),
    ):
        path.write_bytes(payload)
    attestation = {
        "result_artifacts": {
            str(result.relative_to(input_dir)): {
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                "size_bytes": result.stat().st_size,
            }
        }
    }
    protected = _protected_campaign_paths(
        input_dir,
        manifest_path=manifest,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    targets = {
        "split_csv": input_dir / "paired_splits.csv",
        "repeat_csv": input_dir / "per_repeat.csv",
        "child_csv": input_dir / "paired_children.csv",
        "summary_json": input_dir / "summary.json",
        "report_md": input_dir / "report.md",
    }
    return input_dir, targets, protected, {
        "manifest": manifest,
        "attestation": attestation_path,
        "analysis_payload": analysis_payload,
        "warmup_history": warmup_history,
        "resume_history": resume_history,
        "result": result,
    }


@pytest.mark.parametrize(
    "protected_name",
    [
        "manifest",
        "attestation",
        "analysis_payload",
        "warmup_history",
        "resume_history",
        "result",
    ],
)
def test_output_targets_reject_every_protected_campaign_artifact(
    tmp_path, protected_name
):
    input_dir, targets, protected, artifacts = _output_boundary_fixture(tmp_path)
    targets["summary_json"] = artifacts[protected_name]

    with pytest.raises(RuntimeError, match="protected campaign artifact"):
        _canonical_output_targets(
            input_dir,
            targets,
            protected_paths=protected,
        )


def test_output_targets_reject_output_dir_duplicates_directories_and_results_name(
    tmp_path,
):
    input_dir, targets, protected, _ = _output_boundary_fixture(tmp_path)

    targets["summary_json"] = input_dir
    with pytest.raises(RuntimeError, match="input directory itself"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)

    targets["summary_json"] = targets["split_csv"]
    with pytest.raises(RuntimeError, match="not distinct"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)

    directory_target = input_dir / "existing-directory"
    directory_target.mkdir()
    targets["summary_json"] = directory_target
    with pytest.raises(RuntimeError, match="regular-file target"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)

    targets["summary_json"] = input_dir / "not-attested" / "results.pkl"
    with pytest.raises(RuntimeError, match="protected results.pkl name"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)


def test_output_targets_reject_symlink_escape_and_protected_hard_link(tmp_path):
    input_dir, targets, protected, artifacts = _output_boundary_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = input_dir / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    targets["report_md"] = escape / "report.md"
    with pytest.raises(RuntimeError, match="strictly under"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)

    targets["report_md"] = input_dir / "report.md"
    protected_alias = input_dir / "manifest-alias.json"
    protected_alias.hardlink_to(artifacts["manifest"])
    targets["summary_json"] = protected_alias
    with pytest.raises(RuntimeError, match="aliases a protected"):
        _canonical_output_targets(input_dir, targets, protected_paths=protected)


def test_atomic_single_output_preserves_old_file_when_replace_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "summary.json"
    target.write_bytes(b"old summary")

    def fail_replace(source, destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(cap_analysis.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        _atomic_write_bytes(target, b"new summary")

    assert target.read_bytes() == b"old summary"
    assert list(tmp_path.glob(".summary.json.*")) == []


def test_atomic_decision_group_rolls_back_both_files_on_partial_failure(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    report = tmp_path / "report.md"
    summary.write_bytes(b"old summary")
    report.write_bytes(b"old report")
    original_replace = cap_analysis.os.replace
    injected = False

    def fail_second_install(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        if (
            not injected
            and destination == report
            and source.name.startswith(".report.md.")
            and source.suffix == ".tmp"
        ):
            injected = True
            raise OSError("synthetic decision publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(cap_analysis.os, "replace", fail_second_install)
    with pytest.raises(OSError, match="synthetic decision publication failure"):
        _atomic_write_group(
            [(summary, b"new summary"), (report, b"new report")]
        )

    assert injected is True
    assert summary.read_bytes() == b"old summary"
    assert report.read_bytes() == b"old report"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.backup")) == []


def test_atomic_decision_group_rolls_back_when_final_provenance_check_fails(
    tmp_path,
):
    summary = tmp_path / "summary.json"
    report = tmp_path / "report.md"

    def fail_final_check():
        raise RuntimeError("provenance changed")

    with pytest.raises(RuntimeError, match="provenance changed"):
        _atomic_write_group(
            [(summary, b"new summary"), (report, b"new report")],
            post_write_check=fail_final_check,
        )

    assert not summary.exists()
    assert not report.exists()
    assert list(tmp_path.iterdir()) == []


def test_campaign_snapshot_revalidation_rejects_changed_artifact_digest(
    tmp_path, monkeypatch
):
    manifest = {"manifest": "baseline"}
    attestation = {"attestation": "baseline"}
    analysis_payload = {"analysis": "baseline"}
    digests = {"manifest_sha256": "baseline"}

    monkeypatch.setattr(
        cap_analysis,
        "verify_campaign_integrity",
        lambda *args, **kwargs: (
            manifest,
            attestation,
            analysis_payload,
            {"manifest_sha256": "changed"},
        ),
    )

    with pytest.raises(RuntimeError, match="campaign artifacts.*changed"):
        _assert_campaign_snapshot_unchanged(
            tmp_path,
            manifest_path=tmp_path / "run_manifest.json",
            attestation_path=tmp_path / "completion_attestation.json",
            baseline_manifest=manifest,
            baseline_attestation=attestation,
            baseline_analysis_payload=analysis_payload,
            baseline_digests=digests,
        )


def test_hierarchical_point_and_bootstrap_are_seeded_and_equal_dataset():
    ratios = {f"dataset_{index:02d}": 0.98 + 0.001 * index for index in range(13)}
    rows = _split_panel(test_ratios=ratios)
    point, datasets, repeats = hierarchical_point_log_ratio(
        rows, "test_rmse_log_ratio"
    )
    expected = sum(math.log(value) for value in ratios.values()) / 13
    assert point == pytest.approx(expected)
    assert {round(math.exp(value), 12) for value in datasets.values()} == {
        round(value, 12) for value in ratios.values()
    }
    assert all(len(value) == 3 for value in repeats.values())

    first = hierarchical_bootstrap_log_ratios(rows, draws=128, seed=20260713)
    second = hierarchical_bootstrap_log_ratios(rows, draws=128, seed=20260713)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_frozen_sign_test_and_all_gates_pass_on_clear_synthetic_win():
    assert exact_one_sided_sign_test_pvalue(10, 3) == pytest.approx(378 / 8192)
    rows = _split_panel()
    summary = analyze_paired_rows(rows, _child_pairs(rows), draws=256)
    assert summary["primary_test"]["dataset_wins"] == 13
    assert summary["primary_test"]["ratio"] == pytest.approx(0.99)
    assert len(summary["repeats"]) == 39
    assert summary["repeats"][0]["ratio"] == pytest.approx(0.99)
    assert summary["repeats"][0]["fold_count"] == 3
    assert summary["child_metadata"]["cap1000"]["near_cap_threshold"] == 950
    assert summary["child_metadata"]["cap1000"]["at_cap_fraction"] == 1.0
    assert summary["child_metadata"]["cap1000"]["near_cap_fraction"] == 1.0
    assert summary["child_metadata"]["cap10000"]["near_cap_threshold"] == 9_500
    assert summary["child_metadata"]["cap10000"]["near_cap_fraction"] == 0.0
    for arm in ("cap1000", "cap10000"):
        diagnostics = summary["child_metadata"][arm]["stop_reason_diagnostics"]
        assert set(diagnostics) == {
            "iteration_limit",
            "early_stopping",
            "no_split",
            "time_limit",
        }
        assert all(item["denominator"] == len(rows) for item in diagnostics.values())
        assert sum(item["count"] for item in diagnostics.values()) == len(rows)
        assert sum(item["fraction"] for item in diagnostics.values()) == pytest.approx(1.0)
    assert summary["child_metadata"]["cap1000"]["stop_reason_diagnostics"][
        "no_split"
    ] == {"count": 0, "denominator": len(rows), "fraction": 0.0}
    assert summary["gates"]["advance"] is True
    assert all(summary["gates"].values())


def test_markdown_emits_every_diagnostic_before_the_final_decision():
    rows = _split_panel()
    summary = analyze_paired_rows(rows, _child_pairs(rows), draws=64)
    summary["integrity_diagnostics"] = {
        "validation_basis": "synthetic exact-grid validation",
        "expected_outer_results": 2 * len(rows),
        "observed_outer_results": 2 * len(rows),
        "missing_outer_results": 0,
        "failed_outer_results": 0,
        "imputed_outer_results": 0,
        "duplicate_outer_results": 0,
        "expected_child_fit_blocks": 2 * len(rows),
        "observed_child_fit_blocks": 2 * len(rows),
        "missing_child_fit_metadata": 0,
        "duplicate_child_fit_metadata": 0,
        "metadata_incomplete_child_fits": 0,
    }
    summary["provenance"] = {
        "manifest_sha256": "manifest",
        "attestation_sha256": "attestation",
        "analysis_payload_sha256": "payload",
        "protocol_sha256": "protocol",
        "git_head": "commit",
        "completed_at_utc": "2026-07-13T00:00:00Z",
        "manifest_path": "/tmp/run_manifest.json",
        "attestation_path": "/tmp/completion_attestation.json",
    }

    report = render_markdown_report(summary)

    ordered_sections = [
        "## Dataset estimates",
        "## Repeat estimates",
        "### Fit-iteration distributions",
        "### Resolved configuration diagnostics",
        "### Stop-reason diagnostics",
        "### At/near-cap diagnostics",
        "## Campaign integrity diagnostics",
        "## Frozen gates",
        "## Provenance",
        "## Decision",
    ]
    section_indexes = [report.index(section) for section in ordered_sections]
    assert section_indexes == sorted(section_indexes)
    decision_index = report.index("## Decision")
    assert report.rfind("## ") == decision_index
    assert report.endswith("**ADVANCE 10,000 rounds.**")

    bootstrap = summary["primary_test"]["bootstrap"]
    assert (
        "Hierarchical bootstrap two-sided 95% interval: "
        f"[{bootstrap['ratio_lower95_two_sided']:.6f}, "
        f"{bootstrap['ratio_upper95_two_sided']:.6f}]"
    ) in report
    sensitivity = summary["primary_test"]["t_interval_sensitivity"]
    assert (
        "Dataset-level t-interval sensitivity: "
        f"[{sensitivity['ratio_lower95']:.6f}, "
        f"{sensitivity['ratio_upper95']:.6f}] "
        f"(df={sensitivity['degrees_of_freedom']})"
    ) in report

    assert report.count("| dataset_") == len(summary["datasets"]) + len(
        summary["repeats"]
    )
    for item in summary["datasets"]:
        expected = (
            f"| {item['dataset']} | {item['ratio']:.6f} | "
            f"{item['repeat_wins']}/{item['repeat_losses']}/{item['repeat_ties']} | "
            f"{item['repeat_block_bootstrap_ratio_lower90']:.6f} | "
            f"{item['worst_split']} | {item['worst_split_ratio']:.6f} | "
            f"{'yes' if item['conditional_harm'] else 'no'} |"
        )
        assert expected in report
    for item in summary["repeats"]:
        expected = (
            f"| {item['dataset']} | {item['repeat']} | {item['ratio']:.6f} | "
            f"{item['fold_wins']}/{item['fold_losses']}/{item['fold_ties']} | "
            f"f{item['worst_fold']} {item['worst_split_ratio']:.6f} |"
        )
        assert expected in report

    for arm in ("cap1000", "cap10000"):
        for field in ("best_iteration", "rounds_completed"):
            distribution = summary["child_metadata"][arm][field]
            expected = (
                f"| {arm} | `{field}` | {distribution['count']} | "
                f"{distribution['min']:.0f} | {distribution['median']:.0f} | "
                f"{distribution['p90']:.0f} | {distribution['max']:.0f} |"
            )
            assert expected in report
        for field in (
            "resolved_learning_rate_counts",
            "selected_tree_mode_counts",
            "selected_lane_counts",
        ):
            for value, count in summary["child_metadata"][arm][field].items():
                assert f"| {arm} | `{field}` | `{value}` | {count} |" in report
        for reason in (
            "iteration_limit",
            "early_stopping",
            "no_split",
            "time_limit",
        ):
            diagnostics = summary["child_metadata"][arm][
                "stop_reason_diagnostics"
            ][reason]
            expected = (
                f"| {arm} | `{reason}` | {diagnostics['count']} | "
                f"{diagnostics['denominator']} | {diagnostics['fraction']:.1%} |"
            )
            assert expected in report
        metadata = summary["child_metadata"][arm]
        assert (
            f"| {arm} | {metadata['at_cap_count']} / {metadata['child_fit_count']} "
            f"({metadata['at_cap_fraction']:.1%}) | {metadata['near_cap_count']} / "
            f"{metadata['child_fit_count']} ({metadata['near_cap_fraction']:.1%}) | "
            f"{metadata['near_cap_threshold']} |"
        ) in report

    for label in (
        "Missing outer results",
        "Failed outer results",
        "Imputed outer results",
        "Duplicated outer results",
        "Missing child-fit metadata",
        "Duplicated child-fit metadata",
        "Metadata-incomplete child fits",
    ):
        assert f"| {label} | 0 |" in report
    for name, passed in summary["gates"].items():
        if name != "advance":
            assert f"| `{name}` | {'yes' if passed else '**no**'} |" in report
    for field, value in summary["provenance"].items():
        assert f"- `{field}`: `{value}`" in report


def test_frozen_gates_detect_dataset_harm_time_stop_and_inactive_cap():
    ratios = {f"dataset_{index:02d}": 0.98 for index in range(13)}
    ratios["dataset_12"] = 1.03
    rows = _split_panel(test_ratios=ratios, infer_ratio=1.2)
    children = _child_pairs(rows, cap_reason="early_stopping", long_reason="time_limit")
    summary = analyze_paired_rows(rows, children, draws=256)
    gates = summary["gates"]
    assert gates["no_conditional_dataset_harm"] is False
    assert gates["no_dataset_point_ratio_above_1_02"] is False
    assert gates["cap1000_has_iteration_limit_child"] is False
    assert gates["zero_time_limit_stops"] is False
    assert gates["inference_time_ratio_at_most_1_10"] is False
    assert gates["advance"] is False


def test_mechanism_fraction_uses_all_child_pairs_not_only_capped_pairs():
    rows = _split_panel()
    children = _child_pairs(rows, cap_reason="early_stopping")
    for child in children:
        child["cap10000_completed_over_1000"] = False
    children[0]["cap1000_hit_cap"] = True
    children[0]["cap1000_stop_reason"] = "iteration_limit"
    children[0]["cap10000_completed_over_1000"] = True

    summary = analyze_paired_rows(rows, children, draws=64)

    assert summary["mechanism"]["cap1000_iteration_limit_children"] == 1
    assert summary["mechanism"]["cap10000_children_over_1000_fraction"] == pytest.approx(
        1 / len(children)
    )
    assert summary["gates"][
        "at_least_20pct_paired_cap10000_children_exceed_1000"
    ] is False


def test_integrity_verification_never_decodes_raw_result_pickles(tmp_path, monkeypatch):
    input_dir = tmp_path.resolve()
    result_path = input_dir / "experiments" / "job" / "results.pkl"
    result_path.parent.mkdir(parents=True)
    # This is intentionally not a valid pickle. Integrity verification hashes
    # raw TabArena artifacts but analysis reads only the safe JSON snapshot.
    raw_result = b"\x80malicious-looking-but-never-decoded"
    result_path.write_bytes(raw_result)
    relative = str(result_path.relative_to(input_dir))
    artifacts = {
        relative: {
            "sha256": hashlib.sha256(raw_result).hexdigest(),
            "size_bytes": len(raw_result),
        }
    }
    protocol = {"frozen": True}
    protocol_digest = hashlib.sha256(_canonical_json(protocol)).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon",
        "output_dir": str(input_dir),
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "resolved_child_num_cpus": 2,
        "source": {"git_head": "trusted-local-run"},
    }
    manifest_path = input_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    safe_payload = {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon_analysis_payload",
        "protocol_sha256": protocol_digest,
        "result_artifacts_sha256": hashlib.sha256(
            _canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": [],
        "child_rows": [],
    }
    safe_path = input_dir / ANALYSIS_PAYLOAD_FILENAME
    safe_path.write_text(json.dumps(safe_payload), encoding="utf-8")
    safe_bytes = safe_path.read_bytes()
    attestation = {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon_completion",
        "result_count": 1,
        "expected_result_count": 1,
        "expected_child_fits": 0,
        "warmup_thread_count": 2,
        "protocol_sha256": protocol_digest,
        "git_head": "trusted-local-run",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "result_artifacts": artifacts,
        "analysis_payload_artifact": {
            "path": ANALYSIS_PAYLOAD_FILENAME,
            "sha256": hashlib.sha256(safe_bytes).hexdigest(),
            "size_bytes": len(safe_bytes),
        },
        "validation": {
            "result_count": 1,
            "child_fit_count": 0,
            "resource_allocation": {
                "num_cpus": 2,
                "num_gpus": 0,
                "num_cpus_child": 2,
                "num_gpus_child": 0,
            },
        },
    }
    (input_dir / "completion_attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )

    provenance_checks = []

    def record_execution_provenance(loaded_manifest, loaded_input_dir):
        provenance_checks.append((loaded_manifest, loaded_input_dir))
        return {
            "executing_source_verified": True,
            "analysis_runtime_verified": True,
            "dependency_provenance_verified": True,
        }

    monkeypatch.setattr(
        cap_analysis,
        "verify_execution_provenance",
        record_execution_provenance,
    )
    monkeypatch.setattr(
        cap_analysis,
        "_verify_history_artifact",
        lambda *args, required, **kwargs: "warmup-digest" if required else None,
    )

    _, _, loaded, _ = verify_campaign_integrity(
        input_dir,
        expected_protocol=protocol,
        expected_jobs=1,
        expected_child_fits=0,
    )
    assert loaded == safe_payload
    assert provenance_checks == [(manifest, input_dir)]


def test_history_artifact_verification_binds_bytes_and_schema_callback(tmp_path):
    path = tmp_path / "warmup_history.json"
    payload = b"[]"
    path.write_bytes(payload)
    attestation = {
        "warmup_history_artifact": {
            "path": "warmup_history.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    }
    observed = []
    assert _verify_history_artifact(
        tmp_path,
        attestation,
        attestation_field="warmup_history_artifact",
        filename="warmup_history.json",
        required=True,
        validator=observed.append,
    ) == hashlib.sha256(payload).hexdigest()
    assert observed == [[]]

    path.write_bytes(b"[{}]")
    with pytest.raises(RuntimeError, match="does not match"):
        _verify_history_artifact(
            tmp_path,
            attestation,
            attestation_field="warmup_history_artifact",
            filename="warmup_history.json",
            required=True,
            validator=lambda value: None,
        )


def test_execution_source_rejects_changed_analyzer_bytes(tmp_path):
    repository, output_dir, source = _frozen_source_repository(tmp_path)
    _verify_repository_source(source, output_dir, repository=repository)

    analyzer = repository / "benchmarks/analyze_tabarena_regression_cap_horizon.py"
    analyzer.write_text("changed analyzer bytes\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source SHA-256 mismatch.*analyze"):
        _verify_repository_source(source, output_dir, repository=repository)


def test_execution_source_rejects_dirty_unrecorded_code_but_not_run_output(
    tmp_path,
):
    repository, output_dir, source = _frozen_source_repository(tmp_path)

    # The generated output directory is expected to be untracked and is the
    # only untracked subtree excluded from the cleanliness check.
    _verify_repository_source(source, output_dir, repository=repository)

    (repository / "unrecorded_analysis.py").write_text(
        "decision = 'changed'\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="dirty or unrecorded code"):
        _verify_repository_source(source, output_dir, repository=repository)


def test_execution_source_rejects_a_later_commit(tmp_path):
    repository, output_dir, source = _frozen_source_repository(tmp_path)
    (repository / "unrelated_tracked_file.txt").write_text("later\n", encoding="utf-8")
    _git(repository, "add", "unrelated_tracked_file.txt")
    _git(repository, "commit", "-m", "later commit")

    with pytest.raises(RuntimeError, match="Git HEAD does not match"):
        _verify_repository_source(source, output_dir, repository=repository)


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        ("numpy", "runtime package numpy"),
        ("psutil", "runtime package psutil"),
    ],
)
def test_analysis_runtime_rejects_package_version_changes(distribution, expected):
    recorded = _current_runtime_provenance()
    mutated = copy.deepcopy(recorded)
    mutated["packages"][distribution] = "0.0.0-mutated"

    with pytest.raises(RuntimeError, match=expected):
        _verify_runtime_provenance(mutated)


def test_analysis_runtime_rejects_python_version_changes():
    recorded = _current_runtime_provenance()
    mutated = copy.deepcopy(recorded)
    mutated["python_version"] = "0.0.0-mutated"

    with pytest.raises(RuntimeError, match="runtime python_version"):
        _verify_runtime_provenance(mutated)


def test_analysis_runtime_rejects_different_host_hardware():
    recorded = _current_runtime_provenance()
    mutated = copy.deepcopy(recorded)
    mutated["hardware"]["host_identity_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="runtime hardware"):
        _verify_runtime_provenance(mutated)


@pytest.mark.parametrize(
    "key",
    [
        "NUMBA_THREADING_LAYER",
        "NUMBA_THREADING_LAYER_PRIORITY",
        "NUMBA_DISABLE_JIT",
        "NUMBA_CPU_NAME",
        "NUMBA_CPU_FEATURES",
        "NUMBA_OPT",
        "NUMBA_LOOP_VECTORIZE",
        "NUMBA_SLP_VECTORIZE",
        "NUMBA_ENABLE_AVX",
        "NUMBA_BOUNDSCHECK",
    ],
)
def test_analysis_runtime_rejects_numba_environment_changes(key):
    recorded = _current_runtime_provenance()
    mutated = copy.deepcopy(recorded)
    original = mutated["environment"][key]
    mutated["environment"][key] = f"{original!r}-mutated"

    with pytest.raises(RuntimeError, match=f"environment {key}"):
        _verify_runtime_provenance(mutated)


def test_analysis_runtime_rejects_environment_changes():
    recorded = _current_runtime_provenance()
    mutated = copy.deepcopy(recorded)
    mutated["environment"]["PYTHONHASHSEED"] = "mutated"

    with pytest.raises(RuntimeError, match="environment PYTHONHASHSEED"):
        _verify_runtime_provenance(mutated)


def test_result_parser_requires_complete_exact_child_fit_metadata():
    record = _result_record()
    outer, children = parse_result_record(
        record,
        source="synthetic",
        task_split_counts={"toy": (7, 1)},
    )
    assert outer["arm"] == "cap1000"
    assert len(children) == 8
    bagged = record["method_metadata"]["info"]["bagged_info"]
    first_child = record["method_metadata"]["info"]["children_info"]["S1F1"]
    assert set(bagged["child_hyperparameters_fit"]) == REQUIRED_REFIT_PARAMS
    assert set(first_child["darkofit_fit"]) == REQUIRED_FIT_METADATA

    del record["method_metadata"]["info"]["children_info"]["S1F4"][
        "darkofit_fit"
    ]["stop_reason"]
    with pytest.raises(RuntimeError, match="fit metadata incomplete"):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


@pytest.mark.parametrize(
    "mutation",
    ["resolved_budget", "bag_seed", "child_seed", "child_resources", "child_metric"],
)
def test_result_parser_rejects_mismatched_resolved_execution_semantics(mutation):
    record = _result_record()
    method = record["method_metadata"]
    bag = method["info"]["bagged_info"]
    child = method["info"]["children_info"]["S1F1"]
    if mutation == "resolved_budget":
        method["hyperparameters"]["ag_args_ensemble"]["ag_args_fit"][
            "max_time_limit"
        ] = 600.0
    elif mutation == "bag_seed":
        bag["_random_state"] = 0
    elif mutation == "child_seed":
        child["hyperparameters"]["random_state"] = 7
    elif mutation == "child_resources":
        child["num_cpus"] = 1
    elif mutation == "child_metric":
        child["stopping_metric"] = "mean_squared_error"
    with pytest.raises(RuntimeError):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


def test_result_parser_rejects_deadline_above_frozen_outer_budget():
    record = _result_record()
    fitted = record["method_metadata"]["info"]["children_info"]["S1F1"][
        "darkofit_fit"
    ]
    fitted["wall_clock_limit_seconds"] = 7_200.0
    fitted["wall_clock_safety_margin_seconds"] = 5.0
    fitted["wall_clock_effective_seconds"] = 7_195.0

    with pytest.raises(RuntimeError, match="deadline metadata"):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


def test_result_parser_rejects_impossible_iteration_limit():
    record = _result_record()
    record["method_metadata"]["info"]["children_info"]["S1F1"][
        "darkofit_fit"
    ]["stop_reason"] = "iteration_limit"

    with pytest.raises(RuntimeError, match="iteration_limit"):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


@pytest.mark.parametrize(
    "field",
    [
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
    ],
)
def test_result_parser_rejects_boolean_deadline_numbers(field):
    record = _result_record()
    record["method_metadata"]["info"]["children_info"]["S1F1"][
        "darkofit_fit"
    ][field] = True

    with pytest.raises(RuntimeError, match="must be numeric"):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


def test_result_parser_rejects_stale_compressed_refit_iterations():
    record = _result_record()
    record["method_metadata"]["info"]["bagged_info"][
        "child_hyperparameters_fit"
    ]["iterations"] += 1

    with pytest.raises(RuntimeError, match="aggregation"):
        parse_result_record(
            record,
            source="synthetic",
            task_split_counts={"toy": (7, 1)},
        )


def test_safe_analysis_rows_enforce_exact_seed_and_time_configuration():
    payload = _safe_payload()
    outer_rows = payload["outer_rows"]
    child_rows = payload["child_rows"]
    loaded_outer, loaded_children = load_safe_rows(
        payload,
        task_split_counts={"toy": (7, 1)},
    )
    assert len(loaded_outer) == 2
    assert len(loaded_children) == 16

    outer_rows[0]["ag_ensemble"]["vary_seed_across_folds"] = False
    with pytest.raises(RuntimeError, match="seed/resource configuration"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    outer_rows[0]["ag_ensemble"]["vary_seed_across_folds"] = True

    outer_rows[0]["imputed"] = True
    with pytest.raises(RuntimeError, match="imputation metadata"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    outer_rows[0]["imputed"] = False

    outer_rows[0]["resolved_method_hyperparameters"]["ag_args_ensemble"][
        "ag_args_fit"
    ]["max_time_limit"] = 600.0
    with pytest.raises(RuntimeError, match="resolved method configuration"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    outer_rows[0]["resolved_method_hyperparameters"] = (
        expected_resolved_method_hyperparameters("cap1000")
    )

    child_rows[0]["initial_hyperparameters"]["random_state"] = 7
    with pytest.raises(RuntimeError, match="initial policy or seed"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    child_rows[0]["initial_hyperparameters"] = expected_child_hyperparameters(
        "cap1000", 0
    )

    child_rows[0]["num_cpus"] = 1
    with pytest.raises(RuntimeError, match="resources/policy"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    child_rows[0]["num_cpus"] = 18

    child_rows[0]["iterations_attempted"] = 900.5
    with pytest.raises(RuntimeError, match="must be an integer"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
    child_rows[0]["iterations_attempted"] = 900

    child_rows[0]["wall_clock_limit_seconds"] = 7_200.0
    child_rows[0]["wall_clock_safety_margin_seconds"] = 5.0
    child_rows[0]["wall_clock_effective_seconds"] = 7_195.0
    with pytest.raises(RuntimeError, match="deadline metadata"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})


def test_safe_analysis_rows_reject_impossible_iteration_limit():
    payload = _safe_payload()
    payload["child_rows"][0]["stop_reason"] = "iteration_limit"

    with pytest.raises(RuntimeError, match="iteration_limit"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})


@pytest.mark.parametrize(
    "field",
    [
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
    ],
)
def test_safe_analysis_rows_reject_boolean_deadline_numbers(field):
    payload = _safe_payload()
    payload["child_rows"][0][field] = True

    with pytest.raises(RuntimeError, match="must be numeric"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})


def test_safe_analysis_rows_reject_stale_compressed_refit_iterations():
    payload = _safe_payload()
    payload["outer_rows"][0]["compressed_refit_params"]["iterations"] += 1

    with pytest.raises(RuntimeError, match="aggregation"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("iterations", 851),
        ("learning_rate", 0.2),
        ("tree_mode", "auto"),
        ("early_stopping", True),
        ("early_stopping_rounds", 50),
        ("use_best_model", True),
        ("refit", True),
        ("depth", 7),
        ("num_leaves", 64),
        ("l2_leaf_reg", 4.0),
        ("min_child_samples", 21),
        ("min_child_weight", 2.0),
        ("cat_smoothing", 2.0),
    ],
)
def test_safe_analysis_rows_reject_every_nonfrozen_refit_value(
    field, mutated_value
):
    payload = _safe_payload()
    payload["child_rows"][0]["refit_params"][field] = mutated_value

    with pytest.raises(RuntimeError, match="refit policy"):
        load_safe_rows(payload, task_split_counts={"toy": (7, 1)})
