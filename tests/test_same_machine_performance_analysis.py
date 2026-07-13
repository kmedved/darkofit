import copy
import gzip
import json
import pickle

import pytest

from benchmarks.analyze_tabarena_same_machine_performance import (
    CONFIG_SPECS,
    EXPECTED_ROWS,
    SPLIT_COORDINATES,
    analyze_performance_rows,
    load_performance_rows,
    load_provenance,
    main,
    performance_result_row,
    validate_performance_rows,
    validate_provenance,
)
from benchmarks.run_tabarena_regression_remaining9 import (
    FROZEN_CANDIDATE,
    TASK_SPLIT_COUNTS,
)
from benchmarks.run_tabarena_same_machine_performance import (
    CACHE_POLICY,
    CHIMERA_REGRESSOR_PRODUCT_DEFAULTS,
    FROZEN_CHIMERA_COMMIT,
    FROZEN_CHIMERA_VERSION,
    SPLIT_INDICES,
    TIME_LIMIT_SECONDS,
    WARMUP_CASES,
)


def _provenance():
    return {
        "repository": "/source/chimeraboost",
        "commit": FROZEN_CHIMERA_COMMIT,
        "version_expected": FROZEN_CHIMERA_VERSION,
        "dirty": False,
        "package": "chimeraboost",
        "version_imported": FROZEN_CHIMERA_VERSION,
        "module_file": "/source/chimeraboost/chimeraboost/__init__.py",
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "split_indices": list(SPLIT_INDICES),
        "candidate": dict(FROZEN_CANDIDATE),
        "cache_policy": CACHE_POLICY,
        "chimera_regressor_product_defaults": dict(
            CHIMERA_REGRESSOR_PRODUCT_DEFAULTS
        ),
        "darkofit_warmup_seconds": 0.75,
        "chimeraboost_warmup_seconds": 1.25,
        "warmup_threads": 16,
        "warmup_cases": list(WARMUP_CASES),
        "darkofit_repository": "/source/darkofit",
        "darkofit_commit": "a" * 40,
        "darkofit_dirty": False,
        "darkofit_package": "darkofit",
        "darkofit_version_imported": "0.9.0",
        "darkofit_module_file": "/source/darkofit/darkofit/__init__.py",
        "runtime": {
            "python_version": "3.12.10",
            "python_implementation": "CPython",
            "python_executable": "/venv/bin/python",
            "platform": "macOS-test",
            "machine": "arm64",
            "processor": "arm",
            "logical_cpu_count": 16,
        },
    }


def _payload(config, dataset=None, repeat=0, fold=0, scale=1.0):
    dataset = dataset or next(iter(TASK_SPLIT_COUNTS))
    task_id = TASK_SPLIT_COUNTS[dataset][0]
    spec = CONFIG_SPECS[config]
    registered_fold = 3 * repeat + fold
    children = {}
    for child_index in range(8):
        child = {
            "preprocessing_fit_transform_seconds": 0.01 * scale,
            "preprocessing_fit_transform_calls": 1,
        }
        if config == "chimeraboost_default":
            child.update(
                {
                    "benchmark_package": "chimeraboost",
                    "benchmark_package_version": FROZEN_CHIMERA_VERSION,
                    "benchmark_source_commit": FROZEN_CHIMERA_COMMIT,
                    "benchmark_regressor_product_parameters": dict(
                        CHIMERA_REGRESSOR_PRODUCT_DEFAULTS
                    ),
                }
            )
        children[f"S1F{child_index + 1}"] = child
    model_hyperparameters = dict(spec["parameters"])
    model_hyperparameters.update(
        {
            "ag_args": {"name_suffix": spec["name_suffix"]},
            "ag_args_ensemble": {
                "model_random_seed": 0,
                "vary_seed_across_folds": True,
                "fold_fitting_strategy": "sequential_local",
                "ag.max_time_limit": TIME_LIMIT_SECONDS,
            },
        }
    )
    return {
        "framework": spec["framework"],
        "problem_type": "regression",
        "metric": "rmse",
        "metric_error": 10.0 * scale,
        "metric_error_val": 11.0 * scale,
        "time_train_s": 2.0 * scale,
        "time_infer_s": 0.2 * scale,
        "memory_usage": {"peak_mem_cpu": 1_000_000 * scale},
        "task_metadata": {
            "name": dataset,
            "tid": task_id,
            "repeat": repeat,
            "fold": fold,
            "split_idx": registered_fold,
            "sample": 0,
        },
        "method_metadata": {
            "model_cls": spec["model_cls"],
            "model_type": spec["model_type"],
            "name_prefix": spec["name_prefix"],
            "model_hyperparameters": model_hyperparameters,
            "info": {
                "is_valid": True,
                "can_infer": True,
                "bagged_info": {
                    "num_child_models": 8,
                    "max_memory_size": int(10_000 * scale),
                    "min_memory_size": int(2_000 * scale),
                },
                "children_info": children,
            },
        },
    }


def _normalized_rows():
    scales = {
        "darkofit_default": 1.0,
        "darkofit_candidate": 0.9,
        "chimeraboost_default": 1.2,
    }
    rows = []
    for dataset in TASK_SPLIT_COUNTS:
        for repeat, fold, _ in SPLIT_COORDINATES:
            for config in CONFIG_SPECS:
                rows.append(
                    performance_result_row(
                        _payload(
                            config,
                            dataset=dataset,
                            repeat=repeat,
                            fold=fold,
                            scale=scales[config],
                        ),
                        source=f"{config}/{dataset}/{repeat}_{fold}/results.pkl",
                    )
                )
    return rows


def test_same_machine_result_parser_sums_children_and_sizes():
    row = performance_result_row(
        _payload("chimeraboost_default", scale=1.2), source="result.pkl"
    )

    assert row["config"] == "chimeraboost_default"
    assert row["preprocessing_time_s"] == pytest.approx(0.096)
    assert row["preprocessing_fit_transform_calls"] == 8
    assert row["model_size_all_children_bytes"] == 12_000
    assert row["model_size_low_memory_bytes"] == 2_400
    assert row["child_model_count"] == 8


def test_same_machine_result_parser_rejects_noncanonical_configuration():
    payload = _payload("darkofit_candidate")
    payload["method_metadata"]["model_hyperparameters"]["learning_rate"] = 0.2

    with pytest.raises(RuntimeError, match="product parameters"):
        performance_result_row(payload, source="result.pkl")


def test_same_machine_result_parser_rejects_hidden_chimera_default_override():
    payload = _payload("chimeraboost_default")
    child = payload["method_metadata"]["info"]["children_info"]["S1F1"]
    child["benchmark_regressor_product_parameters"]["n_estimators"] = 10_000

    with pytest.raises(RuntimeError, match="child provenance"):
        performance_result_row(payload, source="result.pkl")


def test_same_machine_analyzer_is_complete_paired_and_deterministic():
    rows = _normalized_rows()

    summary = analyze_performance_rows(rows, _provenance())
    reversed_summary = analyze_performance_rows(list(reversed(rows)), _provenance())

    assert len(rows) == EXPECTED_ROWS == 81
    assert summary == reversed_summary
    assert summary["counts"] == {
        "datasets": 9,
        "coordinates": 27,
        "configs": 3,
        "rows": 81,
        "bagged_children": 648,
    }
    candidate = summary["pairwise"]["candidate_vs_darkofit_default"]
    for metric in (
        "rmse",
        "train_time_s",
        "preprocessing_time_s",
        "infer_time_s",
        "peak_rss_bytes",
        "model_size_all_children_bytes",
        "model_size_low_memory_bytes",
    ):
        assert candidate["equal_dataset"][metric]["ratio"] == pytest.approx(0.9)
        assert candidate["coordinate_summary"][metric]["numerator_better"] == 27
    assert summary["provenance"]["commit"] == FROZEN_CHIMERA_COMMIT


def test_same_machine_analyzer_rejects_incomplete_or_duplicate_panel():
    rows = _normalized_rows()
    with pytest.raises(RuntimeError, match="expected exactly 81"):
        validate_performance_rows(rows[:-1])

    duplicate = copy.deepcopy(rows)
    duplicate[-1] = duplicate[0]
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_performance_rows(duplicate)


def test_same_machine_analyzer_rejects_noncanonical_provenance():
    provenance = _provenance()
    provenance["commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="noncanonical provenance commit"):
        validate_provenance(provenance)

    provenance = _provenance()
    provenance["darkofit_module_file"] = "/site-packages/darkofit/__init__.py"
    with pytest.raises(RuntimeError, match="DarkoFit module is outside"):
        validate_provenance(provenance)


def test_same_machine_analyzer_reads_gzip_and_writes_tidy_artifacts(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "provenance.json").write_text(json.dumps(_provenance()))
    scales = {
        "darkofit_default": 1.0,
        "darkofit_candidate": 0.9,
        "chimeraboost_default": 1.2,
    }
    for dataset in TASK_SPLIT_COUNTS:
        for repeat, fold, _ in SPLIT_COORDINATES:
            for config in CONFIG_SPECS:
                path = input_dir / config / dataset / f"{repeat}_{fold}" / "results.pkl"
                path.parent.mkdir(parents=True)
                with gzip.open(path, "wb") as stream:
                    pickle.dump(
                        _payload(
                            config,
                            dataset=dataset,
                            repeat=repeat,
                            fold=fold,
                            scale=scales[config],
                        ),
                        stream,
                    )

    assert len(load_performance_rows(input_dir)) == 81
    assert load_provenance(input_dir)["commit"] == FROZEN_CHIMERA_COMMIT
    csv_path = tmp_path / "tidy.csv"
    json_path = tmp_path / "summary.json"
    assert main(
        [
            "--input-dir",
            str(input_dir),
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
        ]
    ) == 0
    assert len(csv_path.read_text().splitlines()) == 82
    summary = json.loads(json_path.read_text())
    assert summary["counts"]["rows"] == 81
    assert summary["pairwise"]["candidate_vs_darkofit_default"][
        "equal_dataset"
    ]["train_time_s"]["ratio"] == pytest.approx(0.9)
