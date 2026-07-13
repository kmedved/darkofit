import gzip
import hashlib
import json
import pickle
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

import benchmarks.run_tabarena_regression_cap_horizon as cap_module
from benchmarks.run_tabarena_regression_cap_horizon import (
    ANALYSIS_PAYLOAD_FILENAME,
    COMPLETION_ATTESTATION_FILENAME,
    COMMON_CONFIG,
    EXPECTED_CHILD_FITS,
    EXPECTED_DATASET_SPLITS,
    EXPECTED_JOBS,
    HORIZON_ARMS,
    HORIZONS,
    MANIFEST_FILENAME,
    PACKAGE_DISTRIBUTIONS,
    RUNTIME_ENVIRONMENT_KEYS,
    TASK_SPLIT_COUNTS,
    TIME_LIMIT_SECONDS,
    autogluon_compressed_refit_iterations,
    _decode_result_pickle,
    _exact_int,
    _validate_refit_params,
    _validate_resume_history,
    _repository_status,
    _sanitize_git_remote,
    expected_coordinates,
    expected_ag_ensemble_config,
    expected_child_hyperparameters,
    expected_fit_kwargs_extra,
    expected_resolved_method_hyperparameters,
    frozen_protocol,
    interleave_horizon_jobs,
    parse_args,
    prepare_paired_resume,
    protocol_sha256,
    resolve_and_pin_child_cpu_allocation,
    validate_chimera_coverage,
    validate_completed_result_artifacts,
    validate_stop_reason_causality,
    validate_output_state,
    write_completion_attestation,
    write_or_validate_run_manifest,
)


def _fake_job(dataset: str, repeat: int, fold: int, iterations: int):
    arm = "cap1000" if iterations == 1_000 else "cap10000"
    experiment = SimpleNamespace(
        name=f"DarkoFit_c1_{arm}_horizon_BAG_L1",
        method_kwargs={"model_hyperparameters": {"iterations": iterations}}
    )
    task = SimpleNamespace(dataset=dataset, repeat=repeat, fold=fold)
    return SimpleNamespace(experiment=experiment, task=task)


def _refit_params(iterations: int):
    return {
        "iterations": iterations,
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


def _warmup_history(thread_count=18):
    config = {
        "iterations": 5,
        "loss": "RMSE",
        "tree_mode": "catboost",
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
        "ordered_boosting": "auto",
        "sampling": "uniform",
        "linear_residual": False,
        "early_stopping": False,
        "use_best_model": False,
        "diagnostic_warnings": "never",
        "random_state": 20_260_713,
        "thread_count": thread_count,
    }
    stages = []
    for index, name in enumerate(("numeric", "categorical")):
        stages.append(
            {
                "name": name,
                "categorical_features": [] if index == 0 else [12],
                "train_rows": 2048,
                "validation_rows": 512,
                "fit_seconds": 0.01,
                "iterations_fitted": 5,
                "tree_depths": [6] * 5,
                "resolved_learning_rate": 0.1,
                "resolved_tree_mode": "catboost",
                "resolved_ordered_boosting": False,
                "resolved_thread_count": thread_count,
                "flat_ensemble_type": "FlatEnsemble",
                "flat_prediction_router_selected": True,
                "prediction_parallel_min_rows": 8192,
                "prediction_batches": [
                    {
                        "name": "serial_subthreshold",
                        "route": "flat_serial",
                        "input_shape": [8191, 12 + index],
                        "prediction_shape": [8191],
                        "predict_seconds": 0.01,
                        "prediction_sha256": "0" * 64,
                    },
                    {
                        "name": "parallel_at_threshold",
                        "route": "flat_parallel",
                        "input_shape": [8192, 12 + index],
                        "prediction_shape": [8192],
                        "predict_seconds": 0.01,
                        "prediction_sha256": "1" * 64,
                    },
                ],
            }
        )
    return [
        {
            "completed_at_utc": "2026-07-13T00:00:00+00:00",
            "pid": 1,
            "warmup": {
                "schema_version": 2,
                "clock": "time.monotonic_ns",
                "duration_seconds": 0.1,
                "config": config,
                "stages": stages,
            },
        }
    ]


def _cached_result_record(job):
    dataset, repeat, fold = job.task.dataset, job.task.repeat, job.task.fold
    task_id = TASK_SPLIT_COUNTS[dataset][0]
    horizon = job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
    arm = "cap1000" if horizon == 1_000 else "cap10000"
    attempted = 100
    best = 90
    fit_metadata = {
        "iterations_requested": horizon,
        "iterations_attempted": attempted,
        "rounds_completed": attempted,
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
    children = {}
    for index in range(1, 9):
        child_name = f"S1F{index}"
        children[child_name] = {
            "name": child_name,
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
            "darkofit_fit": dict(fit_metadata),
        }
    suffix = f"_c1_{arm}_horizon"
    return {
        "problem_type": "regression",
        "metric": "rmse",
        "metric_error": 1.0,
        "metric_error_val": 1.1,
        "time_train_s": 2.0,
        "time_infer_s": 0.1,
        "memory_usage": {"peak_mem_cpu": 1_000_000},
        "task_metadata": {
            "name": dataset,
            "tid": task_id,
            "repeat": repeat,
            "fold": fold,
            "split_idx": 3 * repeat + fold,
        },
        "framework": f"DarkoFit{suffix}_BAG_L1",
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
                "ag_args": {"name_suffix": suffix},
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


def _paired_validation_artifacts(tmp_path, mutate_cap1000=None):
    coordinate = expected_coordinates()[0]
    jobs = [
        _fake_job(coordinate[0], coordinate[1], coordinate[2], horizon)
        for horizon in HORIZONS
    ]
    artifacts = {}
    for job in jobs:
        horizon = job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
        arm = "cap1000" if horizon == 1_000 else "cap10000"
        record = _cached_result_record(job)
        if arm == "cap1000" and mutate_cap1000 is not None:
            mutate_cap1000(record)
        path = tmp_path / arm / "results.pkl"
        path.parent.mkdir(parents=True)
        payload = pickle.dumps(record)
        path.write_bytes(payload)
        artifacts[str(path.relative_to(tmp_path))] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return coordinate, jobs, artifacts


def _chimera_rows():
    rows = []
    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        rows.extend(
            {
                "dataset": dataset,
                "method": "CHIMERA (default)",
                "fold": fold,
                "imputed": False,
            }
            for fold in range(split_count)
        )
    return rows


def _manifest(tmp_path):
    return {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "output_dir": str(tmp_path.resolve()),
        "time_limit_seconds": 3600.0,
        "resolved_child_num_cpus": 18,
        "protocol_sha256": protocol_sha256(),
        "protocol": frozen_protocol(),
        "source": {"git_head": "abc", "git_tree": "def", "files": {}},
        "runtime": {"python_executable": "/python", "packages": {}},
    }


def test_cap_horizon_matrix_is_exact_and_single_lever():
    assert TASK_SPLIT_COUNTS == {
        "airfoil_self_noise": (363612, 30),
        "Another-Dataset-on-used-Fiat-500": (363615, 30),
        "concrete_compressive_strength": (363625, 30),
        "diamonds": (363631, 9),
        "Food_Delivery_Time": (363672, 9),
        "healthcare_insurance_expenses": (363675, 30),
        "houses": (363678, 9),
        "miami_housing": (363686, 9),
        "physiochemical_protein": (363693, 9),
        "QSAR-TID-11": (363697, 9),
        "QSAR_fish_toxicity": (363698, 30),
        "superconductivity": (363705, 9),
        "wine_quality": (363708, 9),
    }
    assert EXPECTED_DATASET_SPLITS == 222
    assert EXPECTED_JOBS == 444
    assert EXPECTED_CHILD_FITS == 3_552
    assert tuple(HORIZON_ARMS) == ("cap1000", "cap10000")
    assert HORIZONS == (1_000, 10_000)

    for arm, iterations in zip(HORIZON_ARMS.values(), HORIZONS):
        assert arm == {**COMMON_CONFIG, "iterations": iterations}
    differences = {
        key
        for key in HORIZON_ARMS["cap1000"]
        if HORIZON_ARMS["cap1000"][key] != HORIZON_ARMS["cap10000"][key]
    }
    assert differences == {"iterations"}
    assert COMMON_CONFIG == {
        "tree_mode": "catboost",
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
        "early_stopping": True,
        "use_best_model": True,
    }


def test_cap_horizon_coordinates_and_balanced_pair_order():
    coordinates = expected_coordinates()
    assert len(coordinates) == EXPECTED_DATASET_SPLITS
    assert len(set(coordinates)) == len(coordinates)
    assert coordinates[:2] == [
        ("airfoil_self_noise", 0, 0),
        ("airfoil_self_noise", 0, 1),
    ]
    assert coordinates[29] == ("airfoil_self_noise", 9, 2)
    assert coordinates[-1] == ("wine_quality", 2, 2)

    # Feed the scheduler a deliberately hostile order.  Its output must be the
    # canonical task/split order with adjacent, counterbalanced arm pairs.
    jobs = [
        _fake_job(dataset, repeat, fold, horizon)
        for dataset, repeat, fold in reversed(coordinates)
        for horizon in reversed(HORIZONS)
    ]
    ordered = interleave_horizon_jobs(jobs)
    assert len(ordered) == EXPECTED_JOBS
    for index, coordinate in enumerate(coordinates):
        pair = ordered[2 * index : 2 * index + 2]
        assert [_job.task.dataset for _job in pair] == [coordinate[0], coordinate[0]]
        assert [(_job.task.repeat, _job.task.fold) for _job in pair] == [
            coordinate[1:],
            coordinate[1:],
        ]
        expected_horizons = HORIZONS if index % 2 == 0 else HORIZONS[::-1]
        assert [
            _job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
            for _job in pair
        ] == list(expected_horizons)


def test_cap_horizon_scheduler_rejects_missing_and_duplicate_arms():
    jobs = [
        _fake_job(dataset, repeat, fold, horizon)
        for dataset, repeat, fold in expected_coordinates()
        for horizon in HORIZONS
    ]
    with pytest.raises(RuntimeError, match="has horizons"):
        interleave_horizon_jobs(jobs[:-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        interleave_horizon_jobs([*jobs, jobs[0]])


def test_cap_horizon_validates_chimera_coverage_without_reading_metrics():
    rows = _chimera_rows()
    rows.append(
        {
            "dataset": "unrelated",
            "method": "CHIMERA (default)",
            "fold": 0,
            "imputed": True,
            "metric_error": -999,
        }
    )
    validate_chimera_coverage(pd.DataFrame(rows))

    imputed = _chimera_rows()
    imputed[-1]["imputed"] = True
    with pytest.raises(RuntimeError, match="imputed"):
        validate_chimera_coverage(pd.DataFrame(imputed))

    duplicate = _chimera_rows()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_chimera_coverage(pd.DataFrame(duplicate))

    missing = _chimera_rows()[:-1]
    with pytest.raises(RuntimeError, match="unexpected registered CHIMERA folds"):
        validate_chimera_coverage(pd.DataFrame(missing))


def test_cap_horizon_protocol_is_json_stable():
    protocol = frozen_protocol()
    assert protocol["expected_dataset_splits"] == 222
    assert protocol["expected_jobs"] == 444
    assert protocol["time_limit_seconds"] == TIME_LIMIT_SECONDS == 3_600.0
    assert protocol["seed_configuration"] == {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
    }
    assert expected_ag_ensemble_config() == {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": 3_600.0,
    }
    assert protocol["chimera_role"].startswith("coverage validation only")
    assert protocol_sha256() == protocol_sha256()
    assert len(protocol_sha256()) == 64
    json.dumps(protocol, allow_nan=False)


def test_built_job_cpu_allocation_is_resolved_and_pinned(monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def _get_default_resources(self):
            return 6, 0

    experiments = []
    jobs = []
    for arm, config in HORIZON_ARMS.items():
        experiment = SimpleNamespace(
            method_kwargs={
                "model_cls": FakeModel,
                "model_hyperparameters": {
                    **config,
                    "ag_args": {"name_suffix": f"_c1_{arm}_horizon"},
                    "ag_args_ensemble": expected_ag_ensemble_config(),
                },
                "fit_kwargs": {},
            }
        )
        experiments.append(experiment)
        jobs.append(SimpleNamespace(experiment=experiment))
    monkeypatch.setattr(cap_module, "_autogluon_cpu_count", lambda: 12)

    assert resolve_and_pin_child_cpu_allocation(jobs) == 6
    assert all(
        experiment.method_kwargs["fit_kwargs"]["num_cpus"] == 6
        for experiment in experiments
    )


def test_warmup_history_requires_exact_child_cpu_allocation():
    cap_module._validate_warmup_history(
        _warmup_history(18), expected_thread_count=18
    )
    with pytest.raises(RuntimeError, match="resolved child resources"):
        cap_module._validate_warmup_history(
            _warmup_history(18), expected_thread_count=17
        )


def test_cap_horizon_runtime_provenance_covers_metric_and_numba_inputs():
    assert "psutil" in PACKAGE_DISTRIBUTIONS
    assert {
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
    } <= set(RUNTIME_ENVIRONMENT_KEYS)


def test_provenance_sanitizes_remote_credentials_and_only_ignores_run_output(
    tmp_path,
):
    assert _sanitize_git_remote(
        "https://oauth2:secret-token@example.com/org/repo.git?access_token=secret"
    ) == "https://example.com/org/repo.git"
    assert _sanitize_git_remote("secret@example.com:org/repo.git") == (
        "example.com:org/repo.git"
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    (repository / "tracked.py").write_text("clean\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repository, check=True)
    output_dir = repository / "custom-results"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text("generated\n")
    assert _repository_status(repository, output_dir) == ""

    (repository / "other.py").write_text("untracked\n")
    assert "untracked:other.py" in _repository_status(repository, output_dir)
    (repository / "tracked.py").write_text("dirty\n")
    assert "tracked:tracked.py" in _repository_status(repository, output_dir)


def test_atomic_json_write_does_not_follow_predictable_temp_symlink(tmp_path):
    target = tmp_path / "warmup_history.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite\n", encoding="utf-8")
    predictable = target.with_suffix(target.suffix + ".tmp")
    predictable.symlink_to(victim)

    cap_module._atomic_write_json(target, {"complete": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    assert victim.read_text(encoding="utf-8") == "do not overwrite\n"
    assert predictable.is_symlink()


def test_atomic_json_write_rejects_symlink_target(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_text('{"trusted": true}\n', encoding="utf-8")
    target = tmp_path / "run_manifest.json"
    target.symlink_to(victim)

    with pytest.raises(RuntimeError, match="must not be a symbolic link"):
        cap_module._atomic_write_json(target, {"trusted": False})

    assert json.loads(victim.read_text(encoding="utf-8")) == {"trusted": True}


def test_completion_inputs_reject_symlink_results_and_history(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments" / "job"
    experiments.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_bytes(b"payload")
    (experiments / "results.pkl").symlink_to(victim)
    monkeypatch.setattr(cap_module, "EXPECTED_JOBS", 1)

    with pytest.raises(RuntimeError, match="result artifact is not a regular file"):
        cap_module.collect_result_artifacts(tmp_path)

    history = tmp_path / "warmup_history.json"
    history.symlink_to(victim)
    with pytest.raises(RuntimeError, match="history is not a regular file"):
        cap_module._history_artifact(
            tmp_path,
            "warmup_history.json",
            required=True,
            validator=lambda value: None,
        )


def test_cap_horizon_output_cache_requires_explicit_matching_resume(tmp_path):
    output_dir = tmp_path / "campaign"
    manifest = _manifest(output_dir)

    validate_output_state(output_dir, resume=False)
    written = write_or_validate_run_manifest(output_dir, manifest, resume=False)
    assert written == manifest
    assert json.loads((output_dir / MANIFEST_FILENAME).read_text()) == manifest

    with pytest.raises(RuntimeError, match="not empty"):
        validate_output_state(output_dir, resume=False)
    assert write_or_validate_run_manifest(output_dir, manifest, resume=True) == manifest

    changed = {**manifest, "time_limit_seconds": 1800.0}
    with pytest.raises(RuntimeError, match="time_limit_seconds"):
        write_or_validate_run_manifest(output_dir, changed, resume=True)

    unmanifested = tmp_path / "unmanifested"
    unmanifested.mkdir()
    (unmanifested / "result.pkl").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="run_manifest.json is missing"):
        validate_output_state(unmanifested, resume=True)


def test_cap_horizon_completion_attestation_rejects_partial_runs(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "campaign"
    manifest = _manifest(output_dir)
    write_or_validate_run_manifest(output_dir, manifest, resume=False)

    with pytest.raises(RuntimeError, match="completed results"):
        write_completion_attestation(
            output_dir,
            manifest=manifest,
            result_count=EXPECTED_JOBS - 1,
        )

    for index in range(EXPECTED_JOBS):
        result_path = (
            output_dir / "experiments" / f"job-{index}" / "results.pkl"
        )
        result_path.parent.mkdir(parents=True)
        result_path.write_bytes(f"result-{index}".encode())
    monkeypatch.setattr(
        cap_module,
        "validate_completed_result_artifacts",
        lambda output_dir, artifacts: (
            {
                "result_count": EXPECTED_JOBS,
                "child_fit_count": EXPECTED_CHILD_FITS,
                "stop_reason_counts": {"iteration_limit": EXPECTED_CHILD_FITS},
                "resource_allocation": {
                    "num_cpus": 18,
                    "num_gpus": 0,
                    "num_cpus_child": 18,
                    "num_gpus_child": 0,
                },
            },
            [],
            [],
        ),
    )
    (output_dir / "warmup_history.json").write_text(
        json.dumps(_warmup_history()), encoding="utf-8"
    )

    mismatched_manifest = {**manifest, "resolved_child_num_cpus": 17}
    with pytest.raises(RuntimeError, match="does not match the run manifest"):
        write_completion_attestation(
            output_dir,
            manifest=mismatched_manifest,
            result_count=EXPECTED_JOBS,
        )

    attestation = write_completion_attestation(
        output_dir,
        manifest=manifest,
        result_count=EXPECTED_JOBS,
    )
    assert attestation["result_count"] == EXPECTED_JOBS
    assert attestation["manifest_sha256"]
    assert len(attestation["result_artifacts"]) == EXPECTED_JOBS
    assert attestation["analysis_payload_artifact"]["path"] == (
        ANALYSIS_PAYLOAD_FILENAME
    )
    assert attestation["warmup_history_artifact"]["path"] == (
        "warmup_history.json"
    )
    assert attestation["resume_history_artifact"] is None
    assert attestation["warmup_thread_count"] == 18
    assert (output_dir / ANALYSIS_PAYLOAD_FILENAME).is_file()
    assert attestation["validation"]["child_fit_count"] == EXPECTED_CHILD_FITS
    assert json.loads(
        (output_dir / COMPLETION_ATTESTATION_FILENAME).read_text()
    ) == attestation


def test_cap_horizon_resume_invalidates_incomplete_pairs_atomically(tmp_path):
    coordinates = expected_coordinates()
    jobs = [
        _fake_job(dataset, repeat, fold, horizon)
        for dataset, repeat, fold in coordinates
        for horizon in HORIZONS
    ]

    def result_path(job):
        dataset, repeat, fold = (
            job.task.dataset,
            job.task.repeat,
            job.task.fold,
        )
        task_id = TASK_SPLIT_COUNTS[dataset][0]
        return (
            tmp_path
            / "experiments"
            / "data"
            / job.experiment.name
            / str(task_id)
            / f"{repeat}_{fold}"
            / "results.pkl"
        )

    lone = result_path(jobs[0])
    paired = [result_path(jobs[2]), result_path(jobs[3])]
    corrupt_pair = [result_path(jobs[4]), result_path(jobs[5])]
    incomplete_pair = [result_path(jobs[6]), result_path(jobs[7])]
    for path in [lone, *paired, *corrupt_pair, *incomplete_pair]:
        path.parent.mkdir(parents=True, exist_ok=True)
    lone.write_bytes(pickle.dumps(_cached_result_record(jobs[0])))
    for path, job in zip(paired, jobs[2:4]):
        path.write_bytes(pickle.dumps(_cached_result_record(job)))
    corrupt_pair[0].write_bytes(pickle.dumps(_cached_result_record(jobs[4])))
    corrupt_pair[1].write_bytes(b"truncated pickle")
    incomplete_pair[0].write_bytes(pickle.dumps(_cached_result_record(jobs[6])))
    incomplete_pair[1].write_bytes(pickle.dumps({"problem_type": "regression"}))
    (tmp_path / COMPLETION_ATTESTATION_FILENAME).write_text("stale")
    (tmp_path / ANALYSIS_PAYLOAD_FILENAME).write_text("stale")
    for filename in cap_module.DEFAULT_ANALYSIS_OUTPUT_FILENAMES:
        (tmp_path / filename).write_text(f"stale {filename}")

    record = prepare_paired_resume(tmp_path, jobs, resume=True)

    assert record["invalidated_pair_count"] == 3
    assert record["invalidated_result_count"] == 5
    assert not lone.exists()
    assert all(path.exists() for path in paired)
    assert not any(path.exists() for path in corrupt_pair)
    assert not any(path.exists() for path in incomplete_pair)
    assert not (tmp_path / COMPLETION_ATTESTATION_FILENAME).exists()
    assert not (tmp_path / ANALYSIS_PAYLOAD_FILENAME).exists()
    assert all(
        not (tmp_path / filename).exists()
        for filename in cap_module.DEFAULT_ANALYSIS_OUTPUT_FILENAMES
    )
    invalidated = {
        (entry["coordinate"]["repeat"], entry["coordinate"]["fold"]): entry
        for entry in record["invalidated_pairs"]
    }
    assert invalidated[(0, 0)]["arm_status"] == {
        "cap1000": "valid",
        "cap10000": "missing",
    }
    assert invalidated[(0, 2)]["arm_status"] == {
        "cap1000": "valid",
        "cap10000": "unreadable",
    }
    assert invalidated[(1, 0)]["arm_status"] == {
        "cap1000": "valid",
        "cap10000": "incomplete_or_mismatched",
    }
    archived_results = [
        result
        for pair in record["invalidated_pairs"]
        for result in pair["archived_results"]
    ]
    assert len(archived_results) == 5
    assert all((tmp_path / result["archive"]).is_file() for result in archived_results)
    assert (tmp_path / record["archived_completion_attestation"]).is_file()
    assert (tmp_path / record["archived_analysis_payload"]).is_file()
    assert {
        item["source"] for item in record["archived_analysis_outputs"]
    } == set(cap_module.DEFAULT_ANALYSIS_OUTPUT_FILENAMES)
    assert all(
        (tmp_path / item["archive"]).is_file()
        for item in record["archived_analysis_outputs"]
    )
    _validate_resume_history([record], tmp_path)


@pytest.mark.parametrize(
    "invalid_target",
    [
        "result_directory",
        "result_symlink",
        "analysis_directory",
        "analysis_symlink",
    ],
)
def test_resume_rejects_nonregular_archive_sources_before_mutation(
    tmp_path, invalid_target
):
    dataset, (task_id, _) = next(iter(TASK_SPLIT_COUNTS.items()))
    jobs = [_fake_job(dataset, 0, 0, horizon) for horizon in HORIZONS]
    result_path = (
        tmp_path
        / "experiments"
        / "data"
        / jobs[0].experiment.name
        / str(task_id)
        / "0_0"
        / "results.pkl"
    )
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched\n", encoding="utf-8")
    if invalid_target == "result_directory":
        result_path.mkdir(parents=True)
    elif invalid_target == "result_symlink":
        result_path.parent.mkdir(parents=True)
        result_path.symlink_to(victim)
    elif invalid_target == "analysis_directory":
        (tmp_path / "summary.json").mkdir()
    else:
        (tmp_path / "summary.json").symlink_to(victim)

    with pytest.raises(RuntimeError, match="is not a regular file"):
        prepare_paired_resume(tmp_path, jobs, resume=True)

    assert not (tmp_path / "resume_invalidated").exists()
    assert not (tmp_path / "resume_history.json").exists()
    if invalid_target == "result_directory":
        assert result_path.is_dir()
    elif invalid_target == "result_symlink":
        assert result_path.is_symlink()
    elif invalid_target == "analysis_directory":
        assert not result_path.exists()
        assert (tmp_path / "summary.json").is_dir()
    else:
        assert not result_path.exists()
        assert (tmp_path / "summary.json").is_symlink()
    assert victim.read_text(encoding="utf-8") == "untouched\n"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("depth", 7),
        ("num_leaves", 64),
        ("l2_leaf_reg", 4.0),
        ("min_child_samples", 21),
        ("min_child_weight", 2.0),
        ("cat_smoothing", 2.0),
    ],
)
def test_cap_horizon_completion_rejects_wrong_refit_policy_values(
    tmp_path, monkeypatch, field, bad_value
):
    coordinate = expected_coordinates()[0]
    jobs = [
        _fake_job(coordinate[0], coordinate[1], coordinate[2], horizon)
        for horizon in HORIZONS
    ]
    artifacts = {}
    for job in jobs:
        horizon = job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
        arm = "cap1000" if horizon == 1_000 else "cap10000"
        record = _cached_result_record(job)
        if arm == "cap1000":
            record["method_metadata"]["info"]["children_info"]["S1F1"][
                "hyperparameters_fit"
            ][field] = bad_value
        path = tmp_path / arm / "results.pkl"
        path.parent.mkdir(parents=True)
        payload = pickle.dumps(record)
        path.write_bytes(payload)
        artifacts[str(path.relative_to(tmp_path))] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    monkeypatch.setattr(cap_module, "expected_coordinates", lambda: [coordinate])
    monkeypatch.setattr(cap_module, "EXPECTED_CHILD_FITS", 16)
    with pytest.raises(RuntimeError, match=field):
        validate_completed_result_artifacts(tmp_path, artifacts)


def test_cap_horizon_refit_validator_accepts_only_complete_frozen_values():
    params = _refit_params(90)
    assert _validate_refit_params(
        params,
        "refit",
        expected_iterations=90,
        max_iterations=1_000,
    ) == params

    for field, bad_value in {
        "iterations": 89,
        "learning_rate": 0.2,
        "tree_mode": "auto",
        "early_stopping": True,
        "early_stopping_rounds": 50,
        "use_best_model": True,
        "refit": True,
    }.items():
        bad = dict(params)
        bad[field] = bad_value
        with pytest.raises(RuntimeError, match=field):
            _validate_refit_params(
                bad,
                "refit",
                expected_iterations=90,
                max_iterations=1_000,
            )


def test_cap_horizon_rejects_child_deadline_above_frozen_outer_budget(
    tmp_path, monkeypatch
):
    coordinate = expected_coordinates()[0]
    jobs = [
        _fake_job(coordinate[0], coordinate[1], coordinate[2], horizon)
        for horizon in HORIZONS
    ]
    artifacts = {}
    for job in jobs:
        horizon = job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
        arm = "cap1000" if horizon == 1_000 else "cap10000"
        record = _cached_result_record(job)
        if arm == "cap1000":
            fitted = record["method_metadata"]["info"]["children_info"]["S1F1"][
                "darkofit_fit"
            ]
            fitted["wall_clock_limit_seconds"] = 7_200.0
            fitted["wall_clock_safety_margin_seconds"] = 5.0
            fitted["wall_clock_effective_seconds"] = 7_195.0
        path = tmp_path / arm / "results.pkl"
        path.parent.mkdir(parents=True)
        payload = pickle.dumps(record)
        path.write_bytes(payload)
        artifacts[str(path.relative_to(tmp_path))] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        if arm == "cap1000":
            assert cap_module._cached_result_issue(path, job) == (
                "incomplete_or_mismatched"
            )

    monkeypatch.setattr(cap_module, "expected_coordinates", lambda: [coordinate])
    monkeypatch.setattr(cap_module, "EXPECTED_CHILD_FITS", 16)
    with pytest.raises(RuntimeError, match="deadline metadata"):
        validate_completed_result_artifacts(tmp_path, artifacts)


def test_cap_horizon_cli_enforces_frozen_time_limit():
    assert parse_args(["--time-limit", "3600", "--dry-run"]).time_limit == 3600
    for invalid in ("12.5", "0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parse_args(["--time-limit", invalid])


@pytest.mark.parametrize("compressed", [False, True])
def test_cap_horizon_decodes_both_tabarena_cache_formats(compressed):
    record = {"format": "gzip" if compressed else "raw"}
    payload = pickle.dumps(record)
    if compressed:
        payload = gzip.compress(payload)
    assert _decode_result_pickle(payload, "result.pkl") == record


@pytest.mark.parametrize("value", [1.5, True, "1"])
def test_cap_horizon_completion_counter_validation_never_truncates(value):
    with pytest.raises(RuntimeError, match="must be an integer"):
        _exact_int(value, "counter")


def test_autogluon_compressed_iterations_use_all_children_and_bankers_rounding():
    # These are the eight child best-prefix values from the trusted live QSAR
    # schema preflight. AutoGluon's round(mean(int_values)) yields 106.
    assert autogluon_compressed_refit_iterations(
        [37, 239, 97, 53, 123, 44, 96, 156], field="children"
    ) == 106
    assert autogluon_compressed_refit_iterations(
        [0, 0, 0, 0, 1, 1, 1, 1], field="children"
    ) == 0


@pytest.mark.parametrize(
    ("reason", "requested", "attempted", "completed"),
    [
        ("iteration_limit", 1_000, 999, 999),
        ("time_limit", 1_000, 1_000, 1_000),
        ("no_split", 1_000, 100, 100),
        ("early_stopping", 1_000, 0, 0),
    ],
)
def test_stop_reason_causality_rejects_impossible_counters(
    reason, requested, attempted, completed
):
    with pytest.raises(RuntimeError, match=reason):
        validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field="child",
        )


def test_resume_cache_rejects_impossible_iteration_limit(tmp_path):
    job = _fake_job(*expected_coordinates()[0], 1_000)
    record = _cached_result_record(job)
    record["method_metadata"]["info"]["children_info"]["S1F1"][
        "darkofit_fit"
    ]["stop_reason"] = "iteration_limit"
    path = tmp_path / "results.pkl"
    path.write_bytes(pickle.dumps(record))

    assert cap_module._cached_result_issue(path, job) == "incomplete_or_mismatched"


@pytest.mark.parametrize(
    "field",
    [
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
    ],
)
def test_resume_cache_rejects_boolean_deadline_numbers(tmp_path, field):
    job = _fake_job(*expected_coordinates()[0], 1_000)
    record = _cached_result_record(job)
    record["method_metadata"]["info"]["children_info"]["S1F1"][
        "darkofit_fit"
    ][field] = True
    path = tmp_path / "results.pkl"
    path.write_bytes(pickle.dumps(record))

    assert cap_module._cached_result_issue(path, job) == "incomplete_or_mismatched"


def test_resume_cache_rejects_stale_compressed_refit_iterations(tmp_path):
    job = _fake_job(*expected_coordinates()[0], 1_000)
    record = _cached_result_record(job)
    record["method_metadata"]["info"]["bagged_info"][
        "child_hyperparameters_fit"
    ]["iterations"] += 1
    path = tmp_path / "results.pkl"
    path.write_bytes(pickle.dumps(record))

    assert cap_module._cached_result_issue(path, job) == "incomplete_or_mismatched"


@pytest.mark.parametrize(
    "mutation",
    [
        "resolved_budget",
        "bag_fold_count",
        "method_resources",
        "bag_repeat_count",
        "bag_budget_ratio",
        "child_max_bins",
        "child_fold_seed",
        "child_resources",
        "child_validation_policy",
    ],
)
def test_resume_cache_rejects_mismatched_resolved_execution_semantics(
    tmp_path, mutation
):
    job = _fake_job(*expected_coordinates()[0], 1_000)
    record = _cached_result_record(job)
    method = record["method_metadata"]
    bag = method["info"]["bagged_info"]
    child = method["info"]["children_info"]["S1F1"]
    if mutation == "resolved_budget":
        method["hyperparameters"]["ag_args_ensemble"]["ag_args_fit"][
            "max_time_limit"
        ] = 600.0
    elif mutation == "bag_fold_count":
        method["fit_kwargs_extra"]["num_bag_folds"] = 4
    elif mutation == "method_resources":
        method["num_cpus_child"] = 1
    elif mutation == "bag_repeat_count":
        bag["_n_repeats"] = 2
    elif mutation == "bag_budget_ratio":
        bag["child_ag_args_fit"]["max_time_limit_ratio"] = 0.5
    elif mutation == "child_max_bins":
        child["hyperparameters"]["max_bins"] = 254
    elif mutation == "child_fold_seed":
        child["hyperparameters"]["random_state"] = 7
    elif mutation == "child_resources":
        child["num_cpus"] = 1
    elif mutation == "child_validation_policy":
        child["val_in_fit"] = False
    path = tmp_path / "results.pkl"
    path.write_bytes(pickle.dumps(record))

    assert cap_module._cached_result_issue(path, job) == "incomplete_or_mismatched"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("iteration_limit", "iteration_limit"),
        ("compressed_iterations", "aggregation"),
        ("bool_wall_clock_limit_seconds", "must be numeric"),
        ("bool_wall_clock_safety_margin_seconds", "must be numeric"),
        ("bool_wall_clock_effective_seconds", "must be numeric"),
        ("bool_wall_clock_elapsed_seconds", "must be numeric"),
    ],
)
def test_completion_normalization_rejects_impossible_child_metadata(
    tmp_path, monkeypatch, mutation, expected_error
):
    def mutate(record):
        info = record["method_metadata"]["info"]
        if mutation == "iteration_limit":
            info["children_info"]["S1F1"]["darkofit_fit"][
                "stop_reason"
            ] = "iteration_limit"
        elif mutation == "compressed_iterations":
            info["bagged_info"]["child_hyperparameters_fit"]["iterations"] += 1
        else:
            field = mutation.removeprefix("bool_")
            info["children_info"]["S1F1"]["darkofit_fit"][field] = True

    coordinate, _, artifacts = _paired_validation_artifacts(
        tmp_path, mutate_cap1000=mutate
    )
    monkeypatch.setattr(cap_module, "expected_coordinates", lambda: [coordinate])
    monkeypatch.setattr(cap_module, "EXPECTED_CHILD_FITS", 16)

    with pytest.raises(RuntimeError, match=expected_error):
        validate_completed_result_artifacts(tmp_path, artifacts)
