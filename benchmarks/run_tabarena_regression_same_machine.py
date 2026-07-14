"""Run the frozen same-machine DarkoFit/ChimeraBoost/CatBoost campaign.

The primary lane compares the three current product defaults over the same
three registered coordinates of the 13-dataset regression panel.  A separate
Airfoil/Diamonds diagnostic applies the identical source-declared safe ordinal
transform to all three engines.  The two lanes are never pooled.

This runner owns every artifact beneath its output directory.  ``--resume``
may therefore be used only with a trusted cache produced by this exact source;
validating cached TabArena records necessarily unpickles those runner-owned
records.  The standalone analyzer consumes only the normalized JSON payload
and opaque raw-result hashes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
import platform
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# A dry-run must not leave import bytecode behind, including bytecode emitted
# while importing the local benchmark helpers below.  ``main`` restores the
# caller's setting after the complete read-only audit.
_DONT_WRITE_BYTECODE_BEFORE_IMPORT = sys.dont_write_bytecode
_DRY_RUN_REQUESTED_AT_IMPORT = (
    __name__ == "__main__" and "--dry-run" in sys.argv[1:]
)
if _DRY_RUN_REQUESTED_AT_IMPORT:
    sys.dont_write_bytecode = True

try:
    from benchmarks import run_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_tabarena_regression_followon_screen as followon
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import run_tabarena_regression_cap_horizon as hardened
    import run_tabarena_regression_followon_screen as followon


# The source mapping and insertion order are part of the frozen protocol.
TASK_SPLIT_COUNTS = dict(hardened.TASK_SPLIT_COUNTS)
TASKS = {dataset: task_id for dataset, (task_id, _) in TASK_SPLIT_COUNTS.items()}
PRIMARY_COORDINATE_PAIRS = ((0, 0), (1, 1), (2, 2))
DIAGNOSTIC_DATASETS = ("airfoil_self_noise", "diamonds")
PRIMARY_LANE = "primary"
ORDINAL_DIAGNOSTIC_LANE = "ordinal_diagnostic"
LANES = (PRIMARY_LANE, ORDINAL_DIAGNOSTIC_LANE)
SPLIT_INDICES = [f"r{repeat}f{fold}" for repeat, fold in PRIMARY_COORDINATE_PAIRS]

ENGINE_SPECS: dict[str, dict[str, Any]] = {
    "darkofit": {
        "code": "D",
        "display_name": "DarkoFit",
        "model_type": "DARKO",
        "version": "0.9.0",
        "native_class": "ComparatorDarkoFitModel",
        "ordinal_class": "ComparatorOrdinalDarkoFitModel",
    },
    "chimeraboost": {
        "code": "M",
        "display_name": "ChimeraBoost",
        "model_type": "CHIMERA",
        "version": "0.14.1",
        "native_class": "ComparatorChimeraBoostModel",
        "ordinal_class": "ComparatorOrdinalChimeraBoostModel",
    },
    "catboost": {
        "code": "C",
        "display_name": "CatBoost",
        "model_type": "CAT",
        "version": "1.2.10",
        "native_class": "ComparatorCatBoostModel",
        "ordinal_class": "ComparatorOrdinalCatBoostModel",
    },
}
ENGINE_CODES = {spec["code"]: engine for engine, spec in ENGINE_SPECS.items()}
ORDER_CYCLE = (
    ("D", "M", "C"),
    ("M", "C", "D"),
    ("C", "D", "M"),
    ("C", "M", "D"),
    ("D", "C", "M"),
    ("M", "D", "C"),
)

ARM_SPECS: dict[str, dict[str, Any]] = {}
for _lane in LANES:
    for _engine, _engine_spec in ENGINE_SPECS.items():
        _ordinal = _lane == ORDINAL_DIAGNOSTIC_LANE
        if _ordinal:
            _arm = {
                "darkofit": "darkofit_product_default_safe_ordinal",
                "chimeraboost": "chimeraboost_0_14_1_default_safe_ordinal",
                "catboost": "catboost_1_2_10_default_safe_ordinal",
            }[_engine]
        else:
            _arm = {
                "darkofit": "darkofit_product_default",
                "chimeraboost": "chimeraboost_0_14_1_default",
                "catboost": "catboost_1_2_10_default",
            }[_engine]
        ARM_SPECS[_arm] = {
            "lane": _lane,
            "engine": _engine,
            "code": _engine_spec["code"],
            "config": {},
            "model_cls": _engine_spec[
                "ordinal_class" if _ordinal else "native_class"
            ],
            "representation": "safe_ordinal" if _ordinal else "native",
        }
del _lane, _engine, _engine_spec, _ordinal, _arm

ARM_BY_LANE_CODE = {
    (spec["lane"], spec["code"]): arm for arm, spec in ARM_SPECS.items()
}

EXPECTED_PRIMARY_COORDINATES = len(TASKS) * len(PRIMARY_COORDINATE_PAIRS)
EXPECTED_DIAGNOSTIC_COORDINATES = len(DIAGNOSTIC_DATASETS) * len(
    PRIMARY_COORDINATE_PAIRS
)
EXPECTED_COORDINATES = EXPECTED_PRIMARY_COORDINATES + EXPECTED_DIAGNOSTIC_COORDINATES
EXPECTED_PRIMARY_JOBS = EXPECTED_PRIMARY_COORDINATES * len(ENGINE_SPECS)
EXPECTED_DIAGNOSTIC_JOBS = EXPECTED_DIAGNOSTIC_COORDINATES * len(ENGINE_SPECS)
EXPECTED_JOBS = EXPECTED_PRIMARY_JOBS + EXPECTED_DIAGNOSTIC_JOBS
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
EXPECTED_CHILD_CPUS = 18
TIME_LIMIT_SECONDS = 3_600.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_719
WARMUP_STAGE_NAMES = (
    "darkofit_numeric",
    "darkofit_categorical",
    "chimeraboost_numeric",
    "chimeraboost_categorical",
    "catboost_numeric",
    "catboost_categorical",
)
WARMUP_KIND = CAMPAIGN_WARMUP_KIND = (
    "darkofit_tabarena_regression_same_machine_warmup"
)
WARMUP_SCHEMA_VERSION = 1

CHIMERABOOST_TAG_COMMIT = "9c9ea6e704a9fe2bfe6d6c284b22de73914be048"
CHIMERABOOST_VERSION = "0.14.1"
CATBOOST_VERSION = "1.2.10"
CATBOOST_PROVENANCE_MODULES = (
    "catboost",
    "catboost.core",
    "catboost._catboost",
    "catboost.version",
    "catboost.metrics",
    "catboost.plot_helpers",
)
AUTOGLUON_PROVENANCE_MODULES = {
    "autogluon.common": (
        "autogluon.common.features.types",
        "autogluon.common.utils.pandas_utils",
        "autogluon.common.utils.resource_utils",
        "autogluon.common.utils.try_import",
    ),
    "autogluon.core": (
        "autogluon.core.constants",
        "autogluon.core.metrics",
        "autogluon.core.models",
        "autogluon.core.models._utils",
        "autogluon.core.utils.exceptions",
    ),
    "autogluon.features": (
        "autogluon.features",
        "autogluon.features.generators",
        "autogluon.features.generators.abstract",
        "autogluon.features.generators.auto_ml_pipeline",
        "autogluon.features.generators.category",
        "autogluon.features.generators.drop_unique",
        "autogluon.features.generators.pipeline",
    ),
    "autogluon.tabular": (
        "autogluon.tabular.models.catboost",
        "autogluon.tabular.models.catboost.callbacks",
        "autogluon.tabular.models.catboost.catboost_model",
        "autogluon.tabular.models.catboost.catboost_utils",
        "autogluon.tabular.models.catboost.hyperparameters",
        "autogluon.tabular.models.catboost.hyperparameters.parameters",
        "autogluon.tabular.models.catboost.hyperparameters.searchspaces",
    ),
}
DEFAULT_CHIMERABOOST_PATH = Path("/Users/kmedved/.cache/chimeraboost-v0.14.1")
DEFAULT_OUTPUT_DIR = Path(
    ".cache/tabarena-regression-same-machine-0.9.0-20260713"
)

MANIFEST_FILENAME = hardened.MANIFEST_FILENAME
COMPLETION_ATTESTATION_FILENAME = hardened.COMPLETION_ATTESTATION_FILENAME
ANALYSIS_PAYLOAD_FILENAME = hardened.ANALYSIS_PAYLOAD_FILENAME
WARMUP_HISTORY_FILENAME = hardened.WARMUP_HISTORY_FILENAME
RESUME_HISTORY_FILENAME = hardened.RESUME_HISTORY_FILENAME
DEFAULT_ANALYSIS_OUTPUT_FILENAMES = (
    "primary_paired_splits.csv",
    "primary_per_dataset.csv",
    "ordinal_diagnostic_paired_splits.csv",
    "ordinal_diagnostic_per_dataset.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
CAMPAIGN_KIND = "darkofit_tabarena_regression_same_machine"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"

SOURCE_FILES = (
    Path("pyproject.toml"),
    Path("darkofit/__init__.py"),
    Path("darkofit/booster.py"),
    Path("darkofit/callbacks.py"),
    Path("darkofit/preprocessing.py"),
    Path("darkofit/sklearn_api.py"),
    Path("benchmarks/tabarena_adapter.py"),
    Path("benchmarks/tabarena_screen_adapters.py"),
    Path("benchmarks/tabarena_comparator_adapters.py"),
    Path("benchmarks/tabarena_comparator_warmup.py"),
    Path("benchmarks/run_tabarena_regression_cap_horizon.py"),
    Path("benchmarks/run_tabarena_regression_followon_screen.py"),
    Path("benchmarks/run_tabarena_regression_ordinal_confirmation.py"),
    Path("benchmarks/run_tabarena_regression_same_machine.py"),
    Path("benchmarks/analyze_tabarena_regression_same_machine.py"),
    Path("benchmarks/tabarena_regression_same_machine_protocol.md"),
)
PACKAGE_DISTRIBUTIONS = tuple(
    dict.fromkeys(
        (
            *hardened.PACKAGE_DISTRIBUTIONS,
            "autogluon.features",
            "catboost",
            "graphviz",
        )
    )
)
RUNTIME_ENVIRONMENT_KEYS = tuple(
    dict.fromkeys((*hardened.RUNTIME_ENVIRONMENT_KEYS, "CHIMERABOOST_WARMUP"))
)


def expected_coordinates(lane: str | None = None) -> list[tuple[str, str, int, int]]:
    """Return lane-tagged coordinates in their frozen order."""
    if lane is not None and lane not in LANES:
        raise RuntimeError(f"unexpected comparator lane: {lane}")
    rows: list[tuple[str, str, int, int]] = []
    if lane in (None, PRIMARY_LANE):
        rows.extend(
            (PRIMARY_LANE, dataset, repeat, fold)
            for dataset in TASKS
            for repeat, fold in PRIMARY_COORDINATE_PAIRS
        )
    if lane in (None, ORDINAL_DIAGNOSTIC_LANE):
        rows.extend(
            (ORDINAL_DIAGNOSTIC_LANE, dataset, repeat, fold)
            for dataset in DIAGNOSTIC_DATASETS
            for repeat, fold in PRIMARY_COORDINATE_PAIRS
        )
    if lane is None and len(rows) != EXPECTED_COORDINATES:
        raise RuntimeError("frozen comparator coordinate count changed")
    return rows


def expected_arm_coordinates(arm: str) -> list[tuple[str, str, int, int]]:
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected comparator arm: {arm}")
    return expected_coordinates(ARM_SPECS[arm]["lane"])


def expected_grid() -> set[tuple[str, str, int, int, str]]:
    return {
        (*coordinate, arm)
        for coordinate in expected_coordinates()
        for arm, spec in ARM_SPECS.items()
        if spec["lane"] == coordinate[0]
    }


def expected_ordered_grid() -> list[tuple[str, str, int, int, str]]:
    """Return primary continuous-cycle jobs followed by the diagnostic cycle."""
    ordered: list[tuple[str, str, int, int, str]] = []
    for lane in LANES:
        coordinates = expected_coordinates(lane)
        for index, coordinate in enumerate(coordinates):
            for code in ORDER_CYCLE[index % len(ORDER_CYCLE)]:
                ordered.append((*coordinate, ARM_BY_LANE_CODE[(lane, code)]))
    if len(ordered) != EXPECTED_JOBS or set(ordered) != expected_grid():
        raise RuntimeError("frozen comparator job order is incomplete")
    return ordered


def job_order_sha256() -> str:
    payload = [
        {
            "lane": lane,
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
            "arm_code": ARM_SPECS[arm]["code"],
        }
        for lane, dataset, repeat, fold, arm in expected_ordered_grid()
    ]
    return hashlib.sha256(hardened._canonical_json(payload)).hexdigest()


def expected_ag_ensemble_config() -> dict[str, Any]:
    return {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": TIME_LIMIT_SECONDS,
    }


def expected_resolved_method_hyperparameters() -> dict[str, Any]:
    ensemble = expected_ag_ensemble_config()
    max_time_limit = ensemble.pop("ag.max_time_limit")
    ensemble["ag_args_fit"] = {"max_time_limit": max_time_limit}
    return {"ag_args_ensemble": ensemble}


def expected_fit_kwargs_extra(num_cpus: int) -> dict[str, Any]:
    num_cpus = hardened._exact_int(num_cpus, "expected fit num_cpus")
    if num_cpus != EXPECTED_CHILD_CPUS:
        raise RuntimeError(f"comparator child CPUs must equal {EXPECTED_CHILD_CPUS}")
    return {
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "raise_on_model_failure": True,
        "calibrate": False,
        "num_cpus": num_cpus,
    }


def _experiment_suffix(lane: str, code: str) -> str:
    if lane not in LANES or code not in ENGINE_CODES:
        raise RuntimeError("invalid comparator experiment identity")
    return f"_c1_same_machine_{lane}_{code}"


def _experiment_name(arm: str) -> str:
    spec = ARM_SPECS[arm]
    prefix = ENGINE_SPECS[spec["engine"]]["display_name"]
    return f"{prefix}{_experiment_suffix(spec['lane'], spec['code'])}_BAG_L1"


def frozen_protocol() -> dict[str, Any]:
    """Return the JSON-serializable standalone characterization protocol."""
    return {
        "task_split_counts": {
            dataset: {"task_id": task_id, "registered_split_count": count}
            for dataset, (task_id, count) in TASK_SPLIT_COUNTS.items()
        },
        "lanes": {
            PRIMARY_LANE: {
                "role": "out_of_box_product_default_characterization",
                "coordinates": [
                    {"dataset": d, "repeat": r, "fold": f}
                    for _, d, r, f in expected_coordinates(PRIMARY_LANE)
                ],
                "expected_jobs": EXPECTED_PRIMARY_JOBS,
                "pool_with_other_lanes": False,
            },
            ORDINAL_DIAGNOSTIC_LANE: {
                "role": "source_declared_safe_ordinal_diagnostic",
                "coordinates": [
                    {"dataset": d, "repeat": r, "fold": f}
                    for _, d, r, f in expected_coordinates(
                        ORDINAL_DIAGNOSTIC_LANE
                    )
                ],
                "expected_jobs": EXPECTED_DIAGNOSTIC_JOBS,
                "pool_with_other_lanes": False,
                "policy_advancement_allowed": False,
            },
        },
        "arms": {
            arm: {
                **{name: value for name, value in spec.items() if name != "config"},
                "manual_config": dict(spec["config"]),
                "official_defaults": True,
            }
            for arm, spec in ARM_SPECS.items()
        },
        "official_default_disclosure": {
            "darkofit": {
                "iterations": 1_000,
                "early_stopping": True,
                "tree_mode": "catboost",
                "depth": "auto",
                "learning_rate": None,
                "l2_leaf_reg": "auto",
                "resolved_l2_leaf_reg": 3.0,
                "max_bins": 254,
                "ts_permutations": 1,
                "linear_residual": False,
                "ordered_boosting": "auto",
                "resolved_ordered_boosting_scalar_regression": False,
                "use_best_model": True,
            },
            "chimeraboost": {
                "n_estimators": 10_000,
                "early_stopping": True,
                "resolved_learning_rate": 0.1,
                "resolved_depth": 6,
                "l2_leaf_reg": 1.0,
                "max_bins": 128,
                "cat_n_permutations": 4,
                "ordered_boosting": False,
                "min_child_weight": 1.0,
                "resolved_patience": 50,
                "linear_leaves": None,
            },
            "catboost": {
                "iterations": 10_000,
                "learning_rate": 0.05,
                "allow_writing_files": False,
                "eval_metric": "RMSE",
                "use_best_model_with_eval_set": True,
                "early_stopping": "autogluon_adaptive",
            },
        },
        "order_cycle_codes": [list(row) for row in ORDER_CYCLE],
        "order_cycle_scope": {
            PRIMARY_LANE: "continuous_across_all_39_coordinate_groups",
            ORDINAL_DIAGNOSTIC_LANE: "one_complete_six_group_cycle",
        },
        "execution_dispatch": [
            {
                "lane": PRIMARY_LANE,
                "expected_jobs": EXPECTED_PRIMARY_JOBS,
                "task_groups_are_input_contiguous": True,
                "register_results_in_context": False,
            },
            {
                "lane": ORDINAL_DIAGNOSTIC_LANE,
                "expected_jobs": EXPECTED_DIAGNOSTIC_JOBS,
                "task_groups_are_input_contiguous": True,
                "register_results_in_context": False,
            },
        ],
        "ordered_job_sha256": job_order_sha256(),
        "expected_coordinates": EXPECTED_COORDINATES,
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "bag_folds": 8,
        "bag_sets": 1,
        "seed_policy": "fold-wise",
        "seed_configuration": {
            "model_random_seed": 0,
            "vary_seed_across_folds": True,
        },
        "fold_fitting_strategy": "sequential_local",
        "num_cpus": EXPECTED_CHILD_CPUS,
        "num_gpus": 0,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "competitor_stop_reason_policy": {
            "known_time_or_deadline_invalidates": True,
            "unresolved_external_callback_outcome_allowed": True,
            "unresolved_is_reported_not_inferred": True,
            "qualification": (
                "may_include_early_stopping_time_memory_or_no_legal_split"
            ),
        },
        "analysis": {
            "primary_dataset_weighting": "equal_1_over_13",
            "ordinal_diagnostic_dataset_weighting": "equal_1_over_2",
            "estimator": "paired_log_ratio",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resampling": "coordinates_within_each_fixed_dataset",
            "lanes_pooled": False,
        },
        "chimera_source": {
            "version": CHIMERABOOST_VERSION,
            "exact_git_commit": CHIMERABOOST_TAG_COMMIT,
            "hidden_import_warmup": "disabled",
        },
        "catboost_source": {"version": CATBOOST_VERSION},
        "warmup_outside_run_jobs": True,
        "warmup": {
            "kind": WARMUP_KIND,
            "schema_version": WARMUP_SCHEMA_VERSION,
            "stage_names": list(WARMUP_STAGE_NAMES),
            "engine_routes": ["numeric_regression", "categorical_regression"],
            "prediction_batch_routes": ["small", "large"],
            "chimera_minimum_training_rows": 1_000,
            "chimera_explicit_eval_set": True,
            "thread_count": EXPECTED_CHILD_CPUS,
        },
        "memory": {
            "primary": "incremental_peak_mem_cpu_minus_min_mem_cpu",
            "secondary": "raw_process_peak_mem_cpu",
        },
        "freshness": (
            "non-resume execution requires a nonexistent/empty output; resume "
            "invalidates whole lane-specific D/M/C coordinate triads"
        ),
    }


def protocol_sha256() -> str:
    return hashlib.sha256(hardened._canonical_json(frozen_protocol())).hexdigest()


def _run_git(args: list[str], *, cwd: Path) -> str:
    return hardened._run_command(["git", *args], cwd=cwd)


def _validate_chimeraboost_warmup_environment(value: str | None) -> str | None:
    """Match the exact disabled-value contract in ChimeraBoost v0.14.1."""
    if value is not None and value != "" and value.strip() != "0":
        raise RuntimeError(
            "CHIMERABOOST_WARMUP must be unset, empty, or zero"
        )
    return value


def activate_chimeraboost_checkout(path: Path) -> dict[str, Any]:
    """Activate and prove the exact clean 0.14.1 tag checkout."""
    checkout = path.expanduser().resolve()
    if not checkout.is_dir() or not (checkout / "chimeraboost/__init__.py").is_file():
        raise RuntimeError(f"ChimeraBoost checkout is missing: {checkout}")
    if _run_git(["rev-parse", "HEAD"], cwd=checkout) != CHIMERABOOST_TAG_COMMIT:
        raise RuntimeError("ChimeraBoost checkout is not the exact v0.14.1 commit")
    if _run_git(["status", "--porcelain", "--untracked-files=all"], cwd=checkout):
        raise RuntimeError("ChimeraBoost v0.14.1 checkout is not clean")
    tags = _run_git(["tag", "--points-at", "HEAD"], cwd=checkout).splitlines()
    if "v0.14.1" not in tags:
        raise RuntimeError("ChimeraBoost checkout is not tagged v0.14.1")
    _validate_chimeraboost_warmup_environment(
        os.environ.get("CHIMERABOOST_WARMUP")
    )
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    importlib.invalidate_caches()
    module = importlib.import_module("chimeraboost")
    module_file = Path(module.__file__).resolve()
    try:
        module_file.relative_to(checkout)
    except ValueError as exc:
        raise RuntimeError("imported chimeraboost is not the frozen checkout") from exc
    if getattr(module, "__version__", None) != CHIMERABOOST_VERSION:
        raise RuntimeError("imported ChimeraBoost version is not 0.14.1")
    return {
        "repository": str(checkout),
        "git_head": CHIMERABOOST_TAG_COMMIT,
        "git_tree": _run_git(["rev-parse", "HEAD^{tree}"], cwd=checkout),
        "git_tag": "v0.14.1",
        "git_remote_origin": hardened._sanitize_git_remote(
            _run_git(["remote", "get-url", "origin"], cwd=checkout)
        ),
        "status": "",
        "module_file": str(module_file),
        "module_sha256": hardened._sha256_file(module_file),
        "hidden_import_warmup": "disabled",
    }


def validate_official_defaults(model_classes: Mapping[str, type]) -> None:
    """Fail closed if any inherited official default changes."""
    expected = {
        "darkofit": {
            "iterations": 1_000,
            "early_stopping": True,
            "tree_mode": "catboost",
            "diagnostic_warnings": "never",
        },
        "chimeraboost": {"n_estimators": 10_000, "early_stopping": True},
        "catboost": {
            "iterations": 10_000,
            "learning_rate": 0.05,
            "allow_writing_files": False,
            "eval_metric": "RMSE",
        },
    }
    from autogluon.core.metrics import get_metric

    for engine, spec in ENGINE_SPECS.items():
        for class_key in ("native_class", "ordinal_class"):
            cls = model_classes[spec[class_key]]
            probe = cls(
                path="",
                name=f"Comparator{engine}{class_key}Probe",
                problem_type="regression",
                eval_metric="root_mean_squared_error",
                hyperparameters={},
            )
            probe.stopping_metric = get_metric("root_mean_squared_error")
            probe._set_default_params()
            if probe._get_model_params() != expected[engine]:
                raise RuntimeError(
                    f"{engine} official adapter defaults changed: "
                    f"{probe._get_model_params()!r}"
                )
    from chimeraboost import ChimeraBoostRegressor

    estimator = ChimeraBoostRegressor(n_estimators=10_000, early_stopping=True)
    params = estimator.get_params(deep=False)
    constructor_expected = {
        "learning_rate": None,
        "depth": None,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "cat_n_permutations": 4,
        "ordered_boosting": False,
        "min_child_weight": 1.0,
        "early_stopping_rounds": None,
        "linear_leaves": None,
    }
    if any(params.get(name) != value for name, value in constructor_expected.items()):
        raise RuntimeError("ChimeraBoost estimator defaults changed from v0.14.1")


def build_experiments(
    *, model_classes: Mapping[str, type], config_generator_cls, time_limit: float
) -> dict[str, Any]:
    """Build one empty-manual-config official experiment per lane/engine."""
    if float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError(
            f"frozen campaign time_limit must be {TIME_LIMIT_SECONDS:g} seconds"
        )
    experiments: dict[str, Any] = {}
    for arm, spec in ARM_SPECS.items():
        generator = config_generator_cls(
            model_cls=model_classes[spec["model_cls"]],
            manual_configs=[{}],
            search_space={},
        )
        generated = generator.generate_all_bag_experiments(
            num_random_configs=0,
            name_id_suffix=f"_same_machine_{spec['lane']}_{spec['code']}",
            add_seed="fold-wise",
            fold_fitting_strategy="sequential_local",
            time_limit=time_limit,
        )
        if len(generated) != 1 or generated[0].name != _experiment_name(arm):
            raise RuntimeError(f"unexpected comparator experiment for {arm}")
        experiments[arm] = generated[0]
    return experiments


def _job_coordinate(job) -> tuple[str, str, int, int]:
    arm = _job_arm(job)
    return (
        ARM_SPECS[arm]["lane"],
        str(job.task.dataset),
        int(job.task.repeat),
        int(job.task.fold),
    )


def _job_arm(job) -> str:
    experiment = getattr(job, "experiment", None)
    name = getattr(experiment, "name", "")
    matches = [arm for arm in ARM_SPECS if name == _experiment_name(arm)]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify comparator arm from {name!r}")
    arm = matches[0]
    spec = ARM_SPECS[arm]
    method = getattr(experiment, "method_kwargs", None)
    if not isinstance(method, Mapping):
        raise RuntimeError("comparator job has no resolved method settings")
    raw = dict(method.get("model_hyperparameters", {}))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    if (
        raw
        or ag_args != {"name_suffix": _experiment_suffix(spec["lane"], spec["code"])}
        or not isinstance(ag_ensemble, Mapping)
        or dict(ag_ensemble) != expected_ag_ensemble_config()
        or getattr(method.get("model_cls"), "__name__", None) != spec["model_cls"]
    ):
        raise RuntimeError(f"comparator job does not match frozen arm {arm}")
    return arm


def build_comparator_jobs(context, experiments: Mapping[str, Any]) -> list[Any]:
    jobs: list[Any] = []
    for arm, experiment in experiments.items():
        lane = ARM_SPECS[arm]["lane"]
        task_ids = [
            TASKS[dataset]
            for dataset in (
                TASKS if lane == PRIMARY_LANE else DIAGNOSTIC_DATASETS
            )
        ]
        jobs.extend(
            context.build_jobs(
                [experiment], task_ids=task_ids, split_indices=SPLIT_INDICES
            )
        )
    return order_comparator_jobs(jobs)


def order_comparator_jobs(jobs: Iterable[Any]) -> list[Any]:
    grouped: dict[tuple[str, str, int, int], dict[str, Any]] = defaultdict(dict)
    allowed = set(expected_coordinates())
    for job in jobs:
        arm = _job_arm(job)
        coordinate = (
            ARM_SPECS[arm]["lane"],
            str(job.task.dataset),
            int(job.task.repeat),
            int(job.task.fold),
        )
        if coordinate not in allowed:
            raise RuntimeError(f"unexpected comparator coordinate: {coordinate}")
        if arm in grouped[coordinate]:
            raise RuntimeError(f"duplicate {arm} job for {coordinate}")
        grouped[coordinate][arm] = job
    if set(grouped) != allowed:
        raise RuntimeError("built jobs do not match frozen comparator coordinates")
    ordered = []
    for lane, dataset, repeat, fold, arm in expected_ordered_grid():
        coordinate = (lane, dataset, repeat, fold)
        expected_arms = {
            name for name, spec in ARM_SPECS.items() if spec["lane"] == lane
        }
        if set(grouped[coordinate]) != expected_arms:
            raise RuntimeError(f"coordinate {coordinate} does not contain all engines")
        ordered.append(grouped[coordinate][arm])
    if len(ordered) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, built {len(ordered)}")
    return ordered


def ordering_audit(jobs: Iterable[Any]) -> dict[str, Any]:
    observed = [(*_job_coordinate(job), _job_arm(job)) for job in jobs]
    expected = expected_ordered_grid()
    if observed != expected:
        raise RuntimeError("built job order does not match the frozen cycle")
    lane_counts: dict[str, dict[str, list[int]]] = {}
    cursor = 0
    for lane in LANES:
        coordinates = expected_coordinates(lane)
        counts = {engine: [0, 0, 0] for engine in ENGINE_SPECS}
        for index, coordinate in enumerate(coordinates):
            group = observed[cursor : cursor + 3]
            cursor += 3
            codes = ORDER_CYCLE[index % len(ORDER_CYCLE)]
            arms = [ARM_BY_LANE_CODE[(lane, code)] for code in codes]
            if [row[4] for row in group] != arms or any(
                row[:4] != coordinate for row in group
            ):
                raise RuntimeError("comparator order cycle does not match")
            for position, code in enumerate(codes):
                counts[ENGINE_CODES[code]][position] += 1
        lane_counts[lane] = counts
    expected_positions = {
        PRIMARY_LANE: [13, 13, 13],
        ORDINAL_DIAGNOSTIC_LANE: [2, 2, 2],
    }
    if cursor != EXPECTED_JOBS or any(
        values != expected_positions[lane]
        for lane, counts in lane_counts.items()
        for values in counts.values()
    ):
        raise RuntimeError("comparator positions are not perfectly balanced")
    return {
        "job_order_sha256": job_order_sha256(),
        "lane_position_counts": {
            lane: {
                engine: {
                    "first": values[0],
                    "second": values[1],
                    "third": values[2],
                }
                for engine, values in counts.items()
            }
            for lane, counts in lane_counts.items()
        },
    }


def partition_jobs_for_dispatch(
    jobs: Iterable[Any],
) -> tuple[list[Any], list[Any]]:
    """Split the audited grid into two task-contiguous TabArena dispatches.

    TabArena groups each ``run_jobs`` input by task before execution.  The full
    grid contains the Airfoil and Diamonds tasks in both lanes, so submitting
    it in one call would pull diagnostic jobs into the primary task blocks.
    Each returned lane is already task-contiguous, making that internal
    grouping a no-op while preserving the frozen per-coordinate D/M/C cycle.
    """
    ordered = list(jobs)
    ordering_audit(ordered)
    partitions: dict[str, list[Any]] = {lane: [] for lane in LANES}
    for job in ordered:
        partitions[_job_coordinate(job)[0]].append(job)
    expected_counts = {
        PRIMARY_LANE: EXPECTED_PRIMARY_JOBS,
        ORDINAL_DIAGNOSTIC_LANE: EXPECTED_DIAGNOSTIC_JOBS,
    }
    for lane in LANES:
        lane_jobs = partitions[lane]
        expected = [row for row in expected_ordered_grid() if row[0] == lane]
        observed = [(*_job_coordinate(job), _job_arm(job)) for job in lane_jobs]
        if len(lane_jobs) != expected_counts[lane] or observed != expected:
            raise RuntimeError(f"{lane} dispatch does not match the frozen order")
        completed_tasks: set[str] = set()
        current_task: str | None = None
        for _, dataset, _, _, _ in observed:
            if dataset == current_task:
                continue
            if dataset in completed_tasks:
                raise RuntimeError(f"{lane} dispatch is not task-contiguous")
            if current_task is not None:
                completed_tasks.add(current_task)
            current_task = dataset
    reconstructed = (
        partitions[PRIMARY_LANE] + partitions[ORDINAL_DIAGNOSTIC_LANE]
    )
    if len(reconstructed) != len(ordered) or any(
        actual is not expected
        for actual, expected in zip(reconstructed, ordered)
    ):
        raise RuntimeError("lane dispatch partition changed the frozen job sequence")
    return partitions[PRIMARY_LANE], partitions[ORDINAL_DIAGNOSTIC_LANE]


def resolve_and_pin_child_cpu_allocation(jobs: Iterable[Any]) -> int:
    allocations = set()
    methods = []
    seen_experiments = set()
    seen_arms = set()
    for job in jobs:
        experiment = job.experiment
        if id(experiment) in seen_experiments:
            continue
        seen_experiments.add(id(experiment))
        arm = _job_arm(job)
        seen_arms.add(arm)
        method = experiment.method_kwargs
        methods.append(method)
        fit_kwargs = method.get("fit_kwargs")
        if not isinstance(fit_kwargs, dict):
            raise RuntimeError("comparator experiment has no fit_kwargs")
        raw_total = fit_kwargs.get("num_cpus", "auto")
        total = (
            hardened._autogluon_cpu_count()
            if raw_total in (None, "auto")
            else hardened._exact_int(raw_total, "experiment num_cpus")
        )
        cls = method.get("model_cls")
        probe = cls(
            path="",
            name="ComparatorResourceProbe",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            hyperparameters={},
        )
        default_cpus, default_gpus = probe._get_default_resources()
        if float(default_gpus) != 0.0:
            raise RuntimeError("comparator resources must be CPU-only")
        allocations.add(min(total, hardened._exact_int(default_cpus, "child CPUs")))
    if seen_arms != set(ARM_SPECS) or allocations != {EXPECTED_CHILD_CPUS}:
        raise RuntimeError(
            f"all comparator arms must resolve to exactly {EXPECTED_CHILD_CPUS} CPUs"
        )
    for method in methods:
        method["fit_kwargs"]["num_cpus"] = EXPECTED_CHILD_CPUS
    return EXPECTED_CHILD_CPUS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument(
        "--chimeraboost-path", type=Path, default=DEFAULT_CHIMERABOOST_PATH
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit the exact 135-job grid without creating output or fitting",
    )
    args = parser.parse_args(argv)
    if args.time_limit != TIME_LIMIT_SECONDS:
        parser.error(f"--time-limit is frozen at {TIME_LIMIT_SECONDS:g} seconds")
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")
    return args


def _reject_symlink_components(path: Path, field: str) -> None:
    """Reject every existing symlink component of an absolute managed path."""
    if not path.is_absolute():
        raise RuntimeError(f"{field} must be an absolute path")
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        try:
            component_metadata = cursor.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field} component: {cursor}") from exc
        if stat.S_ISLNK(component_metadata.st_mode):
            raise RuntimeError(f"{field} has a symbolic-link component: {cursor}")


def _validate_comparator_output_state(output_dir: Path, *, resume: bool) -> None:
    """Reject symlinked/non-directory roots and unsafe resume manifests."""
    _reject_symlink_components(output_dir, "campaign output")
    try:
        root_metadata = output_dir.lstat()
    except FileNotFoundError:
        if resume:
            raise RuntimeError(
                f"cannot resume missing output directory: {output_dir}"
            )
        hardened.validate_output_state(output_dir, resume=resume)
        return
    except OSError as exc:
        raise RuntimeError(f"could not inspect output directory: {output_dir}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise RuntimeError("campaign output must be a real directory, not a link")
    if resume and not hardened._require_regular_archive_source(
        output_dir / MANIFEST_FILENAME, "comparator run manifest"
    ):
        raise RuntimeError("resume requested but run_manifest.json is missing")
    hardened.validate_output_state(output_dir, resume=resume)


def _load_model_classes() -> dict[str, type]:
    try:
        from benchmarks.tabarena_comparator_adapters import (
            ComparatorCatBoostModel,
            ComparatorChimeraBoostModel,
            ComparatorDarkoFitModel,
            ComparatorOrdinalCatBoostModel,
            ComparatorOrdinalChimeraBoostModel,
            ComparatorOrdinalDarkoFitModel,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_comparator_adapters import (
            ComparatorCatBoostModel,
            ComparatorChimeraBoostModel,
            ComparatorDarkoFitModel,
            ComparatorOrdinalCatBoostModel,
            ComparatorOrdinalChimeraBoostModel,
            ComparatorOrdinalDarkoFitModel,
        )
    classes = (
        ComparatorDarkoFitModel,
        ComparatorOrdinalDarkoFitModel,
        ComparatorChimeraBoostModel,
        ComparatorOrdinalChimeraBoostModel,
        ComparatorCatBoostModel,
        ComparatorOrdinalCatBoostModel,
    )
    return {cls.__name__: cls for cls in classes}


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    _validate_comparator_output_state(output_dir, resume=args.resume)
    old_dont_write = (
        _DONT_WRITE_BYTECODE_BEFORE_IMPORT
        if _DRY_RUN_REQUESTED_AT_IMPORT
        else sys.dont_write_bytecode
    )
    sys.dont_write_bytecode = True
    try:
        chimera_source = activate_chimeraboost_checkout(args.chimeraboost_path)
        model_classes = _load_model_classes()
        validate_official_defaults(model_classes)

        from tabarena.contexts import TabArenaContext
        from tabarena.utils.config_utils import ConfigGenerator

        context = TabArenaContext()
        experiments = build_experiments(
            model_classes=model_classes,
            config_generator_cls=ConfigGenerator,
            time_limit=args.time_limit,
        )
        jobs = build_comparator_jobs(context, experiments)
        ordering = ordering_audit(jobs)
        child_cpus = resolve_and_pin_child_cpu_allocation(jobs)
        print(
            f"built {len(jobs)} same-machine jobs "
            f"({EXPECTED_PRIMARY_JOBS} primary, "
            f"{EXPECTED_DIAGNOSTIC_JOBS} diagnostic, "
            f"{EXPECTED_CHILD_FITS} child fits); pinned child CPUs={child_cpus}"
        )
        print(
            "order audit "
            + json.dumps(
                {
                    "job_order_sha256": ordering["job_order_sha256"],
                    "lane_position_counts": ordering["lane_position_counts"],
                },
                sort_keys=True,
            )
        )
        if args.dry_run:
            source = collect_source_provenance(
                output_dir=output_dir,
                chimeraboost_path=args.chimeraboost_path,
            )
            if source["chimeraboost"] != chimera_source:
                raise RuntimeError(
                    "ChimeraBoost source changed during the dry-run audit"
                )
            manifest = build_run_manifest(
                output_dir=output_dir,
                source=source,
                ordering=ordering,
                resolved_child_num_cpus=child_cpus,
            )
            if (
                manifest["source"] != source
                or manifest["ordering_audit"] != ordering
                or manifest["resolved_child_num_cpus"] != child_cpus
                or manifest["protocol_sha256"] != protocol_sha256()
                or manifest["job_order_sha256"] != job_order_sha256()
            ):
                raise RuntimeError("dry-run manifest audit is inconsistent")
            print(
                "dry-run source/runtime audit complete "
                + json.dumps(
                    {
                        "git_head": source["git_head"],
                        "protocol_sha256": manifest["protocol_sha256"],
                        "python_version": manifest["runtime"]["python_version"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        return _run_campaign(
            args=args,
            output_dir=output_dir,
            context=context,
            jobs=jobs,
            ordering=ordering,
            child_cpus=child_cpus,
            chimera_source=chimera_source,
        )
    finally:
        sys.dont_write_bytecode = old_dont_write


def _source_file_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required source file is missing or unsafe: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hardened._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _module_file_artifact(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"module has no file provenance: {module_name}")
    artifact = _source_file_artifact(Path(raw).resolve())
    artifact["module"] = module_name
    return artifact


def _installed_distribution_provenance(
    distribution_name: str,
    module_names: tuple[str, ...],
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Bind every installed byte plus the imported module identity."""
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{distribution_name} is not installed") from exc
    if expected_version is not None and distribution.version != expected_version:
        raise RuntimeError(
            f"{distribution_name} must be {expected_version}, "
            f"found {distribution.version}"
        )
    distribution_files = distribution.files
    if not distribution_files:
        raise RuntimeError(f"{distribution_name} has no installed file manifest")
    files: dict[str, Any] = {}
    installed_paths: dict[Path, str] = {}
    record_verified_count = 0
    unhashed_record_paths: list[str] = []
    for relative in sorted(distribution_files, key=str):
        text = str(relative)
        if not text or text in files:
            raise RuntimeError(
                f"{distribution_name} installed file manifest is not unique"
            )
        raw_path = Path(distribution.locate_file(relative))
        if raw_path.is_symlink() or not raw_path.is_file():
            raise RuntimeError(
                f"unsafe {distribution_name} distribution file: {raw_path}"
            )
        path = raw_path.resolve()
        if path in installed_paths:
            raise RuntimeError(
                f"{distribution_name} file manifest resolves one file twice"
            )
        payload = path.read_bytes()
        declared_hash = relative.hash
        declared_size = relative.size
        if declared_hash is None or declared_size is None:
            if (
                declared_hash is not None
                or declared_size is not None
                or Path(text).name != "RECORD"
            ):
                raise RuntimeError(
                    f"{distribution_name} RECORD metadata is incomplete for {text}"
                )
            record_sha256 = None
            record_size_bytes = None
            unhashed_record_paths.append(text)
        else:
            if declared_hash.mode != "sha256":
                raise RuntimeError(
                    f"{distribution_name} uses a non-SHA256 RECORD entry"
                )
            encoded = base64.urlsafe_b64encode(
                hashlib.sha256(payload).digest()
            ).rstrip(b"=").decode("ascii")
            if encoded != declared_hash.value or len(payload) != declared_size:
                raise RuntimeError(
                    f"{distribution_name} installed bytes disagree with RECORD: {text}"
                )
            record_sha256 = declared_hash.value
            record_size_bytes = declared_size
            record_verified_count += 1
        files[text] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "record_sha256": record_sha256,
            "record_size_bytes": record_size_bytes,
        }
        installed_paths[path] = text
    if unhashed_record_paths != [
        text for text in files if Path(text).name == "RECORD"
    ]:
        raise RuntimeError(f"{distribution_name} RECORD exception set is not exact")
    if not module_names or len(set(module_names)) != len(module_names):
        raise RuntimeError(f"{distribution_name} module identity set is invalid")
    modules: dict[str, dict[str, str]] = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        raw_module_path = getattr(module, "__file__", None)
        if not raw_module_path:
            raise RuntimeError(f"imported module has no file identity: {module_name}")
        module_path = Path(raw_module_path).resolve()
        if module_path not in installed_paths:
            raise RuntimeError(
                f"imported {module_name} module is not in the attested distribution"
            )
        modules[module_name] = {
            "path": str(module_path),
            "distribution_path": installed_paths[module_path],
        }
    version_module = importlib.import_module(module_names[0])
    if expected_version is not None and (
        getattr(version_module, "__version__", None) != expected_version
    ):
        raise RuntimeError(
            f"imported {module_names[0]} version does not match its distribution"
        )
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "modules": modules,
        "files": files,
        "record_integrity": {
            "algorithm": "sha256",
            "verified_file_count": record_verified_count,
            "unhashed_record_paths": unhashed_record_paths,
        },
    }


def _catboost_distribution_provenance() -> dict[str, Any]:
    """Bind every file installed by the exact CatBoost 1.2.10 wheel."""
    return _installed_distribution_provenance(
        "catboost",
        CATBOOST_PROVENANCE_MODULES,
        expected_version=CATBOOST_VERSION,
    )


def collect_source_provenance(
    *, output_dir: Path | None, chimeraboost_path: Path
) -> dict[str, Any]:
    """Bind all three engines, adapters, analyzers, and exact Git trees."""
    repo_root = Path(__file__).resolve().parents[1]
    status = hardened._repository_status(repo_root, output_dir)
    if status:
        raise RuntimeError(
            "campaign repository is not clean; commit or remove every change "
            f"before execution:\n{status}"
        )
    files: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"required comparator source is missing: {relative}")
        files[str(relative)] = {
            "sha256": hardened._sha256_file(path),
            "git_blob": hardened._run_command(
                ["git", "hash-object", str(path)], cwd=repo_root
            ),
        }
    darkofit_import = hardened.collect_git_dependency_provenance(
        "darkofit", output_dir=output_dir
    )
    if Path(darkofit_import["repository"]).resolve() != repo_root.resolve():
        raise RuntimeError("imported darkofit is not this comparator repository")
    chimera = activate_chimeraboost_checkout(chimeraboost_path)
    return {
        "repository": str(repo_root),
        "git_head": hardened._run_command(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ),
        "git_tree": hardened._run_command(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root
        ),
        "relevant_status": status,
        "files": files,
        "darkofit_import": darkofit_import,
        "tabarena": hardened.collect_git_dependency_provenance(
            "tabarena", output_dir=output_dir
        ),
        "chimeraboost": chimera,
        "catboost": _catboost_distribution_provenance(),
        "external_adapter_sources": {
            "autogluon_distributions": {
                name: _installed_distribution_provenance(name, module_names)
                for name, module_names in AUTOGLUON_PROVENANCE_MODULES.items()
            },
            "tabarena_chimeraboost_model": _module_file_artifact(
                "tabarena.models.chimeraboost.model"
            ),
        },
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    versions["chimeraboost"] = CHIMERABOOST_VERSION
    return versions


def collect_runtime_provenance() -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "environment": {
            key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
        },
        "hardware": hardened.collect_runtime_hardware_provenance(),
    }


def build_run_manifest(
    *,
    output_dir: Path,
    source: Mapping[str, Any],
    ordering: Mapping[str, Any],
    resolved_child_num_cpus: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "resolved_child_num_cpus": int(resolved_child_num_cpus),
        "protocol_sha256": protocol_sha256(),
        "job_order_sha256": job_order_sha256(),
        "protocol": frozen_protocol(),
        "ordering_audit": dict(ordering),
        "source": dict(source),
        "runtime": collect_runtime_provenance(),
    }


def write_or_validate_run_manifest(
    output_dir: Path, manifest: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    _validate_comparator_output_state(output_dir, resume=resume)
    normalized_manifest = _json_mapping(manifest, "candidate run manifest")
    path = output_dir / MANIFEST_FILENAME
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        hardened._atomic_write_json(path, normalized_manifest)
        return normalized_manifest
    if not hardened._require_regular_archive_source(
        path, "comparator run manifest"
    ):
        raise RuntimeError("resume comparator manifest is missing")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        existing_raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
        existing = _json_mapping(existing_raw, "existing run manifest")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("could not read existing comparator manifest") from exc
    expected_fields = set(normalized_manifest)
    if set(existing) != expected_fields:
        raise RuntimeError("resume comparator manifest fields are not exact")
    created_at = existing.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise RuntimeError("resume comparator manifest timestamp is invalid")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise RuntimeError("resume comparator manifest timestamp is invalid") from exc
    if parsed_created_at.tzinfo is None:
        raise RuntimeError("resume comparator manifest timestamp has no timezone")
    stable = (
        "schema_version",
        "kind",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "job_order_sha256",
        "protocol",
        "ordering_audit",
        "source",
        "runtime",
    )
    mismatches = [
        name
        for name in stable
        if existing.get(name) != normalized_manifest.get(name)
    ]
    if mismatches:
        raise RuntimeError(
            "resume manifest does not match the frozen comparator campaign: "
            + ", ".join(mismatches)
        )
    return existing


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def _finite(
    value: Any, field: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{field} must be finite")
    if positive and number <= 0.0:
        raise RuntimeError(f"{field} must be positive")
    if nonnegative and number < 0.0:
        raise RuntimeError(f"{field} must be nonnegative")
    return number


def expected_child_hyperparameters(engine: str, child_fold: int) -> dict[str, Any]:
    if engine not in ENGINE_SPECS:
        raise RuntimeError(f"unexpected comparator engine: {engine}")
    if type(child_fold) is not int or child_fold not in range(8):
        raise RuntimeError("child_fold must be an integer from 0 through 7")
    if engine == "darkofit":
        return {
            "iterations": 1_000,
            "early_stopping": True,
            "tree_mode": "catboost",
            "diagnostic_warnings": "never",
            "random_state": child_fold,
        }
    if engine == "chimeraboost":
        return {
            "n_estimators": 10_000,
            "early_stopping": True,
            "random_state": child_fold,
        }
    return {
        "iterations": 10_000,
        "learning_rate": 0.05,
        "allow_writing_files": False,
        "eval_metric": "RMSE",
        "random_seed": child_fold,
    }


def _json_mapping(value: Any, field: str) -> dict[str, Any]:
    mapping = dict(_as_mapping(value, field))
    try:
        return json.loads(hardened._canonical_json(mapping))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} is not strict JSON") from exc


def _arm_from_method(method: Mapping[str, Any], field: str) -> str:
    raw = dict(_as_mapping(method.get("model_hyperparameters"), field))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    model_cls = method.get("model_cls")
    matches = []
    for arm, spec in ARM_SPECS.items():
        if (
            not raw
            and model_cls == spec["model_cls"]
            and ag_args
            == {"name_suffix": _experiment_suffix(spec["lane"], spec["code"])}
        ):
            matches.append(arm)
    if len(matches) != 1:
        raise RuntimeError(f"{field} does not match exactly one comparator arm")
    if not isinstance(ag_ensemble, Mapping) or dict(
        ag_ensemble
    ) != expected_ag_ensemble_config():
        raise RuntimeError(f"{field} bag seed/time configuration is not frozen")
    return matches[0]


def _validate_comparator_fit(
    value: Any, *, arm: str, field: str
) -> dict[str, Any]:
    """Validate common telemetry while retaining engine-specific fields."""
    fit = _json_mapping(value, field)
    spec = ARM_SPECS[arm]
    required = {
        "schema_version",
        "engine",
        "iterations_requested",
        "best_iteration",
        "rounds_retained",
        "resolved_params",
        "num_cpus",
        "num_gpus",
        "stop_reason",
    }
    if not required.issubset(fit):
        raise RuntimeError(f"{field} common telemetry is incomplete")
    if fit["schema_version"] != 1 or fit["engine"] != spec["engine"]:
        raise RuntimeError(f"{field} identity does not match the arm")
    requested = hardened._exact_int(
        fit["iterations_requested"], f"{field}.iterations_requested"
    )
    best = hardened._exact_int(fit["best_iteration"], f"{field}.best_iteration")
    retained = hardened._exact_int(
        fit["rounds_retained"], f"{field}.rounds_retained"
    )
    expected_requested = 1_000 if spec["engine"] == "darkofit" else 10_000
    if (
        requested != expected_requested
        or not 0 <= best <= requested
        or not 0 <= retained <= requested
        or best != retained
    ):
        raise RuntimeError(f"{field} fitted iteration metadata is inconsistent")
    if (
        hardened._exact_int(fit["num_cpus"], f"{field}.num_cpus")
        != EXPECTED_CHILD_CPUS
        or hardened._exact_int(fit["num_gpus"], f"{field}.num_gpus") != 0
    ):
        raise RuntimeError(f"{field} fitted resources do not match")
    if fit["stop_reason"] is not None and (
        not isinstance(fit["stop_reason"], str) or not fit["stop_reason"]
    ):
        raise RuntimeError(f"{field}.stop_reason must be null or nonempty")
    if fit["stop_reason"] == "time_limit":
        raise RuntimeError(f"{field} hit the wall-clock limit")
    resolved = _json_mapping(fit["resolved_params"], f"{field}.resolved_params")
    fit["resolved_params"] = resolved
    if hardened._exact_int(
        resolved.get("thread_count"), f"{field}.resolved_params.thread_count"
    ) != EXPECTED_CHILD_CPUS:
        raise RuntimeError(f"{field} core thread count does not match")
    if spec["engine"] == "darkofit":
        required_darko = {
            "resolved_learning_rate",
            "requested_tree_mode",
            "selected_tree_mode",
            "selected_lane",
            "deadline_hit",
        }
        if not required_darko.issubset(fit):
            raise RuntimeError(f"{field} DarkoFit telemetry is incomplete")
        _finite(
            fit["resolved_learning_rate"],
            f"{field}.resolved_learning_rate",
            positive=True,
        )
        if (
            fit["requested_tree_mode"] != "catboost"
            or fit["selected_tree_mode"] != "catboost"
            or fit["selected_lane"] != "boosting"
        ):
            raise RuntimeError(f"{field} DarkoFit selected an unexpected lane")
        if fit["deadline_hit"] is not False:
            raise RuntimeError(f"{field} DarkoFit hit its deadline")
    elif spec["engine"] == "chimeraboost":
        required_chimera = {
            "resolved_learning_rate",
            "selected_lane",
            "linear_leaves_selected",
        }
        if not required_chimera.issubset(fit):
            raise RuntimeError(f"{field} ChimeraBoost telemetry is incomplete")
        if not math.isclose(
            _finite(
            fit["resolved_learning_rate"],
            f"{field}.resolved_learning_rate",
            positive=True,
            ),
            0.1,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{field} ChimeraBoost learning rate changed")
        selected_linear = fit["linear_leaves_selected"]
        expected_lane = "linear" if selected_linear is True else "constant"
        if selected_linear not in (True, False) or fit["selected_lane"] != expected_lane:
            raise RuntimeError(f"{field} ChimeraBoost selected-lane audit changed")
    else:
        required_cat = {"resolved_learning_rate", "tree_count"}
        if not required_cat.issubset(fit):
            raise RuntimeError(f"{field} CatBoost telemetry is incomplete")
        if not math.isclose(
            _finite(
            fit["resolved_learning_rate"],
            f"{field}.resolved_learning_rate",
            positive=True,
            ),
            0.05,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{field} CatBoost learning rate changed")
        if hardened._exact_int(fit["tree_count"], f"{field}.tree_count") != retained:
            raise RuntimeError(f"{field} CatBoost tree count does not match")
        if resolved.get("task_type", "CPU") != "CPU":
            raise RuntimeError(f"{field} CatBoost did not use the CPU lane")
    return fit


def _validate_representation(
    value: Any,
    *,
    arm: str,
    dataset: str,
    field: str,
    child_features: list[str],
) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    if spec["representation"] == "safe_ordinal":
        return followon._validate_representation_metadata(
            value,
            arm="ordinal",
            dataset=dataset,
            field=field,
            child_features=child_features,
        )
    representation = _json_mapping(value, field)
    expected_fields = {
        "schema_version",
        "kind",
        "fit_scope",
        "feature_alignment_policy",
        "target_used_by_representation",
        "input_feature_count",
        "output_feature_count",
        "external_feature_schema_sha256",
        "fitted_feature_schema_sha256",
        "categorical_input_columns",
        "fitted_categorical_input_columns",
        "dropped_constant_input_columns",
        "dropped_constant_input_unique_counts",
    }
    if set(representation) != expected_fields:
        raise RuntimeError(f"{field} native representation fields are not exact")
    if (
        representation["schema_version"] != 2
        or representation["kind"] != "native"
        or representation["fit_scope"] != "comparator_child_training_fold"
        or representation["feature_alignment_policy"]
        != "autogluon_child_drop_unique"
    ):
        raise RuntimeError(f"{field} native representation identity changed")
    external_digest = followon._feature_schema_sha256(
        child_features, f"{field}.child_features"
    )
    input_count = hardened._exact_int(
        representation["input_feature_count"], f"{field}.input_feature_count"
    )
    output_count = hardened._exact_int(
        representation["output_feature_count"], f"{field}.output_feature_count"
    )
    for name in (
        "categorical_input_columns",
        "fitted_categorical_input_columns",
        "dropped_constant_input_columns",
        "dropped_constant_input_unique_counts",
    ):
        if not isinstance(representation[name], list):
            raise RuntimeError(f"{field}.{name} must be a list")
    categorical = representation["categorical_input_columns"]
    fitted_categorical = representation["fitted_categorical_input_columns"]
    dropped = representation["dropped_constant_input_columns"]
    dropped_counts = representation["dropped_constant_input_unique_counts"]
    for name, values in (
        ("categorical_input_columns", categorical),
        ("fitted_categorical_input_columns", fitted_categorical),
        ("dropped_constant_input_columns", dropped),
    ):
        if (
            any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values)
            or values != [column for column in child_features if column in set(values)]
        ):
            raise RuntimeError(f"{field}.{name} is not an ordered feature subset")
    if (
        any(type(count) is not int or count != 1 for count in dropped_counts)
        or len(dropped_counts) != len(dropped)
    ):
        raise RuntimeError(f"{field} dropped-feature count audit is invalid")
    dropped_set = set(dropped)
    fitted_features = [
        column for column in child_features if column not in dropped_set
    ]
    expected_fitted_categorical = [
        column for column in categorical if column not in dropped_set
    ]
    if (
        input_count != len(child_features)
        or output_count != len(fitted_features)
        or representation["external_feature_schema_sha256"] != external_digest
        or representation["fitted_feature_schema_sha256"]
        != followon._feature_schema_sha256(
            fitted_features, f"{field}.fitted_features"
        )
        or fitted_categorical != expected_fitted_categorical
        or representation["target_used_by_representation"]
        is not bool(fitted_categorical)
    ):
        raise RuntimeError(f"{field} native representation audit is inconsistent")
    return representation


def parse_result_record(
    record: Mapping[str, Any], *, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one trusted TabArena record and normalize strict JSON rows."""
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: wrong problem type or metric")
    if record.get("imputed", False) not in (False, None):
        raise RuntimeError(f"{source}: imputed result")
    metrics = {
        "test_rmse": _finite(
            record.get("metric_error"), f"{source}: test RMSE", positive=True
        ),
        "val_rmse": _finite(
            record.get("metric_error_val"), f"{source}: val RMSE", positive=True
        ),
        "train_time_s": _finite(
            record.get("time_train_s"), f"{source}: train time", positive=True
        ),
        "infer_time_s": _finite(
            record.get("time_infer_s"), f"{source}: infer time", positive=True
        ),
    }
    memory = _as_mapping(record.get("memory_usage"), f"{source}: memory")
    peak_memory = _finite(
        memory.get("peak_mem_cpu"), f"{source}: peak memory", positive=True
    )
    baseline_memory = _finite(
        memory.get("min_mem_cpu"), f"{source}: baseline memory", nonnegative=True
    )
    if baseline_memory > peak_memory:
        raise RuntimeError(f"{source}: baseline memory exceeds peak memory")
    incremental_memory = peak_memory - baseline_memory

    experiment_metadata = _as_mapping(
        record.get("experiment_metadata"), f"{source}: experiment metadata"
    )
    if (
        experiment_metadata.get("experiment_cls") != "OOFExperimentRunner"
        or experiment_metadata.get("method_cls") != "AGSingleBagWrapper"
    ):
        raise RuntimeError(f"{source}: wrong experiment implementation")

    method = _as_mapping(record.get("method_metadata"), f"{source}: method metadata")
    arm = _arm_from_method(method, f"{source}: model hyperparameters")
    spec = ARM_SPECS[arm]
    lane = spec["lane"]
    engine = spec["engine"]
    if record.get("framework") != _experiment_name(arm):
        raise RuntimeError(f"{source}: framework name does not match arm")
    if dict(
        _as_mapping(method.get("hyperparameters"), f"{source}: resolved params")
    ) != expected_resolved_method_hyperparameters():
        raise RuntimeError(f"{source}: resolved method policy changed")
    if (
        method.get("model_type") != ENGINE_SPECS[engine]["model_type"]
        or method.get("name_prefix") != ENGINE_SPECS[engine]["display_name"]
        or method.get("init_kwargs_extra") != {}
    ):
        raise RuntimeError(f"{source}: resolved model identity changed")
    num_cpus = hardened._exact_int(method.get("num_cpus"), f"{source}: num_cpus")
    num_gpus = hardened._exact_int(method.get("num_gpus"), f"{source}: num_gpus")
    num_cpus_child = hardened._exact_int(
        method.get("num_cpus_child"), f"{source}: num_cpus_child"
    )
    num_gpus_child = hardened._exact_int(
        method.get("num_gpus_child"), f"{source}: num_gpus_child"
    )
    if (
        num_cpus != EXPECTED_CHILD_CPUS
        or num_cpus_child != EXPECTED_CHILD_CPUS
        or num_gpus != 0
        or num_gpus_child != 0
    ):
        raise RuntimeError(f"{source}: resolved resources changed")
    if dict(
        _as_mapping(method.get("fit_kwargs_extra"), f"{source}: fit kwargs")
    ) != expected_fit_kwargs_extra(num_cpus):
        raise RuntimeError(f"{source}: bag fit settings changed")

    task = _as_mapping(record.get("task_metadata"), f"{source}: task metadata")
    dataset = str(task.get("name"))
    repeat = hardened._exact_int(task.get("repeat"), f"{source}: repeat")
    fold = hardened._exact_int(task.get("fold"), f"{source}: fold")
    coordinate = (lane, dataset, repeat, fold)
    if (
        coordinate not in set(expected_coordinates(lane))
        or hardened._exact_int(task.get("tid"), f"{source}: task id")
        != TASKS.get(dataset)
        or task.get("split_idx") not in (None, 3 * repeat + fold)
    ):
        raise RuntimeError(f"{source}: task coordinate is outside its lane")

    outer_fit = _as_mapping(
        method.get("fit_metadata"), f"{source}: outer fit metadata"
    )
    if (
        outer_fit.get("num_cpus") != EXPECTED_CHILD_CPUS
        or outer_fit.get("num_gpus") != 0
        or outer_fit.get("val_in_fit") is not False
        or outer_fit.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer fit metadata changed")
    info = _as_mapping(method.get("info"), f"{source}: model info")
    if (
        info.get("is_valid") is not True
        or info.get("can_infer") is not True
        or info.get("model_type") != "StackerEnsembleModel"
        or info.get("problem_type") != "regression"
        or info.get("eval_metric") != "root_mean_squared_error"
        or info.get("stopping_metric") != "root_mean_squared_error"
        or info.get("num_cpus") != EXPECTED_CHILD_CPUS
        or info.get("num_gpus") != 0
        or info.get("val_in_fit") is not False
        or info.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer model metadata changed")
    outer_features = info.get("features")
    if not isinstance(outer_features, list) or not all(
        isinstance(name, str) for name in outer_features
    ):
        raise RuntimeError(f"{source}: outer feature schema is invalid")
    if hardened._exact_int(
        info.get("num_features"), f"{source}: outer feature count"
    ) != len(outer_features):
        raise RuntimeError(f"{source}: outer feature count changed")
    followon._feature_schema_sha256(outer_features, f"{source}: outer features")

    bag = _as_mapping(info.get("bagged_info"), f"{source}: bag info")
    child_names = [f"S1F{index}" for index in range(1, 9)]
    expected_model_cls = spec["model_cls"]
    if (
        bag.get("num_child_models") != 8
        or bag.get("child_model_type") != expected_model_cls
        or bag.get("child_model_names") != child_names
        or bag.get("_n_repeats") != 1
        or bag.get("_k_per_n_repeat") != [8]
        or bag.get("_random_state") != 1
        or bag.get("bagged_mode") is not True
        or dict(
            _as_mapping(
                bag.get("child_hyperparameters_user"),
                f"{source}: child user params",
            )
        )
        != {}
        or dict(
            _as_mapping(
                bag.get("child_hyperparameters"), f"{source}: child params"
            )
        )
        != expected_child_hyperparameters(engine, 0)
    ):
        raise RuntimeError(f"{source}: bag construction changed")
    expected_ag_args_fit = {
        "max_memory_usage_ratio": 1.0,
        "max_time_limit_ratio": 1.0,
        "max_time_limit": None,
        "min_time_limit": 0,
    }
    bag_ag_args = _as_mapping(
        bag.get("child_ag_args_fit"), f"{source}: child ag args"
    )
    if any(
        bag_ag_args.get(name) != value
        for name, value in expected_ag_args_fit.items()
    ):
        raise RuntimeError(f"{source}: child budget ratios changed")
    children_info = _as_mapping(info.get("children_info"), f"{source}: children")
    if set(children_info) != set(child_names):
        raise RuntimeError(f"{source}: child info set changed")

    child_rows: list[dict[str, Any]] = []
    for child_fold, child_name in enumerate(child_names):
        child = _as_mapping(children_info[child_name], f"{source}: {child_name}")
        child_features = child.get("features")
        if not isinstance(child_features, list) or not all(
            isinstance(name, str) for name in child_features
        ):
            raise RuntimeError(f"{source}: {child_name} feature schema is invalid")
        if (
            child.get("name") != child_name
            or child.get("model_type") != expected_model_cls
            or child.get("problem_type") != "regression"
            or child.get("eval_metric") != "root_mean_squared_error"
            or child.get("stopping_metric") != "root_mean_squared_error"
            or child.get("is_valid") is not True
            or child.get("can_infer") is not True
            or child.get("num_cpus") != EXPECTED_CHILD_CPUS
            or child.get("num_gpus") != 0
            or child.get("val_in_fit") is not True
            or child.get("unlabeled_in_fit") is not False
            or hardened._exact_int(
                child.get("num_features"), f"{source}: {child_name} feature count"
            )
            != len(child_features)
            or set(child_features) != set(outer_features)
        ):
            raise RuntimeError(f"{source}: {child_name} fitted state changed")
        initial = _json_mapping(
            child.get("hyperparameters"), f"{source}: {child_name} params"
        )
        user = _json_mapping(
            child.get("hyperparameters_user"), f"{source}: {child_name} user params"
        )
        if initial != expected_child_hyperparameters(engine, child_fold) or user:
            raise RuntimeError(f"{source}: {child_name} initialized policy changed")
        child_ag_args = _as_mapping(
            child.get("ag_args_fit"), f"{source}: {child_name} ag args"
        )
        if any(
            child_ag_args.get(name) != value
            for name, value in expected_ag_args_fit.items()
        ):
            raise RuntimeError(f"{source}: {child_name} budget ratios changed")
        comparator_fit = _validate_comparator_fit(
            child.get("comparator_fit"),
            arm=arm,
            field=f"{source}: {child_name} comparator fit",
        )
        representation = _validate_representation(
            child.get("benchmark_representation"),
            arm=arm,
            dataset=dataset,
            field=f"{source}: {child_name} representation",
            child_features=child_features,
        )
        refit_params = _json_mapping(
            child.get("hyperparameters_fit"),
            f"{source}: {child_name} refit params",
        )
        child_rows.append(
            {
                "lane": lane,
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
                "arm": arm,
                "arm_code": spec["code"],
                "engine": engine,
                "child": child_name,
                "child_fold": child_fold,
                "child_features": list(child_features),
                "representation": representation,
                "initial_hyperparameters": initial,
                "user_hyperparameters": user,
                "effective_hyperparameters": dict(comparator_fit["resolved_params"]),
                "comparator_fit": comparator_fit,
                "refit_params": refit_params,
                "num_cpus": EXPECTED_CHILD_CPUS,
                "num_gpus": 0,
                "source": source,
            }
        )

    outer = {
        "lane": lane,
        "dataset": dataset,
        "task_id": TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "registered_fold": 3 * repeat + fold,
        "arm": arm,
        "arm_code": spec["code"],
        "engine": engine,
        "representation": spec["representation"],
        **metrics,
        "peak_memory_bytes": peak_memory,
        "baseline_memory_bytes": baseline_memory,
        "incremental_memory_bytes": incremental_memory,
        "framework": str(record["framework"]),
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_cpus_child": num_cpus_child,
        "num_gpus_child": num_gpus_child,
        "source": source,
    }
    return outer, child_rows


def _decode_result_pickle(path: Path) -> Mapping[str, Any]:
    # Only the runner may decode its own trusted cache.  The analyzer never does.
    return followon._decode_result_pickle(path)


def expected_result_relative_path(
    lane: str, dataset: str, repeat: int, fold: int, arm: str
) -> str:
    coordinate = (lane, dataset, repeat, fold)
    if (
        arm not in ARM_SPECS
        or ARM_SPECS[arm]["lane"] != lane
        or coordinate not in set(expected_coordinates(lane))
    ):
        raise RuntimeError("result coordinate is outside the frozen comparator")
    return str(
        Path("experiments")
        / "data"
        / _experiment_name(arm)
        / str(TASKS[dataset])
        / f"{repeat}_{fold}"
        / "results.pkl"
    )


def _result_path(output_dir: Path, job: Any) -> Path:
    lane, dataset, repeat, fold = _job_coordinate(job)
    return output_dir / expected_result_relative_path(
        lane, dataset, repeat, fold, _job_arm(job)
    )


def _cached_result_issue(path: Path, job: Any) -> str | None:
    try:
        metadata_record = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(metadata_record.st_mode):
        return "not_a_regular_file"
    try:
        outer, children = parse_result_record(
            _decode_result_pickle(path), source=str(path)
        )
        coordinate = (
            outer["lane"],
            outer["dataset"],
            outer["repeat"],
            outer["fold"],
        )
        if (
            coordinate != _job_coordinate(job)
            or outer["arm"] != _job_arm(job)
            or len(children) != 8
        ):
            return "mismatched"
    except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
        return "incomplete_or_mismatched"
    return None


def _validate_resume_history(value: Any, output_dir: Path) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("resume history must be a nonempty list")
    statuses = {
        "valid",
        "missing",
        "not_a_regular_file",
        "unreadable",
        "mismatched",
        "incomplete_or_mismatched",
    }
    allowed_artifacts = {
        COMPLETION_ATTESTATION_FILENAME,
        ANALYSIS_PAYLOAD_FILENAME,
        *DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
    }
    for index, raw in enumerate(value):
        item = _as_mapping(raw, f"resume history[{index}]")
        required = {
            "resumed_at_utc",
            "pid",
            "invalidated_coordinate_count",
            "invalidated_result_count",
            "invalidated_coordinates",
            "archived_campaign_artifacts",
        }
        if set(item) != required:
            raise RuntimeError("resume history fields are incomplete")
        coordinates = item["invalidated_coordinates"]
        artifacts = item["archived_campaign_artifacts"]
        if not isinstance(coordinates, list) or not isinstance(artifacts, list):
            raise RuntimeError("resume history entries must be lists")
        if hardened._exact_int(
            item["invalidated_coordinate_count"], "invalidated coordinate count"
        ) != len(coordinates):
            raise RuntimeError("resume coordinate count changed")
        archived_count = 0
        seen = set()
        archive_roots = set()

        def validate_archived(relative: Any, *, campaign_artifact: bool) -> None:
            nonlocal archived_count
            if not isinstance(relative, str):
                raise RuntimeError("archived path must be a string")
            lexical = Path(relative)
            if (
                lexical.is_absolute()
                or ".." in lexical.parts
                or len(lexical.parts) < 3
                or lexical.parts[0] != "resume_invalidated"
                or not lexical.parts[1]
            ):
                raise RuntimeError("archived path escapes the resume root")
            path = output_dir / lexical
            try:
                record = path.lstat()
            except OSError as exc:
                raise RuntimeError("archived path is missing") from exc
            if not stat.S_ISREG(record.st_mode):
                raise RuntimeError("archived path is not a regular file")
            try:
                path.resolve().relative_to(output_dir.resolve())
            except ValueError as exc:
                raise RuntimeError("archived path escapes campaign output") from exc
            archive_roots.add(tuple(lexical.parts[:2]))
            if campaign_artifact:
                if len(lexical.parts) != 3 or lexical.name not in allowed_artifacts:
                    raise RuntimeError("archived campaign artifact is not allowed")
            else:
                if lexical.name != "results.pkl" or "experiments" not in lexical.parts:
                    raise RuntimeError("archived result path is not a result")
                archived_count += 1

        for raw_coordinate in coordinates:
            coordinate = _as_mapping(raw_coordinate, "invalidated coordinate")
            if set(coordinate) != {
                "lane",
                "dataset",
                "repeat",
                "fold",
                "arm_status",
                "archived",
            }:
                raise RuntimeError("invalidated coordinate fields are incomplete")
            key = (
                coordinate["lane"],
                coordinate["dataset"],
                hardened._exact_int(coordinate["repeat"], "resume repeat"),
                hardened._exact_int(coordinate["fold"], "resume fold"),
            )
            if key not in set(expected_coordinates()) or key in seen:
                raise RuntimeError("invalidated coordinate is outside the campaign")
            seen.add(key)
            arm_status = _as_mapping(
                coordinate["arm_status"], "invalidated arm status"
            )
            lane_arms = {
                arm for arm, spec in ARM_SPECS.items() if spec["lane"] == key[0]
            }
            if (
                set(arm_status) != lane_arms
                or any(status not in statuses for status in arm_status.values())
                or all(status == "valid" for status in arm_status.values())
                or not isinstance(coordinate["archived"], list)
            ):
                raise RuntimeError("invalidated triad status is inconsistent")
            for relative in coordinate["archived"]:
                validate_archived(relative, campaign_artifact=False)
        for relative in artifacts:
            validate_archived(relative, campaign_artifact=True)
        if hardened._exact_int(
            item["invalidated_result_count"], "invalidated result count"
        ) != archived_count or len(archive_roots) > 1:
            raise RuntimeError("resume archive totals or roots are inconsistent")


def prepare_grouped_resume(
    output_dir: Path, jobs: Iterable[Any], *, resume: bool
) -> dict[str, Any] | None:
    """Invalidate a complete lane-specific D/M/C triad if one member is bad."""
    if not resume:
        return None
    grouped: dict[
        tuple[str, str, int, int], dict[str, tuple[Path, Any]]
    ] = defaultdict(dict)
    expected_paths = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        followon._require_no_symlink_components(
            path, root=output_dir, field="expected cached comparator result"
        )
        expected_paths.add(str(path.relative_to(output_dir)))
        grouped[_job_coordinate(job)][_job_arm(job)] = (path, job)
    observed = followon._observed_regular_result_paths(output_dir)
    unexpected = set(observed).difference(expected_paths)
    if unexpected:
        raise RuntimeError(f"resume cache contains unexpected results: {unexpected}")

    stale_paths = [
        output_dir / COMPLETION_ATTESTATION_FILENAME,
        output_dir / ANALYSIS_PAYLOAD_FILENAME,
        *(output_dir / name for name in DEFAULT_ANALYSIS_OUTPUT_FILENAMES),
    ]
    for group in grouped.values():
        for path, _ in group.values():
            hardened._require_regular_archive_source(path, "cached comparator result")
    for path in (
        *stale_paths,
        output_dir / WARMUP_HISTORY_FILENAME,
        output_dir / RESUME_HISTORY_FILENAME,
        output_dir / MANIFEST_FILENAME,
    ):
        hardened._require_regular_archive_source(path, "comparator campaign artifact")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = output_dir / "resume_invalidated" / timestamp
    invalidated: list[dict[str, Any]] = []
    invalidated_result_count = 0
    for coordinate in expected_coordinates():
        group = grouped[coordinate]
        lane_arms = {
            arm for arm, spec in ARM_SPECS.items() if spec["lane"] == coordinate[0]
        }
        if set(group) != lane_arms:
            raise RuntimeError(f"resume triad is incomplete for {coordinate}")
        status_by_arm = {
            arm: _cached_result_issue(path, job)
            for arm, (path, job) in group.items()
        }
        existing = [
            path
            for path, _ in group.values()
            if path.exists() or path.is_symlink()
        ]
        if not existing or all(issue is None for issue in status_by_arm.values()):
            continue
        archived = []
        for source in existing:
            relative = source.relative_to(output_dir)
            destination = archive_root / relative
            followon._prepare_archive_destination(destination, output_dir=output_dir)
            os.replace(source, destination)
            archived.append(str(destination.relative_to(output_dir)))
        invalidated_result_count += len(archived)
        invalidated.append(
            {
                "lane": coordinate[0],
                "dataset": coordinate[1],
                "repeat": coordinate[2],
                "fold": coordinate[3],
                "arm_status": {
                    arm: issue or "valid"
                    for arm, issue in sorted(status_by_arm.items())
                },
                "archived": archived,
            }
        )
    archived_campaign_artifacts = []
    for source in stale_paths:
        if not source.exists():
            continue
        destination = archive_root / source.name
        followon._prepare_archive_destination(destination, output_dir=output_dir)
        os.replace(source, destination)
        archived_campaign_artifacts.append(str(destination.relative_to(output_dir)))
    record = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "invalidated_coordinate_count": len(invalidated),
        "invalidated_result_count": invalidated_result_count,
        "invalidated_coordinates": invalidated,
        "archived_campaign_artifacts": archived_campaign_artifacts,
    }
    history_path = output_dir / RESUME_HISTORY_FILENAME
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read comparator resume history") from exc
        _validate_resume_history(history, output_dir)
    history.append(record)
    hardened._atomic_write_json(history_path, history)
    _validate_resume_history(history, output_dir)
    return record


def _stable_file_artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    return followon._stable_file_artifact(path, output_dir)


def collect_result_artifacts(
    output_dir: Path, jobs: Iterable[Any]
) -> dict[str, dict[str, Any]]:
    expected = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        followon._require_no_symlink_components(
            path, root=output_dir, field="expected completed comparator result"
        )
        expected.add(str(path.relative_to(output_dir)))
    observed = followon._observed_regular_result_paths(output_dir)
    if set(observed) != expected or len(observed) != EXPECTED_JOBS:
        raise RuntimeError(
            f"completed result grid mismatch: expected {EXPECTED_JOBS}, "
            f"found {len(observed)}"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for path in (observed[relative] for relative in sorted(observed)):
        artifact = _stable_file_artifact(path, output_dir)
        relative = artifact.pop("path")
        artifacts[relative] = artifact
    return artifacts


def validate_cross_engine_representations(
    child_rows: Iterable[Mapping[str, Any]],
) -> int:
    """Prove each D/M/C child received the identical representation."""
    rows = list(child_rows)
    index = {
        (
            row["lane"],
            row["dataset"],
            int(row["repeat"]),
            int(row["fold"]),
            row["engine"],
            row["child"],
        ): row
        for row in rows
    }
    if len(index) != len(rows):
        raise RuntimeError("comparator child metadata contains duplicate rows")
    comparisons = 0
    for lane, dataset, repeat, fold in expected_coordinates():
        for child_index in range(1, 9):
            child = f"S1F{child_index}"
            engine_rows = [
                index.get((lane, dataset, repeat, fold, engine, child))
                for engine in ENGINE_SPECS
            ]
            if any(row is None for row in engine_rows):
                raise RuntimeError("cross-engine child block is incomplete")
            reference = engine_rows[0]
            if any(
                row["child_features"] != reference["child_features"]
                or row["representation"] != reference["representation"]
                for row in engine_rows[1:]
            ):
                raise RuntimeError(
                    "D/M/C child representation or feature schema is not identical"
                )
            expected_kind = (
                "native" if lane == PRIMARY_LANE else "safe_ordinal"
            )
            if reference["representation"].get("kind") != expected_kind:
                raise RuntimeError("cross-engine representation kind changed")
            comparisons += 1
    expected = EXPECTED_COORDINATES * 8
    if comparisons != expected:
        raise RuntimeError("cross-engine child comparison count changed")
    return comparisons


def validate_completed_results(
    output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    seen = set()
    resources = set()
    stop_reasons: Counter[str] = Counter()
    inferred_stop_reasons: Counter[str] = Counter()
    deadline_hits = 0
    for relative in sorted(artifacts):
        path = output_dir / relative
        outer, children = parse_result_record(
            _decode_result_pickle(path), source=relative
        )
        key = (
            outer["lane"],
            outer["dataset"],
            outer["repeat"],
            outer["fold"],
            outer["arm"],
        )
        expected_relative = expected_result_relative_path(*key[:4], key[4])
        if relative != expected_relative:
            raise RuntimeError("result payload is bound to the wrong frozen path")
        if key in seen:
            raise RuntimeError(f"duplicate completed comparator result: {key}")
        seen.add(key)
        resources.add(
            (
                outer["num_cpus"],
                outer["num_gpus"],
                outer["num_cpus_child"],
                outer["num_gpus_child"],
            )
        )
        outer_rows.append(outer)
        child_rows.extend(children)
        for child in children:
            fit = child["comparator_fit"]
            reason = fit.get("stop_reason")
            stop_reasons[str(reason) if reason is not None else "unknown"] += 1
            inferred = fit.get("stop_reason_inferred")
            if inferred is not None:
                inferred_stop_reasons[
                    "inferred" if inferred is True else "unresolved"
                ] += 1
            deadline_hits += int(fit.get("deadline_hit") is True)
    if seen != expected_grid():
        raise RuntimeError("completed comparator result grid is incomplete")
    if len(child_rows) != EXPECTED_CHILD_FITS:
        raise RuntimeError(
            f"expected {EXPECTED_CHILD_FITS} child fits, got {len(child_rows)}"
        )
    if resources != {(EXPECTED_CHILD_CPUS, 0, EXPECTED_CHILD_CPUS, 0)}:
        raise RuntimeError("comparator results do not share frozen resources")
    if stop_reasons.get("time_limit", 0) or deadline_hits:
        raise RuntimeError("comparator campaign contains a known deadline stop")
    representation_blocks = validate_cross_engine_representations(child_rows)

    order_rank = {key: index for index, key in enumerate(expected_ordered_grid())}
    outer_rows.sort(
        key=lambda row: order_rank[
            (
                row["lane"],
                row["dataset"],
                row["repeat"],
                row["fold"],
                row["arm"],
            )
        ]
    )
    child_rows.sort(
        key=lambda row: (
            order_rank[
                (
                    row["lane"],
                    row["dataset"],
                    row["repeat"],
                    row["fold"],
                    row["arm"],
                )
            ],
            row["child_fold"],
        )
    )
    observed_order = [
        (
            row["lane"],
            row["dataset"],
            row["repeat"],
            row["fold"],
            row["arm"],
        )
        for row in outer_rows
    ]
    if observed_order != expected_ordered_grid():
        raise RuntimeError("normalized outer rows lost frozen execution order")
    lane_result_counts = Counter(row["lane"] for row in outer_rows)
    lane_child_counts = Counter(row["lane"] for row in child_rows)
    if lane_result_counts != {
        PRIMARY_LANE: EXPECTED_PRIMARY_JOBS,
        ORDINAL_DIAGNOSTIC_LANE: EXPECTED_DIAGNOSTIC_JOBS,
    } or lane_child_counts != {
        PRIMARY_LANE: EXPECTED_PRIMARY_JOBS * 8,
        ORDINAL_DIAGNOSTIC_LANE: EXPECTED_DIAGNOSTIC_JOBS * 8,
    }:
        raise RuntimeError("normalized lane counts changed")
    validation = {
        "result_count": len(outer_rows),
        "child_fit_count": len(child_rows),
        "lane_result_counts": dict(sorted(lane_result_counts.items())),
        "lane_child_counts": dict(sorted(lane_child_counts.items())),
        "cross_engine_representation_blocks": representation_blocks,
        "failure_count": 0,
        "imputation_count": 0,
        "known_deadline_hit_count": deadline_hits,
        "known_time_limit_stop_count": int(stop_reasons.get("time_limit", 0)),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "competitor_stop_reason_inference_counts": dict(
            sorted(inferred_stop_reasons.items())
        ),
        "job_order_sha256": job_order_sha256(),
        "resource_allocation": {
            "num_cpus": EXPECTED_CHILD_CPUS,
            "num_gpus": 0,
            "num_cpus_child": EXPECTED_CHILD_CPUS,
            "num_gpus_child": 0,
        },
        "memory_metric": "peak_mem_cpu_minus_min_mem_cpu",
    }
    return validation, outer_rows, child_rows


def _history_artifact(
    output_dir: Path,
    filename: str,
    *,
    required: bool,
    validator,
) -> dict[str, Any] | None:
    path = output_dir / filename
    if not path.exists():
        if required:
            raise RuntimeError(f"required campaign history is missing: {filename}")
        return None
    hardened._require_regular_archive_source(path, "campaign history")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"campaign history is invalid: {filename}") from exc
    validator(value)
    return _stable_file_artifact(path, output_dir)


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jobs: Iterable[Any],
    result_count: int,
) -> dict[str, Any]:
    jobs = list(jobs)
    if result_count != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} completed results, got {result_count}"
        )
    if ordering_audit(jobs) != manifest.get("ordering_audit"):
        raise RuntimeError("completion job order does not match the manifest")
    artifacts = collect_result_artifacts(output_dir, jobs)
    validation, outer_rows, child_rows = validate_completed_results(
        output_dir, artifacts
    )
    payload = {
        "schema_version": 1,
        "kind": PAYLOAD_KIND,
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "result_artifacts_sha256": hashlib.sha256(
            hardened._canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    payload_path = output_dir / ANALYSIS_PAYLOAD_FILENAME
    hardened._atomic_write_json(payload_path, payload)
    payload_artifact = _stable_file_artifact(payload_path, output_dir)

    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_comparator_warmup import validate_comparator_warmup_history

    warmup_artifact = _history_artifact(
        output_dir,
        WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: validate_comparator_warmup_history(
            value,
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=os.getpid(),
        ),
    )
    resume_artifact = _history_artifact(
        output_dir,
        RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: _validate_resume_history(value, output_dir),
    )
    if collect_result_artifacts(output_dir, jobs) != artifacts:
        raise RuntimeError("result artifacts changed during normalization")
    final_source = collect_source_provenance(
        output_dir=output_dir,
        chimeraboost_path=Path(manifest["source"]["chimeraboost"]["repository"]),
    )
    final_runtime = collect_runtime_provenance()
    if final_source != manifest.get("source"):
        raise RuntimeError("source provenance changed during the campaign")
    if final_runtime != manifest.get("runtime"):
        raise RuntimeError("runtime provenance changed during the campaign")
    manifest_path = output_dir / MANIFEST_FILENAME
    attestation = {
        "schema_version": 1,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "result_count": result_count,
        "expected_result_count": EXPECTED_JOBS,
        "expected_primary_result_count": EXPECTED_PRIMARY_JOBS,
        "expected_ordinal_diagnostic_result_count": EXPECTED_DIAGNOSTIC_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "warmup_thread_count": EXPECTED_CHILD_CPUS,
        "warmup_stage_count": len(WARMUP_STAGE_NAMES),
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "git_head": manifest["source"]["git_head"],
        "manifest_sha256": hardened._sha256_file(manifest_path),
        "result_artifacts": artifacts,
        "analysis_payload_artifact": payload_artifact,
        "warmup_history_artifact": warmup_artifact,
        "resume_history_artifact": resume_artifact,
        "validation": validation,
    }
    hardened._atomic_write_json(
        output_dir / COMPLETION_ATTESTATION_FILENAME, attestation
    )
    return attestation


def _run_campaign(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    context: Any,
    jobs: Iterable[Any],
    ordering: Mapping[str, Any],
    child_cpus: int,
    chimera_source: Mapping[str, Any],
) -> int:
    """Execute, normalize, and attest the already-audited frozen grid."""
    jobs = list(jobs)
    source = collect_source_provenance(
        output_dir=output_dir,
        chimeraboost_path=args.chimeraboost_path,
    )
    if source["chimeraboost"] != dict(chimera_source):
        raise RuntimeError("ChimeraBoost source changed after schedule construction")
    manifest = build_run_manifest(
        output_dir=output_dir,
        source=source,
        ordering=ordering,
        resolved_child_num_cpus=child_cpus,
    )
    manifest = write_or_validate_run_manifest(
        output_dir, manifest, resume=args.resume
    )
    prepare_grouped_resume(output_dir, jobs, resume=args.resume)

    try:
        from benchmarks.tabarena_comparator_warmup import (
            STAGE_NAMES as COMPARATOR_WARMUP_STAGE_NAMES,
            warmup_tabarena_comparators,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_comparator_warmup import (
            STAGE_NAMES as COMPARATOR_WARMUP_STAGE_NAMES,
            warmup_tabarena_comparators,
        )

    if tuple(COMPARATOR_WARMUP_STAGE_NAMES) != WARMUP_STAGE_NAMES:
        raise RuntimeError("runner and comparator warmup stage contracts diverged")
    warmup = warmup_tabarena_comparators(thread_count=child_cpus)
    hardened.record_warmup(output_dir, warmup)
    primary_jobs, diagnostic_jobs = partition_jobs_for_dispatch(jobs)
    results: list[dict[str, Any]] = []
    dispatches = (
        (PRIMARY_LANE, primary_jobs, EXPECTED_PRIMARY_JOBS),
        (
            ORDINAL_DIAGNOSTIC_LANE,
            diagnostic_jobs,
            EXPECTED_DIAGNOSTIC_JOBS,
        ),
    )
    for lane, lane_jobs, expected_count in dispatches:
        lane_results = context.run_jobs(
            lane_jobs,
            expname=str(output_dir / "experiments"),
            register=False,
            new_result_prefix="[same-machine D/M/C] ",
            debug_mode=True,
        )
        if len(lane_results) != expected_count:
            raise RuntimeError(
                f"{lane} dispatch returned {len(lane_results)} results; "
                f"expected {expected_count}"
            )
        results.extend(lane_results)
    write_completion_attestation(
        output_dir,
        manifest=manifest,
        jobs=jobs,
        result_count=len(results),
    )
    print(f"SAME_MACHINE_COMPARISON_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
