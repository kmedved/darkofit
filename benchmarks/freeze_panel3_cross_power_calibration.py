#!/usr/bin/env python3
"""Create the immutable source freeze for Panel 3 power calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks import run_tabarena_regression_followon_screen as spent  # noqa: E402
from benchmarks.campaign_lib import provenance  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_source_freeze.json"
)
PROTOCOL = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_protocol.md"
)
RUNNER = ROOT / "benchmarks" / "run_panel3_cross_power_calibration.py"
ANALYZER = ROOT / "benchmarks" / "analyze_panel3_cross_power_calibration.py"
CANDIDATE_CONTRACT = ROOT / "benchmarks" / "panel3_candidate_contract.json"
RUNTIME_CONTRACT = ROOT / "benchmarks" / "panel3_environment_contract.json"
FREEZE_RELATIVE = "benchmarks/panel3_cross_power_calibration_source_freeze.json"

TASKS = dict(spent.TASKS)
COORDINATES = tuple(
    {"repeat": repeat, "fold": fold, "sample": 0}
    for repeat, fold in spent.SCREEN_SPLITS
)
EXPECTED_NATIVE_CATEGORICAL_COLUMNS = {
    name: list(columns)
    for name, columns in spent.EXPECTED_NATIVE_CATEGORICAL_COLUMNS.items()
}
ARMS = (
    "current_default",
    "t5_composite_policy",
    "guarded_cross_features_policy",
)

EXPLICIT_SOURCE_PATHS = (
    "pyproject.toml",
    "benchmarks/panel3_candidate_contract.json",
    "benchmarks/panel3_environment_contract.json",
    "benchmarks/panel3_cross_power_calibration_protocol.md",
    "benchmarks/panel3_power_design_contract.json",
    "benchmarks/panel3_power_design_protocol.md",
    "benchmarks/build_panel3_power_design.py",
    "benchmarks/panel3_registry_declarations.json",
    "benchmarks/freeze_panel3_cross_power_calibration.py",
    "benchmarks/run_panel3_cross_power_calibration.py",
    "benchmarks/analyze_panel3_cross_power_calibration.py",
    "benchmarks/run_panel3_confirmation.py",
    "benchmarks/analyze_panel3_confirmation.py",
    "benchmarks/run_t5_composite_confirmation.py",
    "benchmarks/run_smooth_cross_features.py",
    "benchmarks/basketball_harness.py",
    "benchmarks/basketball_guardrails.py",
    "benchmarks/run_basketball_creator_benchmark.py",
    "benchmarks/build_ctr23_contamination_registry.py",
    "benchmarks/panel3_data_contract.py",
    "benchmarks/panel3_registry_common.py",
    "benchmarks/run_tabarena_regression_followon_screen.py",
    "benchmarks/run_tabarena_regression_cap_horizon.py",
    "benchmarks/tabarena_adapter.py",
    "benchmarks/tabarena_screen_adapters.py",
    "benchmarks/tabarena_followon_warmup.py",
    "benchmarks/tabarena_warmup.py",
    "tests/conftest.py",
    "tests/test_campaign_partition.py",
    "tests/test_panel3_cross_power_calibration.py",
    "tests/test_panel3_execution.py",
    "tests/test_panel3_power_design.py",
    "tests/test_panel3_registry.py",
)
PROSPECTIVE_PANEL3_SOURCE_PATHS = tuple(
    str(path.relative_to(ROOT)) for path in common.PANEL3_SOURCE_PATHS
)
SPENT_PROVENANCE_PATHS = (
    "benchmarks/tabarena_regression_accuracy_shootout_protocol.md",
    "benchmarks/tabarena_regression_accuracy_shootout_run_manifest.json",
    "benchmarks/tabarena_regression_same_machine_run_manifest.json",
    "benchmarks/tabarena_regression_accuracy_shootout_completion_attestation.json",
)
_CANDIDATE_CONTRACT_KEYS = {
    "schema_version",
    "contract_name",
    "random_state",
    "thread_count",
    "runtime",
    "ordinal_features",
    "inner_validation",
    "control",
    "candidates",
    "comparators",
    "decision",
}
_RUNTIME_CONTRACT_KEYS = {
    "schema_version",
    "contract_name",
    "contract_kind",
    "python_implementation",
    "python_version",
    "packages",
}
_RUNTIME_PACKAGE_NAMES = {
    "catboost",
    "darkofit",
    "numba",
    "numpy",
    "openml",
    "pandas",
    "scikit-learn",
    "scipy",
}


def _git(*arguments: str) -> str:
    return provenance.git_output(ROOT, *arguments)


def _sha256(path: Path) -> str:
    return provenance.file_sha256(path)


def _blob(path: str, head: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{head}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def source_paths() -> tuple[str, ...]:
    package = tuple(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "darkofit").rglob("*.py"))
        if path.is_file()
    )
    paths = tuple(
        dict.fromkeys(
            (
                *EXPLICIT_SOURCE_PATHS,
                *PROSPECTIVE_PANEL3_SOURCE_PATHS,
                *package,
            )
        )
    )
    missing = [relative for relative in paths if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"calibration source is missing: {missing}")
    return paths


def source_file_sha256() -> dict[str, str]:
    return {
        relative: _sha256(ROOT / relative)
        for relative in source_paths()
    }


def source_file_sha256_at_head(
    head: str,
    paths: Sequence[str] | None = None,
) -> dict[str, str]:
    return {
        relative: hashlib.sha256(_blob(relative, head)).hexdigest()
        for relative in (source_paths() if paths is None else paths)
    }


def source_tree_sha256(files: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _decode_contract_snapshots(
    snapshots: dict[Path, bytes],
    files: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode the exact contract bytes already captured in the source freeze."""
    candidate = common.decode_json_bytes(
        snapshots[CANDIDATE_CONTRACT.absolute()],
        source=CANDIDATE_CONTRACT,
    )
    runtime_environment = common.decode_json_bytes(
        snapshots[RUNTIME_CONTRACT.absolute()],
        source=RUNTIME_CONTRACT,
    )
    runtime_digest = files[str(RUNTIME_CONTRACT.relative_to(ROOT))]
    runtime_reference = (
        candidate.get("runtime") if isinstance(candidate, dict) else None
    )
    if (
        not isinstance(candidate, dict)
        or set(candidate) != _CANDIDATE_CONTRACT_KEYS
        or candidate.get("schema_version") != 1
        or candidate.get("contract_name")
        != "darkofit_panel3_dual_candidate_contract_v1"
        or not isinstance(runtime_reference, dict)
        or set(runtime_reference) != {"path", "sha256"}
        or runtime_reference.get("path")
        != str(RUNTIME_CONTRACT.relative_to(ROOT))
        or runtime_reference.get("sha256") != runtime_digest
    ):
        raise RuntimeError("Panel 3 candidate contract changed")
    packages = (
        runtime_environment.get("packages")
        if isinstance(runtime_environment, dict)
        else None
    )
    if (
        not isinstance(runtime_environment, dict)
        or set(runtime_environment) != _RUNTIME_CONTRACT_KEYS
        or runtime_environment.get("schema_version") != 1
        or runtime_environment.get("contract_name")
        != "darkofit_panel3_exact_runtime_environment_v1"
        or runtime_environment.get("contract_kind")
        != "exact_active_environment_versions_v1"
        or runtime_environment.get("python_implementation") != "cpython"
        or not isinstance(runtime_environment.get("python_version"), str)
        or not runtime_environment["python_version"]
        or not isinstance(packages, dict)
        or set(packages) != _RUNTIME_PACKAGE_NAMES
        or any(
            not isinstance(value, str) or not value
            for value in packages.values()
        )
    ):
        raise RuntimeError("Panel 3 runtime environment changed")
    return candidate, runtime_environment


def _array_sha256(value: Any, dtype: str) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def ordered_task_view_sha256(X: Any, y: Any) -> str:
    """Hash exact row/column order in addition to the lineage fingerprint."""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(X)
    target = pd.Series(y, name="__target__")
    if len(frame) != len(target):
        raise RuntimeError("calibration ordered-view row counts differ")
    digest = hashlib.sha256()
    for position in range(frame.shape[1]):
        values = pd.util.hash_pandas_object(
            frame.iloc[:, position],
            index=True,
            categorize=False,
        ).to_numpy(dtype="<u8", copy=False)
        digest.update(position.to_bytes(8, "big"))
        digest.update(np.ascontiguousarray(values).tobytes())
    target_hashes = pd.util.hash_pandas_object(
        target,
        index=True,
        categorize=False,
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(b"__target__")
    digest.update(np.ascontiguousarray(target_hashes).tobytes())
    return digest.hexdigest()


def task_view_attestations() -> dict[str, Any]:
    """Bind the exact spent task views and official coordinates without fits."""
    import numpy as np
    import openml
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    from benchmarks import build_ctr23_contamination_registry as fingerprints
    from benchmarks import panel3_data_contract as data_contract

    attestations = {}
    for dataset_name, task_id in TASKS.items():
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
        X, y, declared, names = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
            include_row_id=False,
            include_ignore_attribute=False,
        )
        if (
            not isinstance(X, pd.DataFrame)
            or str(dataset.name) != dataset_name
            or list(X.columns) != list(names)
            or len(declared) != X.shape[1]
        ):
            raise RuntimeError(
                f"calibration source task {task_id} feature view changed"
            )
        target = pd.to_numeric(y, errors="raise").astype(np.float64)
        if (
            len(target) != len(X)
            or not np.isfinite(target.to_numpy(dtype=np.float64)).all()
        ):
            raise RuntimeError(
                f"calibration source task {task_id} target is invalid"
            )
        categorical_indices = [
            index
            for index, (flag, dtype) in enumerate(
                zip(declared, X.dtypes, strict=True)
            )
            if bool(flag) or not is_numeric_dtype(dtype)
        ]
        categorical_names = [
            str(X.columns[index]) for index in categorical_indices
        ]
        if (
            categorical_names
            != EXPECTED_NATIVE_CATEGORICAL_COLUMNS[dataset_name]
        ):
            raise RuntimeError(
                f"calibration source task {task_id} categoricals changed"
            )
        coordinates = []
        for coordinate in COORDINATES:
            train, test = task.get_train_test_split_indices(**coordinate)
            train = np.asarray(train, dtype=np.int64)
            test = np.asarray(test, dtype=np.int64)
            if (
                train.ndim != 1
                or test.ndim != 1
                or len(train) == 0
                or len(test) == 0
                or np.any(train < 0)
                or np.any(test < 0)
                or np.any(train >= len(X))
                or np.any(test >= len(X))
                or len(np.unique(train)) != len(train)
                or len(np.unique(test)) != len(test)
                or np.intersect1d(train, test).size
                or len(train) + len(test) != len(X)
            ):
                raise RuntimeError(
                    f"calibration source task {task_id} split is invalid"
                )
            coordinates.append(
                {
                    **coordinate,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_index_sha256": _array_sha256(train, "<i8"),
                    "test_index_sha256": _array_sha256(test, "<i8"),
                }
            )
        fingerprint = fingerprints.dataset_fingerprint(X, target)
        if (
            not isinstance(fingerprint, dict)
            or fingerprint.get("n_rows") != len(X)
            or fingerprint.get("n_features") != X.shape[1]
        ):
            raise RuntimeError(
                f"calibration source task {task_id} fingerprint is invalid"
            )
        attestations[str(task_id)] = {
            "task_id": task_id,
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": str(dataset.name),
            "target_name": str(task.target_name),
            "openml_declared_md5": str(dataset.md5_checksum),
            "split_dimensions": list(task.get_split_dimensions()),
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "feature_names": [str(value) for value in X.columns],
            "feature_schema": data_contract.feature_schema(X),
            "feature_schema_sha256": data_contract.feature_schema_sha256(X),
            "categorical_feature_indices": categorical_indices,
            "categorical_feature_names": categorical_names,
            "ordered_task_view_sha256": ordered_task_view_sha256(
                X,
                target,
            ),
            "dataset_fingerprint": fingerprint,
            "coordinates": coordinates,
        }
    if len(attestations) != 13:
        raise RuntimeError("calibration task-view ledger is incomplete")
    return attestations


def build() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "calibration source freeze requires a clean committed tree"
        )
    source_head = _git("rev-parse", "HEAD")
    source_files = tuple(ROOT / relative for relative in source_paths())
    source_snapshots, files = common.secure_snapshot_files(source_files)
    spent_files = tuple(
        ROOT / relative for relative in SPENT_PROVENANCE_PATHS
    )
    _spent_snapshots, spent = common.secure_snapshot_files(spent_files)
    committed_files = source_file_sha256_at_head(
        source_head,
        tuple(files),
    )
    committed_spent = {
        relative: hashlib.sha256(_blob(relative, source_head)).hexdigest()
        for relative in SPENT_PROVENANCE_PATHS
    }
    if files != committed_files or spent != committed_spent:
        raise RuntimeError(
            "calibration source snapshot differs from committed H1"
        )
    _candidate, runtime_environment = _decode_contract_snapshots(
        source_snapshots,
        files,
    )
    task_views = task_view_attestations()
    if (
        _git("status", "--porcelain")
        or _git("rev-parse", "HEAD") != source_head
    ):
        raise RuntimeError(
            "calibration source changed during the source freeze"
        )
    common.recheck_snapshot_files(source_files, files)
    common.recheck_snapshot_files(spent_files, spent)
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_cross_power_calibration_source_freeze_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_head": source_head,
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_head_clean": True,
            "post_freeze_allowed_tracked_path": FREEZE_RELATIVE,
            "source_file_sha256": files,
            "source_file_set_sha256": source_tree_sha256(files),
            "protocol_sha256": files[str(PROTOCOL.relative_to(ROOT))],
            "runner_sha256": files[str(RUNNER.relative_to(ROOT))],
            "analyzer_sha256": files[str(ANALYZER.relative_to(ROOT))],
            "candidate_contract_sha256": files[
                str(CANDIDATE_CONTRACT.relative_to(ROOT))
            ],
            "runtime_contract": {
                "path": str(RUNTIME_CONTRACT.relative_to(ROOT)),
                "sha256": files[str(RUNTIME_CONTRACT.relative_to(ROOT))],
            },
            "runtime_environment": runtime_environment,
            "spent_provenance_sha256": spent,
            "tasks": TASKS,
            "native_categorical_columns": (
                EXPECTED_NATIVE_CATEGORICAL_COLUMNS
            ),
            "task_view_attestations": task_views,
            "coordinates": list(COORDINATES),
            "arms": list(ARMS),
            "coordinate_count": len(TASKS) * len(COORDINATES),
            "result_count": len(TASKS) * len(COORDINATES) * len(ARMS),
            "outcome_blind_source_freeze": True,
            "candidate_or_control_models_fitted": False,
            "candidate_or_control_outcomes_inspected": False,
            "development_only": True,
            "panel3_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "source_freeze_sha256",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.expanduser().absolute() != DEFAULT_OUTPUT:
        raise RuntimeError("calibration source-freeze output path changed")
    common.validate_create_path(args.output)
    artifact = build()
    encoded = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    source_files = tuple(
        ROOT / relative for relative in artifact["source_file_sha256"]
    )
    spent_files = tuple(
        ROOT / relative for relative in artifact["spent_provenance_sha256"]
    )
    if (
        _git("status", "--porcelain")
        or _git("rev-parse", "HEAD") != artifact["source_head"]
    ):
        raise RuntimeError(
            "calibration source changed before freeze publication"
        )
    common.recheck_snapshot_files(
        source_files,
        artifact["source_file_sha256"],
    )
    common.recheck_snapshot_files(
        spent_files,
        artifact["spent_provenance_sha256"],
    )
    common.atomic_create(args.output, encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_head": artifact["source_head"],
                "source_freeze_sha256": artifact["source_freeze_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
