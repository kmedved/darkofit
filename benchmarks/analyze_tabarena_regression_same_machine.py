"""Verify and analyze the frozen same-machine comparator campaign.

Only the campaign runner may decode TabArena ``results.pkl`` files.  This
module consumes the runner's attested, normalized JSON and verifies every raw
result as an opaque byte artifact.  It never imports pickle directly and never
calls a raw-result decoder; transitive campaign-library imports are allowed.

The 13-dataset native/default comparison is the primary panel.  The separate
Airfoil/Diamonds safe-ordinal lane is diagnostic only: it is analyzed both
across engines and against the matching native result within each engine, but
is never pooled into the primary estimand and cannot advance a product policy.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks import analyze_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_tabarena_regression_same_machine as campaign
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_cap_horizon as hardened
    import run_tabarena_regression_same_machine as campaign


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_719

# The insertion order is part of the frozen estimand and execution order.
TASKS: dict[str, int] = {
    "airfoil_self_noise": 363612,
    "Another-Dataset-on-used-Fiat-500": 363615,
    "concrete_compressive_strength": 363625,
    "diamonds": 363631,
    "Food_Delivery_Time": 363672,
    "healthcare_insurance_expenses": 363675,
    "houses": 363678,
    "miami_housing": 363686,
    "physiochemical_protein": 363693,
    "QSAR-TID-11": 363697,
    "QSAR_fish_toxicity": 363698,
    "superconductivity": 363705,
    "wine_quality": 363708,
}
DIAGNOSTIC_DATASETS = ("airfoil_self_noise", "diamonds")
EXPECTED_ORDINAL_SCHEMA_SHA256 = {
    "airfoil_self_noise": "c592ba78a13e5434afb8980820a0c0ab668db594fb2491950286c66a3fe071a1",
    "diamonds": "98bc5774a472b47a6c7fd9fbf14fa7ca784877e7896d1cef3c43c41673f21cf6",
}
COORDINATES = ((0, 0), (1, 1), (2, 2))
ORDER_CYCLE = (
    ("D", "M", "C"),
    ("M", "C", "D"),
    ("C", "D", "M"),
    ("C", "M", "D"),
    ("D", "C", "M"),
    ("M", "D", "C"),
)

PRIMARY_ARMS: dict[str, dict[str, str]] = {
    "D": {
        "arm": "darkofit_product_default",
        "engine": "darkofit",
        "representation": "native",
    },
    "M": {
        "arm": "chimeraboost_0_14_1_default",
        "engine": "chimeraboost",
        "representation": "native",
    },
    "C": {
        "arm": "catboost_1_2_10_default",
        "engine": "catboost",
        "representation": "native",
    },
}
DIAGNOSTIC_ARMS: dict[str, dict[str, str]] = {
    code: {
        **spec,
        "arm": spec["arm"] + "_safe_ordinal",
        "representation": "safe_ordinal",
    }
    for code, spec in PRIMARY_ARMS.items()
}
LANE_ARMS = {
    "primary": PRIMARY_ARMS,
    "ordinal_diagnostic": DIAGNOSTIC_ARMS,
}
ARM_TO_CODE = {
    spec["arm"]: code
    for arms in LANE_ARMS.values()
    for code, spec in arms.items()
}
ARM_TO_SPEC = {
    spec["arm"]: {**spec, "code": code, "lane": lane}
    for lane, arms in LANE_ARMS.items()
    for code, spec in arms.items()
}

PAIRWISE_CODES = (("D", "M"), ("D", "C"), ("M", "C"))
METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "incremental_memory_bytes",
    "peak_memory_bytes",
)
PRIMARY_METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "incremental_memory_bytes",
)
ACCURACY_METRICS = ("test_rmse", "val_rmse")
OUTPUT_KEYS = (
    "primary_split_csv",
    "primary_dataset_csv",
    "diagnostic_split_csv",
    "diagnostic_dataset_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
OUTPUT_NAMES = (
    "primary_paired_splits.csv",
    "primary_per_dataset.csv",
    "ordinal_diagnostic_paired_splits.csv",
    "ordinal_diagnostic_per_dataset.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
VALID_STOP_REASONS = {
    "iteration_limit",
    "early_stopping",
    "no_split",
    "time_limit",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _decode_json(payload: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not finite valid JSON") from exc
    return dict(hardened._as_mapping(value, field))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{field} must be a SHA-256 digest") from exc
    return value


def _read_json_stable(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file: {path}")
    payload = hardened._read_stable(path, field)
    return _decode_json(payload, field), payload


def _finite_json(value: Any, field: str) -> Any:
    """Reject non-JSON values, nonfinite floats, and non-string object keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{field} contains a nonfinite float")
        return value
    if isinstance(value, list):
        return [_finite_json(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{field} contains a non-string key")
            clean[key] = _finite_json(item, f"{field}.{key}")
        return clean
    raise RuntimeError(f"{field} contains a non-JSON value: {type(value).__name__}")


def _positive(value: Any, field: str) -> float:
    return hardened._positive_finite(value, field)


def _nonnegative(value: Any, field: str) -> float:
    return hardened._nonnegative_finite(value, field)


def _exact_int(value: Any, field: str) -> int:
    return hardened._exact_int(value, field)


def primary_coordinates() -> list[tuple[str, int, int]]:
    return [(dataset, repeat, fold) for dataset in TASKS for repeat, fold in COORDINATES]


def diagnostic_coordinates() -> list[tuple[str, int, int]]:
    return [
        (dataset, repeat, fold)
        for dataset in DIAGNOSTIC_DATASETS
        for repeat, fold in COORDINATES
    ]


def expected_ordered_grid() -> list[tuple[str, str, int, int, str]]:
    """Reconstruct the frozen 135-job order without trusting safe payload rows."""
    ordered: list[tuple[str, str, int, int, str]] = []
    for index, (dataset, repeat, fold) in enumerate(primary_coordinates()):
        for code in ORDER_CYCLE[index % len(ORDER_CYCLE)]:
            ordered.append(("primary", dataset, repeat, fold, PRIMARY_ARMS[code]["arm"]))
    for index, (dataset, repeat, fold) in enumerate(diagnostic_coordinates()):
        for code in ORDER_CYCLE[index % len(ORDER_CYCLE)]:
            ordered.append(
                (
                    "ordinal_diagnostic",
                    dataset,
                    repeat,
                    fold,
                    DIAGNOSTIC_ARMS[code]["arm"],
                )
            )
    if len(ordered) != 135 or len(set(ordered)) != 135:
        raise RuntimeError("frozen same-machine ordered grid is not exact")
    return ordered


def job_order_sha256() -> str:
    payload = [
        {
            "lane": lane,
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
            "arm_code": ARM_TO_CODE[arm],
        }
        for lane, dataset, repeat, fold, arm in expected_ordered_grid()
    ]
    return _sha256(_canonical_json(payload))


def expected_position_audit() -> dict[str, Any]:
    """Return a deterministic per-lane position audit for the six-order cycle."""
    result: dict[str, Any] = {}
    cursor = 0
    for lane, coordinates in (
        ("primary", primary_coordinates()),
        ("ordinal_diagnostic", diagnostic_coordinates()),
    ):
        counts = {engine: [0, 0, 0] for engine in ("darkofit", "chimeraboost", "catboost")}
        for coordinate in coordinates:
            group = expected_ordered_grid()[cursor : cursor + 3]
            cursor += 3
            if any(row[:4] != (lane, *coordinate) for row in group):
                raise RuntimeError("same-machine order is not coordinate-local")
            for position, row in enumerate(group):
                counts[ARM_TO_SPEC[row[4]]["engine"]][position] += 1
        result[lane] = {
            engine: {"first": values[0], "second": values[1], "third": values[2]}
            for engine, values in counts.items()
        }
    if cursor != 135:
        raise RuntimeError("same-machine position audit did not consume the grid")
    return {"job_order_sha256": job_order_sha256(), "lane_position_counts": result}


def contrast_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for lane, panel in (
        ("primary", "primary"),
        ("ordinal_diagnostic", "ordinal_pairwise"),
    ):
        for numerator_code, denominator_code in PAIRWISE_CODES:
            arms = LANE_ARMS[lane]
            suffix = "" if lane == "primary" else "_ordinal"
            specs.append(
                {
                    "contrast": (
                        f"{arms[numerator_code]['engine']}_vs_"
                        f"{arms[denominator_code]['engine']}{suffix}"
                    ),
                    "code": (
                        f"{numerator_code}/{denominator_code}"
                        if lane == "primary"
                        else f"{numerator_code}ord/{denominator_code}ord"
                    ),
                    "panel": panel,
                    "numerator_lane": lane,
                    "denominator_lane": lane,
                    "numerator": arms[numerator_code]["arm"],
                    "denominator": arms[denominator_code]["arm"],
                }
            )
    for code in ("D", "M", "C"):
        specs.append(
            {
                "contrast": f"{PRIMARY_ARMS[code]['engine']}_ordinal_vs_native",
                "code": f"{code}ord/{code}native",
                "panel": "ordinal_uplift",
                "numerator_lane": "ordinal_diagnostic",
                "denominator_lane": "primary",
                "numerator": DIAGNOSTIC_ARMS[code]["arm"],
                "denominator": PRIMARY_ARMS[code]["arm"],
            }
        )
    if len(specs) != 9 or len({item["code"] for item in specs}) != 9:
        raise RuntimeError("same-machine contrast declarations are not exact")
    return specs


def _artifact_bytes(
    input_dir: Path,
    relative: str,
    metadata: Mapping[str, Any],
    field: str,
) -> bytes:
    if set(metadata) != {"sha256", "size_bytes"}:
        raise RuntimeError(f"{field} attestation fields are not exact")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
        raise RuntimeError(f"unsafe attested path: {relative!r}")
    cursor = input_dir
    for component in relative_path.parts:
        cursor = cursor / component
        try:
            component_metadata = cursor.lstat()
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field}: {cursor}") from exc
        if stat.S_ISLNK(component_metadata.st_mode):
            raise RuntimeError(f"{field} must not contain symbolic-link components")
    try:
        raw_metadata = cursor.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {cursor}") from exc
    if not stat.S_ISREG(raw_metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(input_dir.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"attested path escapes campaign: {relative}") from exc
    payload = hardened._read_stable(resolved, field)
    expected_size = _exact_int(metadata.get("size_bytes"), f"{field} size")
    expected_digest = _validate_digest(metadata.get("sha256"), f"{field} digest")
    if len(payload) != expected_size or _sha256(payload) != expected_digest:
        raise RuntimeError(f"{field} does not match its attestation")
    return payload


def _verify_repository_source(
    source: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1].resolve(strict=True)
    base_fields = {
        "repository",
        "git_head",
        "git_tree",
        "relevant_status",
        "files",
        "darkofit_import",
        "tabarena",
        "chimeraboost",
        "catboost",
        "external_adapter_sources",
    }
    if set(source) != base_fields or source.get("relevant_status") != "":
        raise RuntimeError("same-machine source provenance is incomplete or dirty")
    recorded_repository = hardened._manifest_path(
        source.get("repository"), "recorded same-machine repository"
    )
    if recorded_repository != repository:
        raise RuntimeError("executing analyzer repository does not match the run")
    files = hardened._as_mapping(source.get("files"), "source files")
    expected_files = {str(path) for path in campaign.SOURCE_FILES}
    if set(files) != expected_files:
        raise RuntimeError("same-machine source file set is not exact")
    for relative in campaign.SOURCE_FILES:
        key = str(relative)
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RuntimeError(f"source file escapes repository: {relative}") from exc
        metadata = hardened._as_mapping(files[key], f"source metadata for {key}")
        if set(metadata) != {"sha256", "git_blob"}:
            raise RuntimeError(f"source metadata is incomplete for {key}")
        payload = hardened._read_stable(path, f"source {key}")
        if metadata.get("sha256") != _sha256(payload):
            raise RuntimeError(f"source SHA-256 mismatch for {key}")
        if metadata.get("git_blob") != hardened._git_hash_payload(repository, payload, key):
            raise RuntimeError(f"source Git-blob mismatch for {key}")
    head = hardened._git_output(repository, ["rev-parse", "HEAD"], "Git HEAD")
    tree = hardened._git_output(repository, ["rev-parse", "HEAD^{tree}"], "Git tree")
    if source.get("git_head") != head or source.get("git_tree") != tree:
        raise RuntimeError("executing Git revision does not match the run")
    changes = hardened._repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError(
            "executing same-machine repository has dirty or unrecorded code: "
            + ", ".join(changes)
        )
    return {
        "executing_repository": str(repository),
        "executing_git_head": head,
        "executing_git_tree": tree,
    }


def verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    source = hardened._as_mapping(manifest.get("source"), "manifest source")
    diagnostics = _verify_repository_source(source, input_dir)
    repository = Path(__file__).resolve().parents[1]
    hardened._verify_dependency_provenance(
        source.get("darkofit_import"),
        "darkofit",
        input_dir,
        required_repository=repository,
    )
    hardened._verify_dependency_provenance(source.get("tabarena"), "tabarena", input_dir)
    _verify_chimeraboost_source(source.get("chimeraboost"), input_dir)
    _verify_catboost_source(source.get("catboost"))
    _verify_external_adapter_sources(source.get("external_adapter_sources"))
    _verify_runtime_provenance(manifest.get("runtime"))
    return {
        **diagnostics,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
        "hardware_identity_verified": True,
    }


def _verify_chimeraboost_source(value: Any, input_dir: Path) -> None:
    recorded = dict(hardened._as_mapping(value, "ChimeraBoost source"))
    fields = {
        "repository",
        "git_head",
        "git_tree",
        "git_tag",
        "git_remote_origin",
        "status",
        "module_file",
        "module_sha256",
        "hidden_import_warmup",
    }
    if set(recorded) != fields or recorded.get("status") != "":
        raise RuntimeError("ChimeraBoost source provenance is not exact and clean")
    repository = hardened._manifest_path(
        recorded.get("repository"), "ChimeraBoost repository"
    ).resolve(strict=True)
    changes = hardened._repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError("ChimeraBoost checkout changed after the campaign")
    head = hardened._git_output(repository, ["rev-parse", "HEAD"], "ChimeraBoost HEAD")
    tree = hardened._git_output(
        repository, ["rev-parse", "HEAD^{tree}"], "ChimeraBoost tree"
    )
    tags = hardened._git_output(
        repository, ["tag", "--points-at", "HEAD"], "ChimeraBoost tag"
    ).splitlines()
    remote = hardened._sanitize_git_remote(
        hardened._git_output(
            repository,
            ["remote", "get-url", "origin"],
            "ChimeraBoost origin",
        )
    )
    module_path = hardened._manifest_path(
        recorded.get("module_file"), "ChimeraBoost module file"
    ).resolve(strict=True)
    try:
        module_path.relative_to(repository)
    except ValueError as exc:
        raise RuntimeError("ChimeraBoost module escapes its checkout") from exc
    if (
        head != campaign.CHIMERABOOST_TAG_COMMIT
        or recorded.get("git_head") != head
        or recorded.get("git_tree") != tree
        or recorded.get("git_tag") != "v0.14.1"
        or recorded.get("git_remote_origin") != remote
        or "v0.14.1" not in tags
        or recorded.get("hidden_import_warmup") != "disabled"
        or recorded.get("module_sha256")
        != _sha256(hardened._read_stable(module_path, "ChimeraBoost module"))
    ):
        raise RuntimeError("ChimeraBoost v0.14.1 provenance changed")


def _verify_file_artifact(value: Any, field: str) -> Path:
    artifact = dict(hardened._as_mapping(value, field))
    required = {"path", "sha256", "size_bytes"}
    optional = {"module"}
    if not required.issubset(artifact) or not set(artifact).issubset(required | optional):
        raise RuntimeError(f"{field} fields are not exact")
    path = hardened._manifest_path(artifact.get("path"), field).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{field} is not a safe regular file")
    payload = hardened._read_stable(path, field)
    if (
        len(payload) != _exact_int(artifact.get("size_bytes"), f"{field} size")
        or _sha256(payload) != _validate_digest(artifact.get("sha256"), f"{field} digest")
    ):
        raise RuntimeError(f"{field} bytes changed")
    return path


def _verify_installed_distribution(
    value: Any,
    *,
    distribution_name: str,
    module_names: tuple[str, ...],
    expected_version: str | None = None,
) -> None:
    field = f"{distribution_name} source"
    recorded = dict(hardened._as_mapping(value, field))
    expected_fields = {
        "distribution",
        "version",
        "modules",
        "files",
        "record_integrity",
    }
    if set(recorded) != expected_fields:
        raise RuntimeError(f"{distribution_name} provenance fields are not exact")
    distribution = importlib_metadata.distribution(distribution_name)
    if (
        recorded.get("distribution") != distribution_name
        or recorded.get("version") != distribution.version
        or (
            expected_version is not None
            and distribution.version != expected_version
        )
    ):
        raise RuntimeError(f"{distribution_name} distribution identity changed")
    distribution_files = distribution.files
    if not distribution_files:
        raise RuntimeError(f"{distribution_name} has no installed file manifest")
    expected: dict[str, dict[str, Any]] = {}
    installed_paths: dict[Path, str] = {}
    record_verified_count = 0
    unhashed_record_paths: list[str] = []
    for relative in sorted(distribution_files, key=str):
        text = str(relative)
        if not text or text in expected:
            raise RuntimeError(
                f"{distribution_name} installed file manifest is not unique"
            )
        raw_path = Path(distribution.locate_file(relative))
        if raw_path.is_symlink() or not raw_path.is_file():
            raise RuntimeError(f"unsafe {distribution_name} source file: {raw_path}")
        path = raw_path.resolve(strict=True)
        if path in installed_paths:
            raise RuntimeError(
                f"{distribution_name} file manifest resolves one file twice"
            )
        payload = hardened._read_stable(path, f"{distribution_name} file {text}")
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
        expected[text] = {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "record_sha256": record_sha256,
            "record_size_bytes": record_size_bytes,
        }
        installed_paths[path] = text
    if unhashed_record_paths != [
        text for text in expected if Path(text).name == "RECORD"
    ]:
        raise RuntimeError(f"{distribution_name} RECORD exception set is not exact")
    if recorded.get("files") != expected:
        raise RuntimeError(f"{distribution_name} installed bytes changed")
    expected_integrity = {
        "algorithm": "sha256",
        "verified_file_count": record_verified_count,
        "unhashed_record_paths": unhashed_record_paths,
    }
    if recorded.get("record_integrity") != expected_integrity:
        raise RuntimeError(f"{distribution_name} RECORD attestation changed")
    if not module_names or len(set(module_names)) != len(module_names):
        raise RuntimeError(f"{distribution_name} module identity set is invalid")
    modules = dict(hardened._as_mapping(recorded.get("modules"), field + " modules"))
    if set(modules) != set(module_names):
        raise RuntimeError(f"{distribution_name} module identity set changed")
    for module_name in module_names:
        module = importlib.import_module(module_name)
        raw_module_path = getattr(module, "__file__", None)
        if not raw_module_path:
            raise RuntimeError(f"imported module has no file identity: {module_name}")
        module_path = Path(raw_module_path).resolve(strict=True)
        expected_module = {
            "path": str(module_path),
            "distribution_path": installed_paths.get(module_path),
        }
        if module_path not in installed_paths or modules[module_name] != expected_module:
            raise RuntimeError(f"imported {module_name} module changed")
    version_module = importlib.import_module(module_names[0])
    if expected_version is not None and (
        getattr(version_module, "__version__", None) != expected_version
    ):
        raise RuntimeError(
            f"imported {module_names[0]} version does not match its distribution"
        )


def _verify_catboost_source(value: Any) -> None:
    _verify_installed_distribution(
        value,
        distribution_name="catboost",
        module_names=campaign.CATBOOST_PROVENANCE_MODULES,
        expected_version=campaign.CATBOOST_VERSION,
    )


def _verify_external_adapter_sources(value: Any) -> None:
    recorded = dict(hardened._as_mapping(value, "external adapter sources"))
    if set(recorded) != {
        "autogluon_distributions",
        "tabarena_chimeraboost_model",
    }:
        raise RuntimeError("external adapter source set is not exact")
    autogluon = dict(
        hardened._as_mapping(
            recorded["autogluon_distributions"], "AutoGluon distributions"
        )
    )
    if set(autogluon) != set(campaign.AUTOGLUON_PROVENANCE_MODULES):
        raise RuntimeError("AutoGluon distribution source set is not exact")
    for distribution_name, module_names in (
        campaign.AUTOGLUON_PROVENANCE_MODULES.items()
    ):
        _verify_installed_distribution(
            autogluon[distribution_name],
            distribution_name=distribution_name,
            module_names=module_names,
        )
    name = "tabarena_chimeraboost_model"
    module_name = "tabarena.models.chimeraboost.model"
    artifact = dict(hardened._as_mapping(recorded[name], f"external source {name}"))
    path = _verify_file_artifact(artifact, f"external source {name}")
    module = importlib.import_module(module_name)
    if artifact.get("module") != module_name or path != Path(module.__file__).resolve():
        raise RuntimeError(f"external source module changed: {name}")


def _verify_runtime_provenance(value: Any) -> None:
    recorded = dict(hardened._as_mapping(value, "run manifest runtime"))
    packages: dict[str, str | None] = {}
    for distribution in campaign.PACKAGE_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    packages["chimeraboost"] = campaign.CHIMERABOOST_VERSION
    current = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "environment": {
            key: os.environ.get(key) for key in campaign.RUNTIME_ENVIRONMENT_KEYS
        },
        "hardware": hardened.collect_runtime_hardware_provenance(),
    }
    if recorded != current:
        raise RuntimeError("analysis runtime/hardware does not match the campaign")


def _validate_ordering_metadata(
    manifest: Mapping[str, Any],
    *,
    attestation: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    digest = job_order_sha256()
    if manifest.get("job_order_sha256") != digest:
        raise RuntimeError("manifest job-order digest does not match frozen order")
    if manifest.get("ordering_audit") != expected_position_audit():
        raise RuntimeError("manifest position audit does not match frozen order")
    for field, value in (("attestation", attestation), ("analysis payload", payload)):
        if value is not None and value.get("job_order_sha256") != digest:
            raise RuntimeError(f"{field} job-order digest does not match")
    return digest


def _expected_framework(arm: str) -> str:
    spec = ARM_TO_SPEC.get(arm)
    if spec is None:
        raise RuntimeError(f"unknown same-machine arm: {arm}")
    display = {
        "darkofit": "DarkoFit",
        "chimeraboost": "ChimeraBoost",
        "catboost": "CatBoost",
    }[spec["engine"]]
    return f"{display}_c1_same_machine_{spec['lane']}_{spec['code']}_BAG_L1"


def _validate_result_source_binding(
    source: str,
    *,
    framework: str,
    task_id: int,
    repeat: int,
    fold: int,
) -> None:
    expected = (
        Path("experiments")
        / "data"
        / framework
        / str(task_id)
        / f"{repeat}_{fold}"
        / "results.pkl"
    ).as_posix()
    if source != expected:
        raise RuntimeError(f"result source does not match its job: {source!r}")


def _validate_outer_rows(
    rows: Any,
    artifacts: Mapping[str, Any],
    child_cpus: int,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, int, str], dict[str, Any]],
]:
    if not isinstance(rows, list) or len(rows) != 135:
        raise RuntimeError("analysis payload outer result count is wrong")
    fields = {
        "lane",
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "engine",
        "representation",
        "test_rmse",
        "val_rmse",
        "train_time_s",
        "infer_time_s",
        "incremental_memory_bytes",
        "baseline_memory_bytes",
        "peak_memory_bytes",
        "framework",
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
        "source",
    }
    expected_order = expected_ordered_grid()
    normalized: list[dict[str, Any]] = []
    index: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for position, raw in enumerate(rows):
        row = dict(hardened._as_mapping(raw, f"outer_rows[{position}]"))
        if set(row) != fields:
            raise RuntimeError(
                f"outer_rows[{position}] fields are not exact: "
                f"missing={sorted(fields - set(row))}, extra={sorted(set(row) - fields)}"
            )
        lane = row["lane"]
        dataset = row["dataset"]
        arm = row["arm"]
        repeat = _exact_int(row["repeat"], "outer repeat")
        fold = _exact_int(row["fold"], "outer fold")
        registered = _exact_int(row["registered_fold"], "outer registered fold")
        task_id = _exact_int(row["task_id"], "outer task id")
        key = (lane, dataset, repeat, fold, arm)
        if key != expected_order[position]:
            raise RuntimeError("safe outer rows are not in frozen execution order")
        spec = ARM_TO_SPEC.get(arm)
        if (
            spec is None
            or dataset not in TASKS
            or task_id != TASKS[dataset]
            or (repeat, fold) not in COORDINATES
            or registered != 3 * repeat + fold
            or lane != spec["lane"]
            or row["arm_code"] != spec["code"]
            or row["engine"] != spec["engine"]
            or row["representation"] != spec["representation"]
            or key in index
        ):
            raise RuntimeError(f"safe outer row does not match frozen grid: {key}")
        framework = _expected_framework(arm)
        if row["framework"] != framework:
            raise RuntimeError(f"outer framework changed for {key}")
        if lane == "ordinal_diagnostic" and dataset not in DIAGNOSTIC_DATASETS:
            raise RuntimeError(f"diagnostic row escaped its dataset scope: {key}")
        num_cpus = _exact_int(row["num_cpus"], "outer CPUs")
        num_gpus = _exact_int(row["num_gpus"], "outer GPUs")
        num_cpus_child = _exact_int(row["num_cpus_child"], "outer child CPUs")
        num_gpus_child = _exact_int(row["num_gpus_child"], "outer child GPUs")
        if (
            num_cpus != child_cpus
            or num_cpus_child != child_cpus
            or num_gpus != 0
            or num_gpus_child != 0
        ):
            raise RuntimeError(f"outer resource allocation changed for {key}")
        source = row["source"]
        if not isinstance(source, str) or source not in artifacts:
            raise RuntimeError(f"outer source is not attested for {key}")
        _validate_result_source_binding(
            source,
            framework=framework,
            task_id=task_id,
            repeat=repeat,
            fold=fold,
        )
        source_counts[source] += 1
        clean = {
            **row,
            "task_id": task_id,
            "repeat": repeat,
            "fold": fold,
            "registered_fold": registered,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "num_cpus_child": num_cpus_child,
            "num_gpus_child": num_gpus_child,
        }
        for metric in METRICS:
            if metric == "incremental_memory_bytes":
                continue
            clean[metric] = _positive(row[metric], f"outer {metric}")
        clean["incremental_memory_bytes"] = _nonnegative(
            row["incremental_memory_bytes"], "outer incremental memory"
        )
        clean["baseline_memory_bytes"] = _nonnegative(
            row["baseline_memory_bytes"], "outer baseline memory"
        )
        expected_increment = clean["peak_memory_bytes"] - clean["baseline_memory_bytes"]
        if expected_increment < 0.0 or not math.isclose(
            clean["incremental_memory_bytes"],
            expected_increment,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"outer incremental-memory audit is inconsistent for {key}")
        normalized.append(clean)
        index[key] = clean
    if set(source_counts) != set(artifacts) or any(
        count != 1 for count in source_counts.values()
    ):
        raise RuntimeError("safe outer rows do not bind one-to-one to raw results")
    return normalized, index


def _feature_schema_sha256(columns: Any, field: str) -> str:
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise RuntimeError(f"{field} must contain unique nonempty strings")
    return _sha256(
        json.dumps(columns, allow_nan=False, separators=(",", ":")).encode("utf-8")
    )


def _validate_representation(
    value: Any,
    *,
    lane: str,
    dataset: str,
    child_features: list[str],
    field: str,
) -> dict[str, Any]:
    representation = dict(hardened._as_mapping(_finite_json(value, field), field))
    external_digest = _feature_schema_sha256(child_features, f"{field} features")
    if lane == "primary":
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
        input_count = _exact_int(representation["input_feature_count"], f"{field} input count")
        output_count = _exact_int(
            representation["output_feature_count"], f"{field} output count"
        )
        categoricals = representation["categorical_input_columns"]
        fitted_categoricals = representation["fitted_categorical_input_columns"]
        dropped = representation["dropped_constant_input_columns"]
        dropped_counts = representation["dropped_constant_input_unique_counts"]
        if any(
            not isinstance(items, list)
            or any(not isinstance(item, str) for item in items)
            or len(set(items)) != len(items)
            for items in (categoricals, fitted_categoricals, dropped)
        ):
            raise RuntimeError(f"{field} native column audit is invalid")
        if (
            representation["schema_version"] != 2
            or representation["kind"] != "native"
            or representation["fit_scope"] != "comparator_child_training_fold"
            or representation["feature_alignment_policy"]
            != "autogluon_child_drop_unique"
            or input_count != len(child_features)
            or output_count != len(child_features) - len(dropped)
            or representation["external_feature_schema_sha256"] != external_digest
            or any(column not in child_features for column in categoricals + dropped)
            or categoricals
            != [column for column in child_features if column in set(categoricals)]
            or dropped
            != [column for column in child_features if column in set(dropped)]
            or any(column not in categoricals for column in fitted_categoricals)
            or fitted_categoricals
            != [column for column in categoricals if column not in set(dropped)]
            or len(dropped_counts) != len(dropped)
            or any(_exact_int(count, f"{field} dropped count") != 1 for count in dropped_counts)
        ):
            raise RuntimeError(f"{field} native representation audit is inconsistent")
        fitted_features = [column for column in child_features if column not in set(dropped)]
        if representation["fitted_feature_schema_sha256"] != _feature_schema_sha256(
            fitted_features, f"{field} fitted features"
        ):
            raise RuntimeError(f"{field} fitted schema digest does not match")
        expected_target_use = bool(fitted_categoricals)
        if representation["target_used_by_representation"] is not expected_target_use:
            raise RuntimeError(f"{field} native target-use audit is inconsistent")
        return representation

    expected_fields = {
        "kind",
        "domain",
        "mapping_source",
        "fit_scope",
        "target_used_by_representation",
        "fit_calls",
        "eval_transform_calls_during_fit",
        "eval_unknown_counts",
        "input_feature_count",
        "output_feature_count",
        "categorical_input_positions",
        "observed_training_category_counts",
        "compact_category_domains",
        "category_schema_sha256",
        "missing_policy",
        "unknown_policy",
        "remaining_native_target_stat_positions",
    }
    if set(representation) != expected_fields:
        raise RuntimeError(f"{field} safe-ordinal representation fields are not exact")
    expected_domain = {
        "airfoil_self_noise": "airfoil_attack_angle_numeric",
        "diamonds": "diamonds_declared_orders",
    }.get(dataset)
    positions = representation["categorical_input_positions"]
    observed = representation["observed_training_category_counts"]
    unknown = representation["eval_unknown_counts"]
    domains = representation["compact_category_domains"]
    expected_ordinal = {
        "airfoil_self_noise": {
            "features": [
                "frequency",
                "chord-length",
                "free-stream-velocity",
                "suction-side-displacement-thickness",
                "attack-angle",
            ],
            "positions": [4],
            "observed_max": [27],
            "domains": {"attack-angle": list(range(27))},
        },
        "diamonds": {
            "features": [
                "carat",
                "depth",
                "table",
                "x",
                "y",
                "z",
                "cut",
                "color",
                "clarity",
            ],
            "positions": [6, 7, 8],
            "observed_max": [5, 7, 8],
            "domains": {
                "cut": list(range(5)),
                "color": list(range(7)),
                "clarity": list(range(8)),
            },
        },
    }.get(dataset)
    if (
        expected_domain is None
        or expected_ordinal is None
        or child_features != expected_ordinal["features"]
        or representation["kind"] != "safe_ordinal"
        or representation["domain"] != expected_domain
        or representation["mapping_source"] != "source_frozen_before_campaign"
        or representation["fit_scope"] != "child_training_rows_only"
        or representation["target_used_by_representation"] is not False
        or representation["fit_calls"] != 1
        or representation["eval_transform_calls_during_fit"] != 1
        or unknown != [0]
        or representation["input_feature_count"] != len(child_features)
        or representation["output_feature_count"] != len(child_features)
        or not isinstance(positions, list)
        or not positions
        or any(_exact_int(item, f"{field} ordinal position") not in range(len(child_features)) for item in positions)
        or len(set(positions)) != len(positions)
        or positions != expected_ordinal["positions"]
        or not isinstance(observed, list)
        or len(observed) != len(positions)
        or any(_exact_int(item, f"{field} observed categories") < 1 for item in observed)
        or any(
            _exact_int(value, f"{field} observed categories") > maximum
            for value, maximum in zip(observed, expected_ordinal["observed_max"])
        )
        or not isinstance(domains, Mapping)
        or not domains
        or domains != expected_ordinal["domains"]
        or representation["missing_policy"] != "numeric_nan"
        or representation["unknown_policy"] != "fail_closed"
        or representation["remaining_native_target_stat_positions"] != []
    ):
        raise RuntimeError(f"{field} safe-ordinal representation audit is inconsistent")
    if (
        _validate_digest(
            representation["category_schema_sha256"], f"{field} schema digest"
        )
        != EXPECTED_ORDINAL_SCHEMA_SHA256[dataset]
    ):
        raise RuntimeError(f"{field} safe-ordinal schema digest changed")
    return representation


def _validate_common_comparator_fit(
    value: Any,
    *,
    engine: str,
    child_cpus: int,
    field: str,
) -> dict[str, Any]:
    fit = dict(hardened._as_mapping(_finite_json(value, field), field))
    common = {
        "schema_version",
        "engine",
        "iterations_requested",
        "best_iteration",
        "rounds_retained",
        "resolved_params",
        "num_cpus",
        "num_gpus",
        "resolved_learning_rate",
        "iterations_attempted",
        "stop_reason",
    }
    extras = {
        "darkofit": {
            "requested_tree_mode",
            "selected_tree_mode",
            "selected_lane",
            "rounds_completed",
            "wall_clock_limit_seconds",
            "wall_clock_safety_margin_seconds",
            "wall_clock_effective_seconds",
            "wall_clock_elapsed_seconds",
            "deadline_hit",
            "deadline_is_soft",
        },
        "chimeraboost": {
            "selected_lane",
            "linear_leaves_selected",
            "linear_selection_performed",
            "stop_reason_inferred",
        },
        "catboost": {
            "tree_count",
            "catboost_best_iteration_zero_based",
            "stop_reason_inferred",
        },
    }[engine]
    if set(fit) != common | extras:
        raise RuntimeError(
            f"{field} fields are not exact: missing={sorted((common | extras) - set(fit))}, "
            f"extra={sorted(set(fit) - (common | extras))}"
        )
    requested = _exact_int(fit["iterations_requested"], f"{field} requested")
    best = _exact_int(fit["best_iteration"], f"{field} best")
    retained = _exact_int(fit["rounds_retained"], f"{field} retained")
    attempted = _exact_int(fit["iterations_attempted"], f"{field} attempted")
    if (
        fit["schema_version"] != 1
        or fit["engine"] != engine
        or _exact_int(fit["num_cpus"], f"{field} CPUs") != child_cpus
        or float(fit["num_gpus"]) != 0.0
        or requested < 1
        or not (0 <= best <= requested)
        or not (0 <= retained <= requested)
        or not (0 <= attempted <= requested)
        or best != retained
    ):
        raise RuntimeError(f"{field} common telemetry is inconsistent")
    learning_rate = _positive(fit["resolved_learning_rate"], f"{field} learning rate")
    params = dict(hardened._as_mapping(fit["resolved_params"], f"{field} resolved params"))
    _finite_json(params, f"{field} resolved params")
    if _exact_int(params.get("thread_count"), f"{field} resolved thread count") != child_cpus:
        raise RuntimeError(f"{field} fitted engine thread count changed")
    reason = fit["stop_reason"]
    if reason is not None and reason not in VALID_STOP_REASONS | {"no_legal_split"}:
        raise RuntimeError(f"{field} stop reason is invalid")
    if reason == "time_limit":
        raise RuntimeError(f"{field} hit a forbidden time limit")

    if engine == "darkofit":
        completed = _exact_int(fit["rounds_completed"], f"{field} completed")
        if not (0 <= retained == best <= completed <= attempted <= requested):
            raise RuntimeError(f"{field} DarkoFit round counters are inconsistent")
        if (
            requested != 1_000
            or fit["requested_tree_mode"] != "catboost"
            or fit["selected_tree_mode"] != "catboost"
            or fit["selected_lane"] != "boosting"
            or fit["deadline_hit"] is not False
            or fit["deadline_is_soft"] is not True
        ):
            raise RuntimeError(f"{field} DarkoFit official default telemetry changed")
        wall_limit = _positive(fit["wall_clock_limit_seconds"], f"{field} wall limit")
        wall_margin = _nonnegative(
            fit["wall_clock_safety_margin_seconds"], f"{field} wall margin"
        )
        wall_effective = _nonnegative(
            fit["wall_clock_effective_seconds"], f"{field} effective wall limit"
        )
        _nonnegative(fit["wall_clock_elapsed_seconds"], f"{field} wall elapsed")
        if (
            wall_limit > 3_600.0
            or not math.isclose(wall_margin, min(5.0, 0.05 * wall_limit), abs_tol=1e-12)
            or not math.isclose(
                wall_effective, max(0.0, wall_limit - wall_margin), abs_tol=1e-12
            )
        ):
            raise RuntimeError(f"{field} DarkoFit wall-clock audit is inconsistent")
    elif engine == "chimeraboost":
        if (
            requested != 10_000
            or retained != best
            or fit["selected_lane"] not in {"constant", "linear"}
            or fit["linear_leaves_selected"]
            is not (fit["selected_lane"] == "linear")
            or not isinstance(fit["linear_selection_performed"], bool)
            or fit["stop_reason_inferred"] is not (reason is not None)
            or not math.isclose(learning_rate, 0.1, rel_tol=1e-7, abs_tol=1e-12)
        ):
            raise RuntimeError(f"{field} ChimeraBoost official telemetry changed")
    else:
        tree_count = _exact_int(fit["tree_count"], f"{field} tree count")
        raw_best = _exact_int(
            fit["catboost_best_iteration_zero_based"], f"{field} CatBoost best"
        )
        if (
            requested != 10_000
            or tree_count != retained
            or raw_best not in {-1, best - 1}
            or fit["stop_reason_inferred"] is not (reason is not None)
            or not math.isclose(learning_rate, 0.05, rel_tol=1e-7, abs_tol=1e-12)
            or params.get("task_type") != "CPU"
        ):
            raise RuntimeError(f"{field} CatBoost official telemetry changed")
    return fit


def _validate_child_config(
    row: Mapping[str, Any], *, engine: str, child_fold: int, field: str
) -> None:
    if row["user_hyperparameters"] != {}:
        raise RuntimeError(f"{field} must retain an empty manual configuration")
    initial = dict(
        hardened._as_mapping(row["initial_hyperparameters"], f"{field} initial params")
    )
    effective = dict(
        hardened._as_mapping(row["effective_hyperparameters"], f"{field} effective params")
    )
    _finite_json(initial, f"{field} initial params")
    _finite_json(effective, f"{field} effective params")
    expected_initial = {
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
    }[engine]
    expected_initial[
        "random_seed" if engine == "catboost" else "random_state"
    ] = child_fold
    if initial != expected_initial:
        raise RuntimeError(f"{field} official initialized defaults changed")
    if engine == "darkofit":
        required_effective = {
            "iterations": 1_000,
            "tree_mode": "catboost",
            "max_bins": 254,
            "ts_permutations": 1,
            "linear_residual": False,
            "random_state": child_fold,
        }
    elif engine == "chimeraboost":
        required_effective = {
            "n_estimators": 10_000,
            "depth": 6,
            "l2_leaf_reg": 1.0,
            "max_bins": 128,
            "cat_n_permutations": 4,
            "ordered_boosting": False,
            "min_child_weight": 1.0,
            "random_state": child_fold,
        }
    else:
        required_effective = {
            "iterations": 10_000,
            "eval_metric": "RMSE",
            "random_seed": child_fold,
        }
    if any(effective.get(name) != value for name, value in required_effective.items()):
        raise RuntimeError(f"{field} official effective defaults changed")
    if engine == "catboost" and not math.isclose(
        _positive(effective.get("learning_rate"), f"{field} effective learning rate"),
        0.05,
        rel_tol=1e-7,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{field} CatBoost effective learning rate changed")


def _validate_child_rows(
    rows: Any,
    outer_index: Mapping[tuple[str, str, int, int, str], Mapping[str, Any]],
    child_cpus: int,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 1_080:
        raise RuntimeError("analysis payload child result count is wrong")
    fields = {
        "lane",
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "engine",
        "child",
        "child_fold",
        "child_features",
        "representation",
        "initial_hyperparameters",
        "user_hyperparameters",
        "effective_hyperparameters",
        "comparator_fit",
        "refit_params",
        "num_cpus",
        "num_gpus",
        "source",
    }
    expected_order = [
        (*outer, child_fold)
        for outer in expected_ordered_grid()
        for child_fold in range(8)
    ]
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str, int]] = set()
    per_outer: Counter[tuple[str, str, int, int, str]] = Counter()
    for position, raw in enumerate(rows):
        row = dict(hardened._as_mapping(raw, f"child_rows[{position}]"))
        if set(row) != fields:
            raise RuntimeError(
                f"child_rows[{position}] fields are not exact: "
                f"missing={sorted(fields - set(row))}, extra={sorted(set(row) - fields)}"
            )
        lane = row["lane"]
        dataset = row["dataset"]
        arm = row["arm"]
        repeat = _exact_int(row["repeat"], "child repeat")
        fold = _exact_int(row["fold"], "child coordinate fold")
        registered = _exact_int(row["registered_fold"], "child registered fold")
        task_id = _exact_int(row["task_id"], "child task id")
        child_fold = _exact_int(row["child_fold"], "child fold")
        outer_key = (lane, dataset, repeat, fold, arm)
        key = (*outer_key, child_fold)
        spec = ARM_TO_SPEC.get(arm)
        outer = outer_index.get(outer_key)
        if key != expected_order[position]:
            raise RuntimeError("safe child rows are not in frozen execution order")
        if (
            outer is None
            or spec is None
            or key in seen
            or child_fold not in range(8)
            or row["child"] != f"S1F{child_fold + 1}"
            or row["arm_code"] != spec["code"]
            or row["engine"] != spec["engine"]
            or task_id != TASKS[dataset]
            or registered != 3 * repeat + fold
            or row["source"] != outer["source"]
            or _exact_int(row["num_cpus"], "child CPUs") != child_cpus
            or float(row["num_gpus"]) != 0.0
        ):
            raise RuntimeError(f"safe child row does not match outer row: {key}")
        child_features = row["child_features"]
        _feature_schema_sha256(child_features, f"child_rows[{position}].child_features")
        clean = {
            **row,
            "task_id": task_id,
            "repeat": repeat,
            "fold": fold,
            "registered_fold": registered,
            "child_fold": child_fold,
            "representation": _validate_representation(
                row["representation"],
                lane=lane,
                dataset=dataset,
                child_features=child_features,
                field=f"child_rows[{position}].representation",
            ),
            "comparator_fit": _validate_common_comparator_fit(
                row["comparator_fit"],
                engine=spec["engine"],
                child_cpus=child_cpus,
                field=f"child_rows[{position}].comparator_fit",
            ),
        }
        _validate_child_config(
            clean,
            engine=spec["engine"],
            child_fold=child_fold,
            field=f"child_rows[{position}]",
        )
        _finite_json(clean["refit_params"], f"child_rows[{position}].refit_params")
        normalized.append(clean)
        seen.add(key)
        per_outer[outer_key] += 1
    expected_children = {
        (*outer, child_fold)
        for outer in expected_ordered_grid()
        for child_fold in range(8)
    }
    if seen != expected_children or any(per_outer[outer] != 8 for outer in expected_ordered_grid()):
        raise RuntimeError("safe child-fit grid is not exact")
    _validate_cross_arm_schemas(normalized)
    return normalized


def _validate_cross_arm_schemas(child_rows: Sequence[Mapping[str, Any]]) -> None:
    by_block: dict[
        tuple[str, str, int, int, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in child_rows:
        key = (
            str(row["lane"]),
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["child_fold"]),
        )
        engine = str(row["engine"])
        if engine in by_block[key]:
            raise RuntimeError(f"duplicate engine in child schema block: {key}")
        by_block[key][engine] = row
    if len(by_block) != 135 * 8 // 3:
        raise RuntimeError("child schema block count is wrong")
    for key, engines in by_block.items():
        if set(engines) != {"darkofit", "chimeraboost", "catboost"}:
            raise RuntimeError(f"child schema block is incomplete: {key}")
        schemas = [engines[name]["child_features"] for name in sorted(engines)]
        if schemas[1:] != schemas[:-1]:
            raise RuntimeError(f"external child schema differs across engines: {key}")
        representations = [
            engines[name]["representation"] for name in sorted(engines)
        ]
        if representations[1:] != representations[:-1]:
            raise RuntimeError(f"representation audit differs across engines: {key}")

    native = {
        (row["engine"], row["dataset"], row["repeat"], row["fold"], row["child_fold"]): row
        for row in child_rows
        if row["lane"] == "primary" and row["dataset"] in DIAGNOSTIC_DATASETS
    }
    ordinal = {
        (row["engine"], row["dataset"], row["repeat"], row["fold"], row["child_fold"]): row
        for row in child_rows
        if row["lane"] == "ordinal_diagnostic"
    }
    if set(native) != set(ordinal) or len(native) != 2 * 3 * 8 * 3:
        raise RuntimeError("native/ordinal diagnostic child blocks do not match")
    for key in native:
        if native[key]["child_features"] != ordinal[key]["child_features"]:
            raise RuntimeError(f"native/ordinal external schema mismatch: {key}")


def _ratio_fields(metric: str, numerator: float, denominator: float) -> dict[str, Any]:
    if metric == "incremental_memory_bytes":
        numerator = _nonnegative(numerator, f"{metric} numerator")
        denominator = _nonnegative(denominator, f"{metric} denominator")
        if numerator == 0.0 or denominator == 0.0:
            return {
                f"{metric}_ratio": None,
                f"{metric}_log_ratio": None,
                f"{metric}_pct": None,
                f"{metric}_ratio_available": False,
                f"{metric}_ratio_unavailable_reason": "zero_incremental_memory_observation",
            }
    else:
        numerator = _positive(numerator, f"{metric} numerator")
        denominator = _positive(denominator, f"{metric} denominator")
    ratio = numerator / denominator
    result: dict[str, Any] = {
        f"{metric}_ratio": ratio,
        f"{metric}_log_ratio": math.log(ratio),
        f"{metric}_pct": 100.0 * (ratio - 1.0),
    }
    if metric == "incremental_memory_bytes":
        result[f"{metric}_ratio_available"] = True
        result[f"{metric}_ratio_unavailable_reason"] = None
    return result


def _coordinates_for_panel(panel: str) -> list[tuple[str, int, int]]:
    if panel == "primary":
        return primary_coordinates()
    if panel in {"ordinal_pairwise", "ordinal_uplift"}:
        return diagnostic_coordinates()
    raise RuntimeError(f"unknown analysis panel: {panel}")


def _metrics_for_panel(panel: str) -> tuple[str, ...]:
    # Cross-lane runtime and memory are confounded by separate execution lanes.
    return ACCURACY_METRICS if panel == "ordinal_uplift" else METRICS


def pair_outer_rows(outer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(row["lane"]),
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            str(row["arm"]),
        ): row
        for row in outer_rows
    }
    if len(index) != len(outer_rows):
        raise RuntimeError("duplicate outer row before contrast pairing")
    paired: list[dict[str, Any]] = []
    for spec in contrast_specs():
        for dataset, repeat, fold in _coordinates_for_panel(spec["panel"]):
            numerator = index[
                (spec["numerator_lane"], dataset, repeat, fold, spec["numerator"])
            ]
            denominator = index[
                (
                    spec["denominator_lane"],
                    dataset,
                    repeat,
                    fold,
                    spec["denominator"],
                )
            ]
            row: dict[str, Any] = {
                "panel": spec["panel"],
                "contrast": spec["contrast"],
                "contrast_code": spec["code"],
                "numerator_lane": spec["numerator_lane"],
                "denominator_lane": spec["denominator_lane"],
                "numerator_arm": spec["numerator"],
                "denominator_arm": spec["denominator"],
                "numerator_engine": numerator["engine"],
                "denominator_engine": denominator["engine"],
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
            }
            for metric in METRICS:
                if metric not in _metrics_for_panel(spec["panel"]):
                    row[f"numerator_{metric}"] = None
                    row[f"denominator_{metric}"] = None
                    row[f"{metric}_ratio"] = None
                    row[f"{metric}_log_ratio"] = None
                    row[f"{metric}_pct"] = None
                    if metric == "incremental_memory_bytes":
                        row[f"{metric}_ratio_available"] = False
                        row[f"{metric}_ratio_unavailable_reason"] = (
                            "cross_lane_resource_comparison_omitted"
                        )
                else:
                    numerator_value = float(numerator[metric])
                    denominator_value = float(denominator[metric])
                    row[f"numerator_{metric}"] = numerator_value
                    row[f"denominator_{metric}"] = denominator_value
                    row.update(_ratio_fields(metric, numerator_value, denominator_value))
            paired.append(row)
    if len(paired) != 117 + 18 + 18:
        raise RuntimeError("paired split contrast grid is not exact")
    return paired


def _scalar_or_none(mapping: Mapping[str, Any], field: str) -> Any:
    value = mapping.get(field)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RuntimeError(f"paired child scalar {field} is not scalar")


def pair_child_rows(child_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(row["lane"]),
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["child_fold"]),
            str(row["arm"]),
        ): row
        for row in child_rows
    }
    if len(index) != len(child_rows):
        raise RuntimeError("duplicate child row before contrast pairing")
    paired: list[dict[str, Any]] = []
    telemetry_fields = (
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "stop_reason",
        "deadline_hit",
    )
    for spec in contrast_specs():
        for dataset, repeat, fold in _coordinates_for_panel(spec["panel"]):
            for child_fold in range(8):
                numerator = index[
                    (
                        spec["numerator_lane"],
                        dataset,
                        repeat,
                        fold,
                        child_fold,
                        spec["numerator"],
                    )
                ]
                denominator = index[
                    (
                        spec["denominator_lane"],
                        dataset,
                        repeat,
                        fold,
                        child_fold,
                        spec["denominator"],
                    )
                ]
                numerator_fit = numerator["comparator_fit"]
                denominator_fit = denominator["comparator_fit"]
                schemas = [numerator["child_features"], denominator["child_features"]]
                if schemas[0] != schemas[1]:
                    raise RuntimeError("paired child external feature schemas differ")
                row: dict[str, Any] = {
                    "panel": spec["panel"],
                    "contrast": spec["contrast"],
                    "contrast_code": spec["code"],
                    "dataset": dataset,
                    "task_id": TASKS[dataset],
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": 3 * repeat + fold,
                    "child": f"S1F{child_fold + 1}",
                    "child_fold": child_fold,
                    "numerator_lane": spec["numerator_lane"],
                    "denominator_lane": spec["denominator_lane"],
                    "numerator_arm": spec["numerator"],
                    "denominator_arm": spec["denominator"],
                    "numerator_engine": numerator["engine"],
                    "denominator_engine": denominator["engine"],
                    "numerator_representation_kind": numerator["representation"]["kind"],
                    "denominator_representation_kind": denominator["representation"]["kind"],
                    "external_feature_schema_sha256": _feature_schema_sha256(
                        schemas[0], "paired child features"
                    ),
                    "numerator_initial_hyperparameters_json": _canonical_json(
                        numerator["initial_hyperparameters"]
                    ).decode("utf-8"),
                    "denominator_initial_hyperparameters_json": _canonical_json(
                        denominator["initial_hyperparameters"]
                    ).decode("utf-8"),
                    "numerator_effective_hyperparameters_json": _canonical_json(
                        numerator["effective_hyperparameters"]
                    ).decode("utf-8"),
                    "denominator_effective_hyperparameters_json": _canonical_json(
                        denominator["effective_hyperparameters"]
                    ).decode("utf-8"),
                    "numerator_refit_params_json": _canonical_json(
                        numerator["refit_params"]
                    ).decode("utf-8"),
                    "denominator_refit_params_json": _canonical_json(
                        denominator["refit_params"]
                    ).decode("utf-8"),
                    "numerator_representation_json": _canonical_json(
                        numerator["representation"]
                    ).decode("utf-8"),
                    "denominator_representation_json": _canonical_json(
                        denominator["representation"]
                    ).decode("utf-8"),
                    "numerator_comparator_fit_json": _canonical_json(
                        numerator_fit
                    ).decode("utf-8"),
                    "denominator_comparator_fit_json": _canonical_json(
                        denominator_fit
                    ).decode("utf-8"),
                }
                for field in telemetry_fields:
                    row[f"numerator_{field}"] = _scalar_or_none(numerator_fit, field)
                    row[f"denominator_{field}"] = _scalar_or_none(denominator_fit, field)
                paired.append(row)
    if len(paired) != (117 + 18 + 18) * 8:
        raise RuntimeError("paired child contrast grid is not exact")
    return paired


def _panel_nested_logs(
    split_rows: Sequence[Mapping[str, Any]], metric_log_key: str
) -> dict[str, list[tuple[int, int, float]]]:
    if not split_rows:
        raise RuntimeError("cannot aggregate an empty contrast")
    panels = {str(row["panel"]) for row in split_rows}
    codes = {str(row["contrast_code"]) for row in split_rows}
    if len(panels) != 1 or len(codes) != 1:
        raise RuntimeError("aggregation requires one panel and one contrast")
    panel = next(iter(panels))
    expected = set(_coordinates_for_panel(panel))
    nested: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    seen = set()
    for row in split_rows:
        key = (str(row["dataset"]), int(row["repeat"]), int(row["fold"]))
        if key in seen:
            raise RuntimeError(f"duplicate paired split for {key}")
        seen.add(key)
        value = float(row[metric_log_key])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite paired log ratio for {key}")
        nested[key[0]].append((key[1], key[2], value))
    if seen != expected:
        raise RuntimeError("paired contrast coordinate set is not exact")
    if any(len(values) != 3 for values in nested.values()):
        raise RuntimeError("every dataset must contain exactly three coordinates")
    return {dataset: values for dataset, values in nested.items()}


def equal_dataset_point_log_ratio(
    split_rows: Sequence[Mapping[str, Any]], metric_log_key: str
) -> tuple[float, dict[str, float]]:
    nested = _panel_nested_logs(split_rows, metric_log_key)
    dataset_logs = {
        dataset: math.fsum(value for _, _, value in values) / 3.0
        for dataset, values in sorted(nested.items())
    }
    overall = math.fsum(dataset_logs.values()) / len(dataset_logs)
    return overall, dataset_logs


def fixed_dataset_bootstrap_log_ratios(
    split_rows: Sequence[Mapping[str, Any]],
    metric_log_key: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Keep datasets fixed and resample the three coordinates within each."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    nested = _panel_nested_logs(split_rows, metric_log_key)
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    arrays = {
        dataset: np.asarray([value for _, _, value in values], dtype=np.float64)
        for dataset, values in sorted(nested.items())
    }
    for draw in range(draws):
        dataset_draws = []
        for dataset in sorted(arrays):
            values = arrays[dataset]
            indices = rng.integers(0, 3, size=3)
            dataset_draws.append(float(np.mean(values[indices])))
        output[draw] = math.fsum(dataset_draws) / len(dataset_draws)
    return output


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("distribution contains nonfinite values")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": _quantile(array, 0.90),
        "max": float(np.max(array)),
    }


def _bootstrap_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "draws": int(len(values)),
        "seed": BOOTSTRAP_SEED,
        "datasets_resampled": False,
        "coordinates_resampled_within_dataset": True,
        "ratio_lower95_two_sided": math.exp(_quantile(values, 0.025)),
        "ratio_upper95_one_sided": math.exp(_quantile(values, 0.95)),
        "ratio_upper95_two_sided": math.exp(_quantile(values, 0.975)),
    }


def _absolute_metric_summary(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    datasets: dict[str, dict[str, float]] = {}
    for dataset, selected in sorted(by_dataset.items()):
        if len(selected) != 3:
            raise RuntimeError(f"{metric} absolute summary requires three coordinates")
        datasets[dataset] = {
            "numerator_mean": math.fsum(float(row[f"numerator_{metric}"]) for row in selected)
            / 3.0,
            "denominator_mean": math.fsum(
                float(row[f"denominator_{metric}"]) for row in selected
            )
            / 3.0,
        }
    return {
        "numerator_equal_dataset_mean": math.fsum(
            item["numerator_mean"] for item in datasets.values()
        )
        / len(datasets),
        "denominator_equal_dataset_mean": math.fsum(
            item["denominator_mean"] for item in datasets.values()
        )
        / len(datasets),
    }, datasets


def _child_metadata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("child metadata panel is empty")
    result: dict[str, Any] = {}
    for side in ("numerator", "denominator"):
        best = [float(row[f"{side}_best_iteration"]) for row in rows]
        retained = [float(row[f"{side}_rounds_retained"]) for row in rows]
        stops = Counter(
            "unknown" if row[f"{side}_stop_reason"] is None else row[f"{side}_stop_reason"]
            for row in rows
        )
        lanes = Counter(
            "not_applicable"
            if row[f"{side}_selected_lane"] is None
            else row[f"{side}_selected_lane"]
            for row in rows
        )
        result[side] = {
            "engine": rows[0][f"{side}_engine"],
            "best_iteration": _distribution(best),
            "rounds_retained": _distribution(retained),
            "stop_reason_counts": dict(sorted(stops.items())),
            "selected_lane_counts": dict(sorted(lanes.items())),
            "explicit_time_limit_stops": int(stops.get("time_limit", 0)),
            "unknown_stop_reasons": int(stops.get("unknown", 0)),
        }
    return result


def analyze_contrasts(
    split_rows: Sequence[Mapping[str, Any]],
    child_pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contrast_summaries: list[dict[str, Any]] = []
    per_dataset: list[dict[str, Any]] = []
    for spec in contrast_specs():
        selected = [
            row
            for row in split_rows
            if row["panel"] == spec["panel"] and row["contrast_code"] == spec["code"]
        ]
        expected_count = 39 if spec["panel"] == "primary" else 6
        if len(selected) != expected_count:
            raise RuntimeError(f"{spec['code']} split panel is incomplete")
        metrics: dict[str, dict[str, Any]] = {}
        metric_dataset_logs: dict[str, dict[str, float]] = {}
        incremental_dataset_absolute: dict[str, dict[str, float]] | None = None
        incremental_dataset_ratios: dict[str, float | None] | None = None
        panel_metrics = _metrics_for_panel(spec["panel"])
        for metric in panel_metrics:
            if metric == "incremental_memory_bytes":
                absolute, incremental_dataset_absolute = _absolute_metric_summary(
                    selected, metric
                )
                unavailable = sum(
                    row[f"{metric}_log_ratio"] is None for row in selected
                )
                if unavailable:
                    incremental_dataset_ratios = {}
                    for dataset in sorted({str(row["dataset"]) for row in selected}):
                        dataset_logs = [
                            row[f"{metric}_log_ratio"]
                            for row in selected
                            if row["dataset"] == dataset
                        ]
                        incremental_dataset_ratios[dataset] = (
                            None
                            if any(value is None for value in dataset_logs)
                            else math.exp(
                                math.fsum(float(value) for value in dataset_logs) / 3.0
                            )
                        )
                    metrics[metric] = {
                        "available": False,
                        "unavailable_reason": "zero_incremental_memory_observation",
                        "zero_observation_pair_count": unavailable,
                        "log_ratio": None,
                        "ratio": None,
                        "pct": None,
                        "bootstrap": None,
                        "absolute_bytes": absolute,
                        "role": "primary",
                    }
                    continue
            point, dataset_logs = equal_dataset_point_log_ratio(
                selected, f"{metric}_log_ratio"
            )
            bootstrap = fixed_dataset_bootstrap_log_ratios(
                selected, f"{metric}_log_ratio"
            )
            metrics[metric] = {
                **_ratio_summary(point),
                "bootstrap": _bootstrap_summary(bootstrap),
                "role": (
                    "primary"
                    if metric in PRIMARY_METRICS
                    else "secondary_raw_process_rss"
                ),
            }
            if metric == "incremental_memory_bytes":
                incremental_dataset_ratios = {
                    dataset: math.exp(value) for dataset, value in dataset_logs.items()
                }
                metrics[metric]["available"] = True
                metrics[metric]["unavailable_reason"] = None
                metrics[metric]["zero_observation_pair_count"] = 0
                metrics[metric]["absolute_bytes"] = absolute
            metric_dataset_logs[metric] = dataset_logs
        datasets = []
        for dataset in sorted(metric_dataset_logs["test_rmse"]):
            rows_for_dataset = [row for row in selected if row["dataset"] == dataset]
            item: dict[str, Any] = {
                "panel": spec["panel"],
                "contrast": spec["contrast"],
                "contrast_code": spec["code"],
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "coordinate_count": 3,
                "test_wins": sum(row["test_rmse_ratio"] < 1.0 for row in rows_for_dataset),
                "test_losses": sum(row["test_rmse_ratio"] > 1.0 for row in rows_for_dataset),
                "test_ties": sum(row["test_rmse_ratio"] == 1.0 for row in rows_for_dataset),
            }
            if (
                "incremental_memory_bytes" in panel_metrics
                and incremental_dataset_absolute is not None
            ):
                item["incremental_memory_bytes_numerator_mean_bytes"] = (
                    incremental_dataset_absolute[dataset]["numerator_mean"]
                )
                item["incremental_memory_bytes_denominator_mean_bytes"] = (
                    incremental_dataset_absolute[dataset]["denominator_mean"]
                )
            else:
                item["incremental_memory_bytes_numerator_mean_bytes"] = None
                item["incremental_memory_bytes_denominator_mean_bytes"] = None
            for metric in METRICS:
                if metric == "incremental_memory_bytes" and (
                    incremental_dataset_ratios is not None
                ):
                    ratio = incremental_dataset_ratios[dataset]
                    item[f"{metric}_ratio"] = ratio
                    item[f"{metric}_pct"] = (
                        None if ratio is None else 100.0 * (ratio - 1.0)
                    )
                elif metric in metric_dataset_logs:
                    ratio = math.exp(metric_dataset_logs[metric][dataset])
                    item[f"{metric}_ratio"] = ratio
                    item[f"{metric}_pct"] = 100.0 * (ratio - 1.0)
                else:
                    item[f"{metric}_ratio"] = None
                    item[f"{metric}_pct"] = None
            if spec["panel"] == "ordinal_uplift":
                item["incremental_memory_bytes_ratio_available"] = False
                item["incremental_memory_bytes_ratio_unavailable_reason"] = (
                    "cross_lane_resource_comparison_omitted"
                )
            elif item["incremental_memory_bytes_ratio"] is None:
                item["incremental_memory_bytes_ratio_available"] = False
                item["incremental_memory_bytes_ratio_unavailable_reason"] = (
                    "zero_incremental_memory_observation"
                )
            else:
                item["incremental_memory_bytes_ratio_available"] = True
                item["incremental_memory_bytes_ratio_unavailable_reason"] = None
            datasets.append(item)
            per_dataset.append(item)
        children = [
            row
            for row in child_pairs
            if row["panel"] == spec["panel"] and row["contrast_code"] == spec["code"]
        ]
        if len(children) != expected_count * 8:
            raise RuntimeError(f"{spec['code']} child panel is incomplete")
        dataset_wins = sum(item["test_rmse_ratio"] < 1.0 for item in datasets)
        dataset_losses = sum(item["test_rmse_ratio"] > 1.0 for item in datasets)
        coordinate_wins = sum(row["test_rmse_ratio"] < 1.0 for row in selected)
        coordinate_losses = sum(row["test_rmse_ratio"] > 1.0 for row in selected)
        contrast_summaries.append(
            {
                **spec,
                "paired_splits": len(selected),
                "paired_children": len(children),
                "dataset_count": len(datasets),
                "metrics": metrics,
                "dataset_counts": {
                    "wins": dataset_wins,
                    "losses": dataset_losses,
                    "ties": len(datasets) - dataset_wins - dataset_losses,
                },
                "coordinate_counts": {
                    "wins": coordinate_wins,
                    "losses": coordinate_losses,
                    "ties": len(selected) - coordinate_wins - coordinate_losses,
                },
                "datasets": datasets,
                "child_metadata": _child_metadata_summary(children),
                "policy_advancement_allowed": False,
                "decision": "descriptive_only",
            }
        )
    return {
        "primary": [item for item in contrast_summaries if item["panel"] == "primary"],
        "ordinal_diagnostic": {
            "pairwise": [
                item for item in contrast_summaries if item["panel"] == "ordinal_pairwise"
            ],
            "within_engine_uplift": [
                item for item in contrast_summaries if item["panel"] == "ordinal_uplift"
            ],
            "pooled_with_primary": False,
            "can_revive_ordinal_policy": False,
        },
    }, per_dataset


def verify_campaign_integrity(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify every frozen campaign byte and load only attested finite JSON."""
    input_dir = input_dir.resolve(strict=True)
    manifest_path = input_dir / campaign.MANIFEST_FILENAME
    attestation_path = input_dir / campaign.COMPLETION_ATTESTATION_FILENAME
    manifest, manifest_bytes = _read_json_stable(manifest_path, "run manifest")
    manifest_fields = {
        "schema_version",
        "kind",
        "created_at_utc",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "job_order_sha256",
        "protocol",
        "ordering_audit",
        "source",
        "runtime",
    }
    if set(manifest) != manifest_fields:
        raise RuntimeError("run manifest fields are not exact")
    protocol = campaign.frozen_protocol()
    protocol_digest = _sha256(_canonical_json(protocol))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != campaign.CAMPAIGN_KIND
        or Path(str(manifest.get("output_dir", ""))).resolve() != input_dir
        or manifest.get("protocol") != protocol
        or manifest.get("protocol_sha256") != protocol_digest
        or manifest.get("time_limit_seconds") != 3_600.0
        or protocol.get("analysis", {}).get("bootstrap_draws") != BOOTSTRAP_DRAWS
        or protocol.get("analysis", {}).get("bootstrap_seed") != BOOTSTRAP_SEED
        or protocol.get("analysis", {}).get("lanes_pooled") is not False
    ):
        raise RuntimeError("run manifest does not match the frozen same-machine campaign")
    child_cpus = _exact_int(
        manifest.get("resolved_child_num_cpus"), "manifest child CPUs"
    )
    if child_cpus != 18:
        raise RuntimeError("same-machine campaign must use exactly 18 child CPUs")
    execution = verify_execution_provenance(manifest, input_dir)
    expected_order_digest = _validate_ordering_metadata(manifest)

    attestation, attestation_bytes = _read_json_stable(
        attestation_path, "completion attestation"
    )
    attestation_fields = {
        "schema_version",
        "kind",
        "completed_at_utc",
        "pid",
        "result_count",
        "expected_result_count",
        "expected_primary_result_count",
        "expected_ordinal_diagnostic_result_count",
        "expected_child_fits",
        "warmup_thread_count",
        "warmup_stage_count",
        "protocol_sha256",
        "job_order_sha256",
        "git_head",
        "manifest_sha256",
        "result_artifacts",
        "analysis_payload_artifact",
        "warmup_history_artifact",
        "resume_history_artifact",
        "validation",
    }
    if set(attestation) != attestation_fields:
        raise RuntimeError("completion attestation fields are not exact")
    expected_counts = {
        "result_count": 135,
        "expected_result_count": 135,
        "expected_primary_result_count": 117,
        "expected_ordinal_diagnostic_result_count": 18,
        "expected_child_fits": 1_080,
    }
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind") != campaign.COMPLETION_KIND
        or any(attestation.get(name) != count for name, count in expected_counts.items())
        or attestation.get("protocol_sha256") != protocol_digest
        or attestation.get("git_head") != manifest["source"]["git_head"]
        or attestation.get("manifest_sha256") != _sha256(manifest_bytes)
        or attestation.get("warmup_thread_count") != child_cpus
        or attestation.get("warmup_stage_count") != len(campaign.WARMUP_STAGE_NAMES)
    ):
        raise RuntimeError("completion attestation does not match the campaign")
    _validate_ordering_metadata(manifest, attestation=attestation)

    artifacts = hardened._as_mapping(
        attestation.get("result_artifacts"), "result artifacts"
    )
    if len(artifacts) != 135:
        raise RuntimeError("attested raw-result count is wrong")
    experiments = input_dir / "experiments"
    observed = (
        {str(path.relative_to(input_dir)) for path in experiments.rglob("results.pkl")}
        if experiments.exists()
        else set()
    )
    if observed != set(artifacts):
        raise RuntimeError("on-disk raw-result set does not match the attestation")
    for relative, raw_metadata in artifacts.items():
        if not isinstance(relative, str) or Path(relative).name != "results.pkl":
            raise RuntimeError("attested raw result has an unsafe filename")
        _artifact_bytes(
            input_dir,
            relative,
            hardened._as_mapping(raw_metadata, f"result artifact {relative}"),
            f"raw result {relative}",
        )

    payload_artifact = hardened._as_mapping(
        attestation.get("analysis_payload_artifact"), "analysis payload artifact"
    )
    if set(payload_artifact) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError("analysis payload attestation fields are not exact")
    if payload_artifact.get("path") != campaign.ANALYSIS_PAYLOAD_FILENAME:
        raise RuntimeError("analysis payload path is not frozen")
    payload_bytes = _artifact_bytes(
        input_dir,
        campaign.ANALYSIS_PAYLOAD_FILENAME,
        {name: payload_artifact[name] for name in ("sha256", "size_bytes")},
        "safe analysis payload",
    )
    payload = _decode_json(payload_bytes, "safe analysis payload")
    payload_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "job_order_sha256",
        "result_artifacts_sha256",
        "outer_rows",
        "child_rows",
    }
    if set(payload) != payload_fields:
        raise RuntimeError("safe analysis payload fields are not exact")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != campaign.PAYLOAD_KIND
        or payload.get("protocol_sha256") != protocol_digest
        or payload.get("job_order_sha256") != expected_order_digest
        or payload.get("result_artifacts_sha256")
        != _sha256(_canonical_json(artifacts))
    ):
        raise RuntimeError("safe analysis payload does not bind the campaign")
    _validate_ordering_metadata(manifest, payload=payload)
    outer_rows, outer_index = _validate_outer_rows(
        payload.get("outer_rows"), artifacts, child_cpus
    )
    child_rows = _validate_child_rows(
        payload.get("child_rows"), outer_index, child_cpus
    )
    payload["outer_rows"] = outer_rows
    payload["child_rows"] = child_rows

    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_comparator_warmup import validate_comparator_warmup_history

    completion_pid = _exact_int(attestation.get("pid"), "completion PID")
    warmup_history_digest = hardened._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="warmup_history_artifact",
        filename=campaign.WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: validate_comparator_warmup_history(
            value,
            expected_thread_count=child_cpus,
            expected_latest_pid=completion_pid,
        ),
    )
    resume_history_digest = hardened._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="resume_history_artifact",
        filename=campaign.RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: campaign._validate_resume_history(value, input_dir),
    )

    validation = dict(
        hardened._as_mapping(attestation.get("validation"), "completion validation")
    )
    validation_fields = {
        "result_count",
        "child_fit_count",
        "lane_result_counts",
        "lane_child_counts",
        "cross_engine_representation_blocks",
        "failure_count",
        "imputation_count",
        "known_deadline_hit_count",
        "known_time_limit_stop_count",
        "stop_reason_counts",
        "competitor_stop_reason_inference_counts",
        "job_order_sha256",
        "resource_allocation",
        "memory_metric",
    }
    if set(validation) != validation_fields:
        raise RuntimeError("completion validation fields are not exact")
    expected_validation = {
        "result_count": 135,
        "child_fit_count": 1_080,
        "lane_result_counts": {"ordinal_diagnostic": 18, "primary": 117},
        "lane_child_counts": {"ordinal_diagnostic": 144, "primary": 936},
        "cross_engine_representation_blocks": 360,
        "failure_count": 0,
        "imputation_count": 0,
        "known_deadline_hit_count": 0,
        "known_time_limit_stop_count": 0,
        "job_order_sha256": expected_order_digest,
        "resource_allocation": {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        },
        "memory_metric": "peak_mem_cpu_minus_min_mem_cpu",
    }
    if any(validation.get(name) != expected for name, expected in expected_validation.items()):
        raise RuntimeError("completion validation does not match the campaign")
    stop_counts = Counter(
        "unknown" if row["comparator_fit"]["stop_reason"] is None else row["comparator_fit"]["stop_reason"]
        for row in child_rows
    )
    inferred_counts = Counter()
    for row in child_rows:
        inferred = row["comparator_fit"].get("stop_reason_inferred")
        if inferred is not None:
            inferred_counts["inferred" if inferred is True else "unresolved"] += 1
    if validation.get("stop_reason_counts") != dict(sorted(stop_counts.items())):
        raise RuntimeError("attested stop-reason counts do not match safe child rows")
    if validation.get("competitor_stop_reason_inference_counts") != dict(
        sorted(inferred_counts.items())
    ):
        raise RuntimeError("attested competitor stop inference does not match children")
    if stop_counts.get("time_limit", 0):
        raise RuntimeError("campaign contains an explicit time-limit stop")

    return manifest, attestation, payload, {
        "manifest_sha256": _sha256(manifest_bytes),
        "attestation_sha256": _sha256(attestation_bytes),
        "analysis_payload_sha256": _sha256(payload_bytes),
        "warmup_history_sha256": warmup_history_digest,
        "resume_history_sha256": resume_history_digest,
        **execution,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty {field}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"{field} row schemas are not deterministic")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _format_ratio(item: Mapping[str, Any], metric: str, digits: int = 4) -> str:
    value = item["metrics"][metric]["ratio"]
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _format_optional_number(value: Any, digits: int = 4) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def _format_incremental_memory(item: Mapping[str, Any]) -> str:
    metric = item["metrics"]["incremental_memory_bytes"]
    if metric["ratio"] is not None:
        return f"{metric['ratio']:.4f}"
    absolute = metric["absolute_bytes"]
    return (
        "unavailable "
        f"({absolute['numerator_equal_dataset_mean']:.0f}/"
        f"{absolute['denominator_equal_dataset_mean']:.0f} B)"
    )


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    same_lane_contrasts = [
        *summary["primary"],
        *summary["ordinal_diagnostic"]["pairwise"],
    ]
    zero_memory_pairs = sum(
        item["metrics"]["incremental_memory_bytes"].get(
            "zero_observation_pair_count", 0
        )
        for item in same_lane_contrasts
    )
    lines = [
        "# Same-machine TabArena regression comparison",
        "",
        "The primary panel compares the official DarkoFit 0.9.0, "
        "ChimeraBoost 0.14.1, and CatBoost 1.2.10 defaults on identical "
        "r0f0/r1f1/r2f2 coordinates across 13 datasets. Ratios below one "
        "favor the numerator. Each dataset receives equal weight; the bootstrap "
        "keeps datasets fixed and resamples the three coordinates within each.",
        "",
        "This is descriptive characterization only. The Airfoil/Diamonds "
        "safe-ordinal lane is reported separately, is never pooled with the "
        "primary panel, and cannot revive or advance the rejected ordinal policy.",
        "",
        "## Material default-policy differences",
        "",
        "These are product-default comparisons, not hyperparameter-parity "
        "comparisons. The material frozen differences are:",
        "",
        "| Engine | Cap | Learning rate | Depth/mode | L2 | Bins | Categorical permutations | Ordered boosting | Linear lane | Early stopping |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
        "| DarkoFit | 1,000 | auto | auto depth / catboost mode | auto → 3 | 254 | 1 | auto → off for scalar regression | off | on |",
        "| ChimeraBoost | 10,000 | 0.1 | depth 6 | 1 | 128 | 4 | off | auto-select (`None`) | on; patience 50 |",
        "| CatBoost | 10,000 | 0.05 | official default | official default | official default | native | native policy | off | AutoGluon adaptive; use-best with eval set |",
        "",
        "## Primary out-of-box defaults",
        "",
        "| Contrast | Test RMSE | Test 95% CI | Validation | Train | Infer | Incremental memory | Raw RSS | Dataset wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["primary"]:
        test = item["metrics"]["test_rmse"]
        ci = test["bootstrap"]
        counts = item["dataset_counts"]
        lines.append(
            f"| {item['code']} | {test['ratio']:.6f} | "
            f"[{ci['ratio_lower95_two_sided']:.6f}, "
            f"{ci['ratio_upper95_two_sided']:.6f}] | "
            f"{_format_ratio(item, 'val_rmse')} | "
            f"{_format_ratio(item, 'train_time_s')} | "
            f"{_format_ratio(item, 'infer_time_s')} | "
            f"{_format_incremental_memory(item)} | "
            f"{_format_ratio(item, 'peak_memory_bytes')} | "
            f"{counts['wins']}/{item['dataset_count']} |"
        )
    for item in summary["primary"]:
        lines.extend(
            [
                "",
                f"### {item['code']}: {item['contrast']}",
                "",
                f"Coordinate wins/losses/ties: {item['coordinate_counts']['wins']}/"
                f"{item['coordinate_counts']['losses']}/"
                f"{item['coordinate_counts']['ties']}. Dataset wins/losses/ties: "
                f"{item['dataset_counts']['wins']}/{item['dataset_counts']['losses']}/"
                f"{item['dataset_counts']['ties']}.",
                "",
                "| Dataset | Test | Validation | Train | Infer | Incremental memory |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset in item["datasets"]:
            lines.append(
                f"| {dataset['dataset']} | {dataset['test_rmse_ratio']:.6f} | "
                f"{dataset['val_rmse_ratio']:.6f} | "
                f"{dataset['train_time_s_ratio']:.4f} | "
                f"{dataset['infer_time_s_ratio']:.4f} | "
                f"{_format_optional_number(dataset['incremental_memory_bytes_ratio'])} |"
            )

    diagnostic = summary["ordinal_diagnostic"]
    lines.extend(
        [
            "",
            "## Safe-ordinal diagnostic — separate evidence lane",
            "",
            "### Cross-engine comparison under identical safe ordinal inputs",
            "",
            "| Contrast | Test RMSE | Test 95% CI | Validation | Train | Infer | Incremental memory |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in diagnostic["pairwise"]:
        test = item["metrics"]["test_rmse"]
        ci = test["bootstrap"]
        lines.append(
            f"| {item['code']} | {test['ratio']:.6f} | "
            f"[{ci['ratio_lower95_two_sided']:.6f}, "
            f"{ci['ratio_upper95_two_sided']:.6f}] | "
            f"{_format_ratio(item, 'val_rmse')} | "
            f"{_format_ratio(item, 'train_time_s')} | "
            f"{_format_ratio(item, 'infer_time_s')} | "
            f"{_format_incremental_memory(item)} |"
        )
    lines.extend(
        [
            "",
            "### Within-engine safe ordinal / native uplift",
            "",
            "These rows match each diagnostic job to that engine's primary job "
            "on the same dataset, repeat, fold, and child-fold structure.",
            "",
            "| Engine contrast | Test RMSE | Test 95% CI | Validation |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in diagnostic["within_engine_uplift"]:
        test = item["metrics"]["test_rmse"]
        ci = test["bootstrap"]
        lines.append(
            f"| {item['code']} | {test['ratio']:.6f} | "
            f"[{ci['ratio_lower95_two_sided']:.6f}, "
            f"{ci['ratio_upper95_two_sided']:.6f}] | "
            f"{_format_ratio(item, 'val_rmse')} |"
        )
    lines.extend(
        [
            "",
            "Cross-lane ordinal/native training time, inference time, and memory "
            "are intentionally omitted because separate lane execution makes "
            "those ratios order- and process-history-confounded.",
        ]
    )

    chimera = summary["engine_child_metadata"]["chimeraboost"]
    integrity = summary["integrity_diagnostics"]
    lines.extend(
        [
            "",
            "## Fitted child telemetry",
            "",
            f"- ChimeraBoost selected lanes: `{json.dumps(chimera['selected_lane_counts'], sort_keys=True)}`.",
            f"- ChimeraBoost retained rounds: median {chimera['rounds_retained']['median']:.1f}, "
            f"p90 {chimera['rounds_retained']['p90']:.1f}, max "
            f"{chimera['rounds_retained']['max']:.0f}.",
            f"- Known failures/imputations/deadlines/time limits: "
            f"{integrity['failure_count']}/{integrity['imputation_count']}/"
            f"{integrity['known_deadline_hit_count']}/"
            f"{integrity['known_time_limit_stop_count']}.",
            f"- Competitor children with unresolved stop reason: "
            f"{integrity['unknown_stop_reason_children']} (reported, never "
            "silently classified as early stopping).",
            "- A null competitor stop reason can include an unexposed time or "
            "memory callback outcome; this qualifies the descriptive comparison "
            "and is not evidence that every competitor child avoided truncation.",
            "",
            "## Measurement and integrity",
            "",
            "- Training and inference timings are same-machine measurements from "
            "the same frozen campaign.",
            "- Incremental memory (`peak_mem_cpu - min_mem_cpu`) is the primary "
            "memory comparison; raw process peak RSS is secondary.",
            f"- Zero incremental-memory observations affecting a ratio: "
            f"{zero_memory_pairs}. Affected log-ratio aggregates are marked "
            "unavailable without an epsilon or pseudocount; raw bytes remain "
            "reported.",
            "- All 135 raw result files were verified as opaque bytes. This "
            "analyzer never unpickled them.",
            "- Exact source commits/wheel bytes, adapters, runtime, hardware, "
            "configuration, feature schemas, order, 135 outer rows, and 1,080 "
            "child rows matched the completion attestation.",
            "",
            "## Provenance",
            "",
            f"- DarkoFit Git commit: `{summary['provenance']['git_head']}`.",
            f"- ChimeraBoost Git commit: `{summary['provenance']['chimeraboost_git_head']}`.",
            f"- Frozen protocol semantic SHA-256: `{summary['provenance']['protocol_sha256']}`.",
            f"- Ordered-grid SHA-256: `{summary['provenance']['job_order_sha256']}`.",
            f"- Manifest SHA-256: `{summary['provenance']['manifest_sha256']}`.",
            f"- Completion attestation SHA-256: `{summary['provenance']['attestation_sha256']}`.",
            f"- Safe analysis payload SHA-256: `{summary['provenance']['analysis_payload_sha256']}`.",
            "",
            "## Decision boundary",
            "",
            "**Descriptive comparison only. No default or ordinal policy is "
            "advanced by this analysis.**",
        ]
    )
    return "\n".join(lines) + "\n"


def _engine_child_metadata(child_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for engine in ("darkofit", "chimeraboost", "catboost"):
        rows = [row for row in child_rows if row["engine"] == engine]
        if len(rows) != 360:
            raise RuntimeError(f"{engine} child telemetry count is wrong")
        fits = [row["comparator_fit"] for row in rows]
        lanes = Counter(
            "not_applicable" if fit.get("selected_lane") is None else fit["selected_lane"]
            for fit in fits
        )
        stops = Counter(
            "unknown" if fit.get("stop_reason") is None else fit["stop_reason"]
            for fit in fits
        )
        output[engine] = {
            "child_count": len(rows),
            "best_iteration": _distribution(
                [float(fit["best_iteration"]) for fit in fits]
            ),
            "rounds_retained": _distribution(
                [float(fit["rounds_retained"]) for fit in fits]
            ),
            "selected_lane_counts": dict(sorted(lanes.items())),
            "stop_reason_counts": dict(sorted(stops.items())),
        }
    return output


def _publish_outputs_atomically(
    outputs: Mapping[str, Path],
    payloads: Mapping[str, bytes],
    *,
    post_write_check: Callable[[], None],
) -> None:
    if set(outputs) != set(OUTPUT_KEYS) or set(payloads) != set(OUTPUT_KEYS):
        raise RuntimeError("managed same-machine output fields are not exact")
    hardened._atomic_write_group(
        [(outputs[name], payloads[name]) for name in OUTPUT_KEYS],
        post_write_check=post_write_check,
    )


def _canonical_same_machine_output_targets(
    input_dir: Path,
    targets: Mapping[str, Path],
    *,
    protected_paths: Sequence[Path],
) -> dict[str, Path]:
    """Apply one immutable seven-output path-safety contract at every gate."""
    return hardened._canonical_output_targets(
        input_dir,
        targets,
        protected_paths=protected_paths,
        target_names=OUTPUT_KEYS,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve(strict=True)
    if tuple(campaign.DEFAULT_ANALYSIS_OUTPUT_FILENAMES) != OUTPUT_NAMES:
        raise RuntimeError("runner/analyzer output filename contract changed")
    manifest_path = (input_dir / campaign.MANIFEST_FILENAME).resolve(strict=True)
    attestation_path = (
        input_dir / campaign.COMPLETION_ATTESTATION_FILENAME
    ).resolve(strict=True)
    manifest, attestation, payload, digests = verify_campaign_integrity(input_dir)

    requested_outputs = {
        key: input_dir / name for key, name in zip(OUTPUT_KEYS, OUTPUT_NAMES)
    }
    protected = hardened._protected_campaign_paths(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    outputs = _canonical_same_machine_output_targets(
        input_dir, requested_outputs, protected_paths=protected
    )
    paired_splits = pair_outer_rows(payload["outer_rows"])
    paired_children = pair_child_rows(payload["child_rows"])
    analyzed, per_dataset = analyze_contrasts(paired_splits, paired_children)
    primary_splits = [row for row in paired_splits if row["panel"] == "primary"]
    diagnostic_splits = [row for row in paired_splits if row["panel"] != "primary"]
    primary_datasets = [row for row in per_dataset if row["panel"] == "primary"]
    diagnostic_datasets = [row for row in per_dataset if row["panel"] != "primary"]
    expected_counts = {
        "outer_jobs": 135,
        "primary_outer_jobs": 117,
        "ordinal_diagnostic_outer_jobs": 18,
        "child_fits": 1_080,
        "primary_coordinates": 39,
        "ordinal_diagnostic_coordinates": 6,
        "primary_paired_splits": 117,
        "ordinal_diagnostic_paired_splits": 36,
        "paired_child_rows": 1_224,
        "primary_per_dataset_rows": 39,
        "ordinal_diagnostic_per_dataset_rows": 12,
    }
    observed_counts = {
        "outer_jobs": len(payload["outer_rows"]),
        "primary_outer_jobs": sum(row["lane"] == "primary" for row in payload["outer_rows"]),
        "ordinal_diagnostic_outer_jobs": sum(
            row["lane"] == "ordinal_diagnostic" for row in payload["outer_rows"]
        ),
        "child_fits": len(payload["child_rows"]),
        "primary_coordinates": len(primary_coordinates()),
        "ordinal_diagnostic_coordinates": len(diagnostic_coordinates()),
        "primary_paired_splits": len(primary_splits),
        "ordinal_diagnostic_paired_splits": len(diagnostic_splits),
        "paired_child_rows": len(paired_children),
        "primary_per_dataset_rows": len(primary_datasets),
        "ordinal_diagnostic_per_dataset_rows": len(diagnostic_datasets),
    }
    if observed_counts != expected_counts:
        raise RuntimeError("same-machine analysis counts do not match frozen design")
    validation = dict(hardened._as_mapping(attestation["validation"], "validation"))
    unknown_stops = sum(
        row["comparator_fit"].get("stop_reason") is None
        for row in payload["child_rows"]
    )
    summary: dict[str, Any] = {
        "protocol": (
            "same-machine official-default characterization with a separate, "
            "non-poolable safe-ordinal diagnostic"
        ),
        "decision": "descriptive_only",
        "policy_advancement_allowed": False,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "datasets_fixed": True,
            "resampling": "three_coordinates_within_each_fixed_dataset",
            "primary_dataset_weights": {dataset: 1.0 / 13.0 for dataset in TASKS},
            "ordinal_diagnostic_dataset_weights": {
                dataset: 0.5 for dataset in DIAGNOSTIC_DATASETS
            },
            "lanes_pooled": False,
        },
        "counts": observed_counts,
        "official_default_disclosure": manifest["protocol"][
            "official_default_disclosure"
        ],
        **analyzed,
        "engine_child_metadata": _engine_child_metadata(payload["child_rows"]),
        "integrity_diagnostics": {
            "validation_basis": (
                "completion attestation, opaque raw-result hashes, runner-normalized "
                "finite JSON, and independent exact config/schema/order revalidation"
            ),
            "failure_count": validation["failure_count"],
            "imputation_count": validation["imputation_count"],
            "known_deadline_hit_count": validation["known_deadline_hit_count"],
            "known_time_limit_stop_count": validation["known_time_limit_stop_count"],
            "unknown_stop_reason_children": unknown_stops,
            "raw_result_hashes_verified": 135,
            "outer_rows_verified": 135,
            "child_rows_verified": 1_080,
            "cross_engine_representation_blocks_verified": 360,
            "job_order_verified": True,
            "primary_and_diagnostic_kept_separate": True,
        },
        "provenance": {
            **digests,
            "manifest_path": str(manifest_path),
            "attestation_path": str(attestation_path),
            "protocol_sha256": campaign.protocol_sha256(),
            "job_order_sha256": job_order_sha256(),
            "git_head": manifest["source"]["git_head"],
            "chimeraboost_git_head": manifest["source"]["chimeraboost"]["git_head"],
            "catboost_version": manifest["source"]["catboost"]["version"],
            "completed_at_utc": attestation.get("completed_at_utc"),
        },
    }

    def build_output_payloads() -> dict[str, bytes]:
        return {
            "primary_split_csv": _csv_bytes(
                primary_splits, "primary paired-split CSV"
            ),
            "primary_dataset_csv": _csv_bytes(
                primary_datasets, "primary per-dataset CSV"
            ),
            "diagnostic_split_csv": _csv_bytes(
                diagnostic_splits, "ordinal diagnostic paired-split CSV"
            ),
            "diagnostic_dataset_csv": _csv_bytes(
                diagnostic_datasets, "ordinal diagnostic per-dataset CSV"
            ),
            "child_csv": _csv_bytes(paired_children, "paired-child CSV"),
            "summary_json": (
                json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            "report_md": render_markdown_report(summary).encode("utf-8"),
        }

    output_payloads = build_output_payloads()
    if output_payloads != build_output_payloads():
        raise RuntimeError("same-machine analysis output bytes are not deterministic")
    outputs = _canonical_same_machine_output_targets(
        input_dir, outputs, protected_paths=protected
    )
    baseline_snapshot = (manifest, attestation, payload, digests)

    def assert_campaign_unchanged() -> None:
        if verify_campaign_integrity(input_dir) != baseline_snapshot:
            raise RuntimeError("campaign artifacts changed during analysis")

    assert_campaign_unchanged()
    _publish_outputs_atomically(
        outputs, output_payloads, post_write_check=assert_campaign_unchanged
    )
    print(
        "analyzed 135 jobs and 1,080 child fits; descriptive_only; "
        f"wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
