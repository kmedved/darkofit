"""Verify and analyze the frozen B10/A10 TabArena accuracy shootout.

The analyzer consumes only the runner's attested JSON payload.  Raw pickle
results remain opaque artifacts: their bytes are authenticated here, but only
the runner is allowed to decode them.  Reused product-default (P),
ChimeraBoost (M), and CatBoost (C) observations are joined from the exact
committed same-machine artifacts named in the frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from benchmarks import analyze_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_tabarena_regression_accuracy_shootout as campaign
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_cap_horizon as hardened
    import run_tabarena_regression_accuracy_shootout as campaign


SOURCE_COMMIT = "a1ff4b74510b5e314bb41c27b40544910741543d"
SOURCE_DARKOFIT_SUBTREE = "52278b0326419a45a72bdfd3afcfc13019087838"
CHIMERABOOST_COMMIT = "9c9ea6e704a9fe2bfe6d6c284b22de73914be048"
CATBOOST_VERSION = "1.2.10"
SOURCE_ARTIFACTS = {
    "tabarena_regression_same_machine_primary_paired_splits.csv": (
        "3e7bbe21e0ffe40771f2065dc252dbd4314550f8ab350f2fbed9641401b341b1"
    ),
    "tabarena_regression_same_machine_summary.json": (
        "ca23618bdc3d9e0ab38557e7738c66e95827945ad34e3eb63005f253c92ccf01"
    ),
    "tabarena_regression_same_machine_completion_attestation.json": (
        "213f462aa06103e97864ecd786b75e8fd8e11743c77f556262fa39bdb3e1b7d9"
    ),
    "tabarena_regression_same_machine_run_manifest.json": (
        "2869acaaa4bcc8319d9ba03744a4a9ca8602ed349553a031c3d84ab537de72ee"
    ),
}
REUSED_EVIDENCE_CONTRACT = {
    "source_commit": SOURCE_COMMIT,
    "source_darkofit_subtree": SOURCE_DARKOFIT_SUBTREE,
    "chimeraboost_tag_commit": CHIMERABOOST_COMMIT,
    "catboost_version": CATBOOST_VERSION,
    "artifacts": dict(SOURCE_ARTIFACTS),
}

# Only these two distributions were needed exclusively by the reused live
# comparator arms. Every other distribution in the source lock is part of the
# common DarkoFit/TabArena execution environment and must remain explicit.
COMPARATOR_ONLY_PACKAGES = frozenset({"catboost", "chimeraboost"})
COMMON_DEPENDENCY_PACKAGES = frozenset(
    campaign.REUSED_COMMON_PACKAGE_DISTRIBUTIONS
)
SOURCE_DEPENDENCY_PACKAGES = COMMON_DEPENDENCY_PACKAGES | COMPARATOR_ONLY_PACKAGES
DEPENDENCY_PROVENANCE_FIELDS = frozenset(
    {
        "module",
        "module_file",
        "repository",
        "git_head",
        "git_tree",
        "git_remote_origin",
        "status",
    }
)

QUALITY_METRICS = ("test_rmse", "val_rmse")
NEW_METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "peak_memory_bytes",
)
CONTRASTS = (
    ("A10/M", "A10", "M"),
    ("A10/P", "A10", "P"),
    ("A10/B10", "A10", "B10"),
    ("B10/P", "B10", "P"),
)
GATE_THRESHOLDS = {
    "parity_a10_over_m_max": 1.0,
    "worst_dataset_a10_over_p_max": 1.02,
    "worst_lodo_a10_over_m_max": 1.01,
}

OUTPUT_KEYS = (
    "split_csv",
    "dataset_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
OUTPUT_NAMES = (
    "paired_splits.csv",
    "per_dataset.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)


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


def _read_stable(path: Path, field: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file: {path}")
    return hardened._read_stable(path, field)


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable(path, field)
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not finite valid JSON") from exc
    return dict(hardened._as_mapping(value, field)), payload


def _positive(value: Any, field: str) -> float:
    return hardened._positive_finite(value, field)


def _nonnegative(value: Any, field: str) -> float:
    return hardened._nonnegative_finite(value, field)


def _exact_int(value: Any, field: str) -> int:
    return hardened._exact_int(value, field)


def _csv_int(value: Any, field: str) -> int:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a canonical CSV integer")
    try:
        number = int(value)
    except (ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be a canonical CSV integer") from exc
    if str(number) != value:
        raise RuntimeError(f"{field} must be a canonical CSV integer")
    return number


def expected_outer_grid() -> set[tuple[str, int, int, str]]:
    return {
        (dataset, repeat, fold, internal_arm)
        for dataset, repeat, fold in campaign.expected_coordinates()
        for internal_arm in campaign.INTERNAL_TO_PUBLIC_ARM
    }


def expected_child_grid() -> set[tuple[str, int, int, str, int]]:
    return {
        (*outer, child_fold)
        for outer in expected_outer_grid()
        for child_fold in range(8)
    }


def _git(repository: Path, args: Sequence[str], field: str) -> str:
    return hardened._git_output(repository, list(args), field)


def _validate_reused_tabarena_provenance(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    source_tabarena = dict(
        hardened._as_mapping(source.get("tabarena"), "source TabArena provenance")
    )
    current_tabarena = dict(
        hardened._as_mapping(current.get("tabarena"), "shootout TabArena provenance")
    )
    if (
        set(source_tabarena) != DEPENDENCY_PROVENANCE_FIELDS
        or set(current_tabarena) != DEPENDENCY_PROVENANCE_FIELDS
        or source_tabarena.get("module") != "tabarena"
        or source_tabarena.get("status") != ""
        or current_tabarena.get("status") != ""
    ):
        raise RuntimeError("TabArena provenance fields are incomplete or dirty")
    for field in ("module_file", "repository"):
        source_path = hardened._manifest_path(
            source_tabarena[field], f"source TabArena {field}"
        )
        current_path = hardened._manifest_path(
            current_tabarena[field], f"shootout TabArena {field}"
        )
        if current_path != source_path:
            raise RuntimeError(
                f"shootout TabArena {field} differs from reused evidence"
            )
    for field in ("module", "git_head", "git_tree", "git_remote_origin"):
        if current_tabarena.get(field) != source_tabarena.get(field):
            raise RuntimeError(
                f"shootout TabArena {field} differs from reused evidence"
            )
    return {
        "git_head": source_tabarena["git_head"],
        "git_tree": source_tabarena["git_tree"],
        "repository": source_tabarena["repository"],
    }


def _validate_common_dependency_lock(
    source_packages: Mapping[str, Any], current_packages: Mapping[str, Any]
) -> dict[str, Any]:
    source_names = set(source_packages)
    recorded_current_names = set(current_packages)
    # The shootout records optional comparators explicitly as ``None`` when
    # they are not installed.  Treat that representation the same as an
    # omitted key, while retaining strict version equality for an installed
    # comparator (and for every common dependency).
    current_names = {
        name
        for name, version in current_packages.items()
        if not (name in COMPARATOR_ONLY_PACKAGES and version is None)
    }
    if source_names != SOURCE_DEPENDENCY_PACKAGES:
        raise RuntimeError("reused source dependency lock is not the frozen complete set")
    missing_common = COMMON_DEPENDENCY_PACKAGES.difference(current_names)
    unexpected = recorded_current_names.difference(SOURCE_DEPENDENCY_PACKAGES)
    if missing_common or unexpected:
        detail = []
        if missing_common:
            detail.append("missing common: " + ", ".join(sorted(missing_common)))
        if unexpected:
            detail.append("unexpected: " + ", ".join(sorted(unexpected)))
        raise RuntimeError("shootout dependency lock is not complete (" + "; ".join(detail) + ")")
    omitted = source_names.difference(current_names)
    if not omitted.issubset(COMPARATOR_ONLY_PACKAGES):
        raise RuntimeError("shootout omitted a non-comparator dependency")
    for name in current_names:
        source_version = source_packages.get(name)
        current_version = current_packages.get(name)
        if (
            not isinstance(source_version, str)
            or not source_version
            or current_version != source_version
        ):
            raise RuntimeError(
                f"shootout dependency version differs from reused evidence: {name}"
            )
    return {
        "common_packages": sorted(COMMON_DEPENDENCY_PACKAGES),
        "comparator_only_packages": sorted(COMPARATOR_ONLY_PACKAGES),
        "omitted_comparator_packages": sorted(omitted),
    }


def _verify_source_reuse_contract(
    repository: Path, manifest: Mapping[str, Any]
) -> tuple[dict[tuple[str, int, int], dict[str, float]], dict[str, Any]]:
    """Authenticate the committed P/M/C observations and their reuse boundary."""
    if manifest.get("reused_evidence") != REUSED_EVIDENCE_CONTRACT:
        raise RuntimeError("run manifest reused-evidence contract is not exact")

    source_tree = _git(
        repository,
        ["rev-parse", f"{SOURCE_COMMIT}:darkofit"],
        "source DarkoFit subtree",
    )
    current_tree = _git(
        repository, ["rev-parse", "HEAD:darkofit"], "current DarkoFit subtree"
    )
    if source_tree != SOURCE_DARKOFIT_SUBTREE or current_tree != source_tree:
        raise RuntimeError("DarkoFit package subtree changed since reused evidence")

    artifact_payloads: dict[str, bytes] = {}
    for name, expected_digest in SOURCE_ARTIFACTS.items():
        payload = _read_stable(repository / "benchmarks" / name, f"reused {name}")
        if _sha256(payload) != expected_digest:
            raise RuntimeError(f"reused artifact digest changed: {name}")
        artifact_payloads[name] = payload

    source_manifest = json.loads(
        artifact_payloads["tabarena_regression_same_machine_run_manifest.json"]
    )
    source_attestation = json.loads(
        artifact_payloads[
            "tabarena_regression_same_machine_completion_attestation.json"
        ]
    )
    source_summary = json.loads(
        artifact_payloads["tabarena_regression_same_machine_summary.json"]
    )
    source_protocol = hardened._as_mapping(
        source_manifest.get("protocol"), "reused source protocol"
    )
    expected_coordinates = [
        {"dataset": dataset, "repeat": repeat, "fold": fold}
        for dataset, repeat, fold in campaign.expected_coordinates()
    ]
    primary = hardened._as_mapping(
        hardened._as_mapping(source_protocol.get("lanes"), "source lanes").get(
            "primary"
        ),
        "source primary lane",
    )
    if (
        source_manifest.get("resolved_child_num_cpus") != campaign.EXPECTED_CHILD_CPUS
        or source_protocol.get("bag_folds") != 8
        or source_protocol.get("bag_sets") != 1
        or source_protocol.get("fold_fitting_strategy") != "sequential_local"
        or source_protocol.get("chimera_source")
        != {
            "exact_git_commit": CHIMERABOOST_COMMIT,
            "hidden_import_warmup": "disabled",
            "version": "0.14.1",
        }
        or source_protocol.get("catboost_source") != {"version": CATBOOST_VERSION}
        or primary.get("coordinates") != expected_coordinates
        or primary.get("expected_jobs") != 117
    ):
        raise RuntimeError("reused source protocol does not match the shootout")

    source_manifest_bytes = artifact_payloads[
        "tabarena_regression_same_machine_run_manifest.json"
    ]
    source_validation = hardened._as_mapping(
        source_attestation.get("validation"), "source completion validation"
    )
    if (
        source_attestation.get("manifest_sha256") != _sha256(source_manifest_bytes)
        or source_attestation.get("result_count") != 135
        or source_attestation.get("expected_child_fits") != 1_080
        or source_validation.get("failure_count") != 0
        or source_validation.get("imputation_count") != 0
        or source_validation.get("known_deadline_hit_count") != 0
        or source_validation.get("known_time_limit_stop_count") != 0
        or source_validation.get("resource_allocation")
        != {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        }
    ):
        raise RuntimeError("reused source completion is not admissible")

    summary_provenance = hardened._as_mapping(
        source_summary.get("provenance"), "source summary provenance"
    )
    if (
        summary_provenance.get("manifest_sha256")
        != SOURCE_ARTIFACTS["tabarena_regression_same_machine_run_manifest.json"]
        or summary_provenance.get("attestation_sha256")
        != SOURCE_ARTIFACTS[
            "tabarena_regression_same_machine_completion_attestation.json"
        ]
        or summary_provenance.get("chimeraboost_git_head") != CHIMERABOOST_COMMIT
        or summary_provenance.get("catboost_version") != CATBOOST_VERSION
        or source_summary.get("counts", {}).get("primary_coordinates") != 39
    ):
        raise RuntimeError("reused source summary does not bind its inputs")

    source_provenance = hardened._as_mapping(
        source_manifest.get("source"), "source provenance"
    )
    shootout_provenance = hardened._as_mapping(
        manifest.get("source"), "shootout provenance"
    )
    tabarena_provenance = _validate_reused_tabarena_provenance(
        source_provenance, shootout_provenance
    )

    # The new run must share the source campaign's environment, hardware, and
    # complete common dependency lock. Only the two live comparator packages
    # may be absent from the DarkoFit-only execution environment.
    old_runtime = hardened._as_mapping(
        source_manifest.get("runtime"), "source runtime"
    )
    new_runtime = hardened._as_mapping(manifest.get("runtime"), "shootout runtime")
    for field in (
        "python_version",
        "platform",
        "machine",
        "environment",
        "hardware",
    ):
        if new_runtime.get(field) != old_runtime.get(field):
            raise RuntimeError(f"shootout runtime differs from reused source: {field}")
    old_packages = hardened._as_mapping(old_runtime.get("packages"), "source packages")
    new_packages = hardened._as_mapping(
        new_runtime.get("packages"), "shootout packages"
    )
    dependency_lock = _validate_common_dependency_lock(old_packages, new_packages)

    source_files = hardened._as_mapping(
        source_provenance.get("files"),
        "source file hashes",
    )
    adapter_hashes = {}
    for relative in (
        "benchmarks/tabarena_adapter.py",
        "benchmarks/tabarena_screen_adapters.py",
    ):
        recorded = hardened._as_mapping(
            source_files.get(relative), f"source hash for {relative}"
        )
        payload = _read_stable(repository / relative, relative)
        digest = _sha256(payload)
        if recorded.get("sha256") != digest:
            raise RuntimeError(
                f"base adapter changed since reused evidence: {relative}"
            )
        adapter_hashes[relative] = digest

    rows = list(
        csv.DictReader(
            io.StringIO(
                artifact_payloads[
                    "tabarena_regression_same_machine_primary_paired_splits.csv"
                ].decode("utf-8")
            )
        )
    )
    comparator_rows = _parse_reused_comparator_rows(rows)
    diagnostics = {
        "contract": REUSED_EVIDENCE_CONTRACT,
        "source_artifact_count": len(SOURCE_ARTIFACTS),
        "source_primary_csv_rows": len(rows),
        "reused_coordinates": len(comparator_rows),
        "darkofit_subtree": current_tree,
        "base_adapter_sha256": adapter_hashes,
        "runtime_lock_verified": True,
        "tabarena_provenance": tabarena_provenance,
        "dependency_lock": dependency_lock,
    }
    return comparator_rows, diagnostics


def _parse_reused_comparator_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int, int], dict[str, float]]:
    if len(rows) != 117:
        raise RuntimeError("reused primary split artifact must contain 117 rows")
    grouped: dict[
        tuple[str, int, int], dict[str, Mapping[str, str]]
    ] = defaultdict(dict)
    expected_codes = {"D/M", "D/C", "M/C"}
    expected_arms = {
        "D/M": ("darkofit_product_default", "chimeraboost_0_14_1_default"),
        "D/C": ("darkofit_product_default", "catboost_1_2_10_default"),
        "M/C": ("chimeraboost_0_14_1_default", "catboost_1_2_10_default"),
    }
    for row in rows:
        if (
            row.get("panel") != "primary"
            or row.get("contrast_code") not in expected_codes
        ):
            raise RuntimeError(
                "reused split row is outside the primary comparator panel"
            )
        dataset = row.get("dataset")
        repeat = _csv_int(row.get("repeat"), "reused repeat")
        fold = _csv_int(row.get("fold"), "reused fold")
        key = (str(dataset), repeat, fold)
        if key not in set(campaign.expected_coordinates()):
            raise RuntimeError(f"reused split coordinate is not frozen: {key}")
        code = str(row["contrast_code"])
        if code in grouped[key]:
            raise RuntimeError(f"duplicate reused contrast: {key} {code}")
        if (
            _csv_int(row.get("task_id"), "reused task id")
            != campaign.TASKS[key[0]]
            or _csv_int(row.get("registered_fold"), "reused registered fold")
            != 3 * repeat + fold
            or (row.get("numerator_arm"), row.get("denominator_arm"))
            != expected_arms[code]
        ):
            raise RuntimeError(f"reused contrast metadata changed: {key} {code}")
        grouped[key][code] = row
    if set(grouped) != set(campaign.expected_coordinates()) or any(
        set(group) != expected_codes for group in grouped.values()
    ):
        raise RuntimeError("reused comparator grid is incomplete")

    output: dict[tuple[str, int, int], dict[str, float]] = {}
    for key, group in grouped.items():
        dm, dc, mc = group["D/M"], group["D/C"], group["M/C"]
        values: dict[str, float] = {}
        for metric in QUALITY_METRICS:
            d_dm = _positive(dm[f"numerator_{metric}"], f"{key} P {metric}")
            m_dm = _positive(dm[f"denominator_{metric}"], f"{key} M {metric}")
            d_dc = _positive(dc[f"numerator_{metric}"], f"{key} P {metric}")
            c_dc = _positive(dc[f"denominator_{metric}"], f"{key} C {metric}")
            m_mc = _positive(mc[f"numerator_{metric}"], f"{key} M {metric}")
            c_mc = _positive(mc[f"denominator_{metric}"], f"{key} C {metric}")
            if d_dm != d_dc or m_dm != m_mc or c_dc != c_mc:
                raise RuntimeError(f"reused comparator values disagree at {key}")
            values[f"P_{metric}"] = d_dm
            values[f"M_{metric}"] = m_dm
            values[f"C_{metric}"] = c_dc
        output[key] = values
    return output


def _artifact_bytes(
    input_dir: Path, relative: str, metadata: Mapping[str, Any], field: str
) -> bytes:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path == Path(".")
    ):
        raise RuntimeError(f"unsafe attested path: {relative!r}")
    raw_path = _confined_regular_path(input_dir, relative_path, field)
    payload = _read_stable(raw_path, field)
    resolved = raw_path.resolve(strict=True)
    try:
        resolved.relative_to(input_dir.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"attested path escapes campaign: {relative}") from exc
    if (
        len(payload) != _exact_int(metadata.get("size_bytes"), f"{field} size")
        or _sha256(payload) != metadata.get("sha256")
    ):
        raise RuntimeError(f"{field} does not match its attestation")
    return payload


def _confined_regular_path(input_dir: Path, relative: Path, field: str) -> Path:
    """Resolve a regular campaign file without following directory symlinks."""
    try:
        canonical_input = input_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"campaign directory does not resolve for {field}") from exc
    if not canonical_input.is_dir():
        raise RuntimeError(f"campaign root is not a directory for {field}")
    raw_path = canonical_input / relative
    try:
        resolved = raw_path.resolve(strict=True)
        resolved.relative_to(canonical_input)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{field} path escapes campaign: {relative}") from exc

    cursor = canonical_input
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise RuntimeError(f"could not inspect parent of {field}: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{field} has a symbolic-link parent: {cursor}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{field} has a non-directory parent: {cursor}")
    return raw_path


def protocol_sha256() -> str:
    return _sha256(_canonical_json(campaign.frozen_protocol()))


def _verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1].resolve(strict=True)
    source = dict(hardened._as_mapping(manifest.get("source"), "manifest source"))
    if set(source) != {
        "repository",
        "git_head",
        "git_tree",
        "relevant_status",
        "files",
        "darkofit_import",
        "tabarena",
    } or source.get("relevant_status") != "":
        raise RuntimeError("run manifest source provenance is incomplete or dirty")
    recorded_repository = hardened._manifest_path(
        source.get("repository"), "recorded shootout repository"
    )
    if recorded_repository != repository:
        raise RuntimeError("executing analyzer repository does not match the run")
    files = hardened._as_mapping(source.get("files"), "manifest source files")
    if set(files) != {str(path) for path in campaign.SOURCE_FILES}:
        raise RuntimeError("manifest source file set is not exact")
    for relative in campaign.SOURCE_FILES:
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RuntimeError(f"source file escapes repository: {relative}") from exc
        metadata = hardened._as_mapping(
            files[str(relative)], f"source metadata for {relative}"
        )
        if set(metadata) != {"sha256", "git_blob"}:
            raise RuntimeError(f"source metadata is incomplete for {relative}")
        payload = _read_stable(path, f"executing source {relative}")
        if metadata.get("sha256") != _sha256(payload):
            raise RuntimeError(f"source SHA-256 mismatch for {relative}")
        if metadata.get("git_blob") != hardened._git_hash_payload(
            repository, payload, str(relative)
        ):
            raise RuntimeError(f"source Git-blob mismatch for {relative}")
    head = _git(repository, ["rev-parse", "HEAD"], "Git HEAD")
    tree = _git(repository, ["rev-parse", "HEAD^{tree}"], "Git tree")
    if source.get("git_head") != head or source.get("git_tree") != tree:
        raise RuntimeError("executing Git revision does not match the shootout")
    changes = hardened._repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError(
            "executing shootout repository has dirty or unrecorded code: "
            + ", ".join(changes)
        )
    hardened._verify_dependency_provenance(
        source.get("darkofit_import"),
        "darkofit",
        input_dir,
        required_repository=repository,
    )
    hardened._verify_dependency_provenance(
        source.get("tabarena"), "tabarena", input_dir
    )
    if manifest.get("runtime") != campaign.screen.collect_runtime_provenance():
        raise RuntimeError("analysis runtime/hardware does not match the shootout")
    return {
        "executing_repository": str(repository),
        "executing_git_head": head,
        "executing_git_tree": tree,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
    }


def _attested_json_artifact(
    input_dir: Path,
    attestation: Mapping[str, Any],
    *,
    field: str,
    filename: str,
    required: bool,
) -> tuple[Any | None, str | None]:
    metadata = attestation.get(field)
    if metadata is None:
        if required:
            raise RuntimeError(f"required attested artifact is missing: {field}")
        return None, None
    metadata = hardened._as_mapping(metadata, field)
    if set(metadata) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"{field} metadata fields are not exact")
    if metadata.get("path") != filename:
        raise RuntimeError(f"{field} path does not match the frozen filename")
    payload = _artifact_bytes(input_dir, filename, metadata, field)
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not finite valid JSON") from exc
    return value, _sha256(payload)


def _validate_optional_artifact_presence(
    input_dir: Path,
    attestation: Mapping[str, Any],
    *,
    field: str,
    filename: str,
) -> None:
    path = input_dir / filename
    try:
        path.lstat()
        present = True
    except FileNotFoundError:
        present = False
    except OSError as exc:
        raise RuntimeError(f"could not inspect optional artifact: {path}") from exc
    if present != (attestation.get(field) is not None):
        raise RuntimeError(
            f"optional artifact presence does not match attestation: {field}"
        )


def _validate_preflight_artifact(
    value: Any, *, execution_mode: str, input_dir: Path
) -> dict[str, Any]:
    report = dict(hardened._as_mapping(value, "preflight report"))
    campaign.validate_preflight_attestation(report, input_dir)
    evaluator = getattr(campaign, "evaluate_preflight", None)
    if evaluator is None:
        raise RuntimeError("runner does not expose its frozen preflight validator")
    decision = evaluator(report)
    if (
        report.get("schema_version") != 1
        or report.get("kind") != campaign.CAMPAIGN_KIND + "_preflight"
        or report.get("protocol_sha256") != protocol_sha256()
        or report.get("wave_schedule_sha256")
        != campaign.wave_schedule_sha256()
        or report.get("decision") != decision
    ):
        raise RuntimeError("preflight report does not bind the frozen protocol")
    passed = bool(decision.get("passed"))
    if execution_mode == "concurrent" and not passed:
        raise RuntimeError("concurrent execution lacks a passing preflight")
    if execution_mode == "sequential_fallback" and passed:
        raise RuntimeError("sequential fallback is not justified by preflight")
    return dict(decision)


def _validate_concurrency_artifact(
    value: Any, *, execution_mode: str, input_dir: Path
) -> None:
    campaign.validate_concurrency_history(
        value, execution_mode=execution_mode, output_dir=input_dir
    )


def _validate_concurrency_result_bindings(
    concurrency: Any,
    artifacts: Mapping[str, Any],
    input_dir: Path,
) -> None:
    """Bind each worker-reported result to one completed raw artifact."""
    root = input_dir.resolve(strict=True)
    history = hardened._as_mapping(concurrency, "concurrency history")
    entries = history.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("concurrency history entries must be a list")
    observed = set()
    for raw_entry in entries:
        entry = hardened._as_mapping(raw_entry, "concurrency entry")
        reports = entry.get("reports")
        if not isinstance(reports, list):
            raise RuntimeError("concurrency entry reports must be a list")
        for raw_report in reports:
            report = hardened._as_mapping(raw_report, "concurrency report")
            path_value = report.get("result_path")
            if not isinstance(path_value, str) or not path_value:
                raise RuntimeError("concurrency result path must be a string")
            path = Path(path_value)
            try:
                resolved = path.resolve(strict=True)
                relative_path = resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise RuntimeError("concurrency result path escapes campaign") from exc
            relative = str(relative_path)
            if path != root / relative_path or relative in observed:
                raise RuntimeError(
                    "concurrency result paths are noncanonical or duplicated"
                )
            metadata = hardened._as_mapping(
                artifacts.get(relative), f"result artifact {relative}"
            )
            if (
                report.get("result_sha256") != metadata.get("sha256")
                or report.get("result_size_bytes")
                != _exact_int(
                    metadata.get("size_bytes"), f"result artifact {relative} size"
                )
            ):
                raise RuntimeError(
                    "concurrency report does not bind the completed result artifact"
                )
            observed.add(relative)
    if observed != set(artifacts):
        raise RuntimeError(
            "concurrency reports and completed result artifacts are not bijective"
        )


def _validate_warmup_sessions(
    value: Any, *, execution_mode: str, output_dir: Path
) -> set[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("production warmup history must contain a session")
    latest_identities: set[tuple[int, int]] = set()
    for session_index, raw_session in enumerate(value):
        session = hardened._as_mapping(
            raw_session, f"warmup session {session_index}"
        )
        if set(session) != {
            "completed_at_utc",
            "execution_mode",
            "worker_ready",
            "worker_warmup",
        }:
            raise RuntimeError("production warmup session fields are incomplete")
        if (
            not isinstance(session["completed_at_utc"], str)
            or not session["completed_at_utc"]
            or session["execution_mode"] != execution_mode
        ):
            raise RuntimeError("production warmup session header does not match")
        ready = session["worker_ready"]
        records = session["worker_warmup"]
        if (
            not isinstance(ready, list)
            or not isinstance(records, list)
            or len(ready) != campaign.WORKER_COUNT
            or len(records) != campaign.WORKER_COUNT
        ):
            raise RuntimeError("production warmup must cover both workers")
        ready_by_slot = {}
        session_root: Path | None = None
        scratch_base = output_dir / "worker_scratch"
        for raw in ready:
            item = hardened._as_mapping(raw, "worker readiness")
            if set(item) != {
                "type",
                "slot",
                "pid",
                "child_cpus",
                "start_method",
                "scratch_root",
            }:
                raise RuntimeError("worker readiness fields are incomplete")
            slot = _exact_int(item["slot"], "ready worker slot")
            pid = _exact_int(item["pid"], "ready worker pid")
            scratch_value = item["scratch_root"]
            if not isinstance(scratch_value, str) or not scratch_value:
                raise RuntimeError("worker scratch root must be a canonical path")
            scratch = Path(scratch_value)
            if (
                not scratch.is_absolute()
                or ".." in scratch.parts
                or str(scratch) != scratch_value
            ):
                raise RuntimeError("worker scratch root must be a canonical path")
            try:
                relative = scratch.relative_to(scratch_base)
            except ValueError as exc:
                raise RuntimeError("worker scratch root escapes campaign") from exc
            if len(relative.parts) != 2:
                raise RuntimeError("worker scratch root has the wrong depth")
            session_name, worker_name = relative.parts
            session_parts = session_name.split("-")
            if (
                len(session_parts) != 3
                or session_parts[0] != "session"
                or not all(part.isdigit() for part in session_parts[1:])
                or any(int(part) <= 0 for part in session_parts[1:])
                or worker_name != f"worker-{slot}"
            ):
                raise RuntimeError("worker scratch root is not session-scoped")
            current_session_root = scratch.parent
            if session_root is None:
                session_root = current_session_root
            elif current_session_root != session_root:
                raise RuntimeError("workers do not share one private session root")
            for private_dir in (scratch_base, current_session_root, scratch):
                try:
                    metadata = private_dir.lstat()
                except OSError as exc:
                    raise RuntimeError(
                        "worker scratch directory does not exist"
                    ) from exc
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError("worker scratch directory must not be a symlink")
            try:
                canonical_scratch = scratch.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError("worker scratch root does not resolve") from exc
            if (
                slot not in range(campaign.WORKER_COUNT)
                or slot in ready_by_slot
                or pid <= 0
                or item["type"] != "ready"
                or item["child_cpus"] != campaign.EXPECTED_CHILD_CPUS
                or item["start_method"] != "spawn"
                or scratch != canonical_scratch
            ):
                raise RuntimeError("worker readiness does not match the shootout")
            ready_by_slot[slot] = item
        if (
            len({item["pid"] for item in ready_by_slot.values()})
            != campaign.WORKER_COUNT
        ):
            raise RuntimeError("production worker PIDs are not distinct")
        normalized = []
        observed_slots = set()
        session_identities: set[tuple[int, int]] = set()
        for raw in records:
            record = dict(hardened._as_mapping(raw, "worker warmup record"))
            if set(record) != {
                "completed_at_utc",
                "pid",
                "worker_slot",
                "warmup",
            }:
                raise RuntimeError("worker warmup record fields are incomplete")
            slot = _exact_int(record.pop("worker_slot"), "warmup worker slot")
            if (
                slot not in ready_by_slot
                or slot in observed_slots
                or record.get("pid") != ready_by_slot[slot]["pid"]
            ):
                raise RuntimeError("worker warmup identity does not match readiness")
            observed_slots.add(slot)
            session_identities.add(
                (slot, _exact_int(record["pid"], "warmup worker pid"))
            )
            normalized.append(record)
        campaign.screen._validate_followon_warmup_history(
            normalized, expected_thread_count=campaign.EXPECTED_CHILD_CPUS
        )
        if observed_slots != set(range(campaign.WORKER_COUNT)):
            raise RuntimeError("worker warmup slots are incomplete")
        latest_identities = session_identities
    return latest_identities


def _validate_production_session_binding(
    warmup_history: Any,
    *,
    latest_warmup_identities: set[tuple[int, int]],
    concurrency: Any,
    resume_history: Any | None,
) -> None:
    """Bind all measured waves to the newest persistent-worker session."""
    if not isinstance(warmup_history, list) or not warmup_history:
        raise RuntimeError("production warmup session history is incomplete")
    if resume_history is None:
        resume_count = 0
    elif isinstance(resume_history, list):
        resume_count = len(resume_history)
    else:
        raise RuntimeError("production resume history must be a list")
    # An interrupted attempt can stop before it records a warmup session, so
    # there may be fewer sessions than invocations. There can never be more
    # than the initial attempt plus one session per resume record.
    if len(warmup_history) > resume_count + 1:
        raise RuntimeError("production warmup count exceeds the resume history")

    history = hardened._as_mapping(concurrency, "concurrency history")
    entries = history.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("concurrency history entries must be a list")
    concurrency_identities = {
        (
            _exact_int(report.get("slot"), "concurrency worker slot"),
            _exact_int(report.get("pid"), "concurrency worker pid"),
        )
        for entry in entries
        for report in hardened._as_mapping(entry, "concurrency entry").get(
            "reports", []
        )
    }
    if concurrency_identities != latest_warmup_identities:
        raise RuntimeError(
            "measured workers do not exactly match the newest warmup session"
        )


def _validate_shootout_resume_history(value: Any, output_dir: Path) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("resume history must contain at least one record")
    all_waves = set(range(campaign.EXPECTED_WAVES))
    schedule = campaign.expected_wave_schedule()
    for record_index, raw_record in enumerate(value):
        record = hardened._as_mapping(raw_record, f"resume record {record_index}")
        if set(record) != {
            "resumed_at_utc",
            "pid",
            "wave_schedule_sha256",
            "reusable_wave_indices",
            "pending_wave_indices",
            "invalidated_waves",
            "archived_campaign_artifacts",
        }:
            raise RuntimeError("resume record fields are incomplete")
        if (
            not isinstance(record["resumed_at_utc"], str)
            or not record["resumed_at_utc"]
            or _exact_int(record["pid"], "resume pid") <= 0
            or record["wave_schedule_sha256"]
            != campaign.wave_schedule_sha256()
        ):
            raise RuntimeError("resume record header does not match")
        reusable = record["reusable_wave_indices"]
        pending = record["pending_wave_indices"]
        if not isinstance(reusable, list) or not isinstance(pending, list):
            raise RuntimeError("resume wave sets must be lists")
        if reusable != [] or pending != list(range(campaign.EXPECTED_WAVES)):
            raise RuntimeError(
                "resume must archive every prior-process pickle and rerun all waves"
            )
        reusable_set = {
            _exact_int(item, "reusable wave index") for item in reusable
        }
        pending_set = {_exact_int(item, "pending wave index") for item in pending}
        if (
            len(reusable_set) != len(reusable)
            or len(pending_set) != len(pending)
            or reusable_set & pending_set
            or reusable_set | pending_set != all_waves
        ):
            raise RuntimeError("resume wave partition is not exact")
        invalidated = record["invalidated_waves"]
        if not isinstance(invalidated, list):
            raise RuntimeError("invalidated waves must be a list")
        invalidated_indices = set()
        archived_paths = set()
        for raw_wave in invalidated:
            wave = hardened._as_mapping(raw_wave, "invalidated wave")
            if set(wave) != {"wave_index", "members"}:
                raise RuntimeError("invalidated wave fields are incomplete")
            wave_index = _exact_int(wave["wave_index"], "invalidated wave index")
            members = wave["members"]
            if (
                wave_index not in pending_set
                or wave_index in invalidated_indices
                or not isinstance(members, list)
                or not members
                or len(members) > 2
            ):
                raise RuntimeError("invalidated wave does not match pending work")
            invalidated_indices.add(wave_index)
            expected_keys = {
                tuple(
                    item["key"][name]
                    for name in ("dataset", "repeat", "fold", "arm")
                )
                for item in schedule[wave_index]["jobs"]
            }
            member_keys = set()
            for raw_member in members:
                member = hardened._as_mapping(raw_member, "invalidated member")
                if set(member) != {"key", "status", "path"}:
                    raise RuntimeError("invalidated member fields are incomplete")
                member_key = hardened._as_mapping(member["key"], "member key")
                key = campaign._key_tuple(member_key)
                if (
                    key not in expected_keys
                    or key in member_keys
                    or member_key != campaign._key_payload(key)
                    or member["status"]
                    not in {
                        "valid",
                        "missing",
                        "unreadable",
                        "not_a_regular_file",
                        "mismatched",
                        "incomplete_or_mismatched",
                        "unattested_or_changed",
                        "prior_process_pickle_archived",
                    }
                ):
                    raise RuntimeError("invalidated member belongs to the wrong wave")
                path = member["path"]
                if not isinstance(path, str):
                    raise RuntimeError("invalidated result path must be a string")
                if path in archived_paths:
                    raise RuntimeError("resume archive path is duplicated")
                member_keys.add(key)
                archived_paths.add(path)
                _validate_archived_path(
                    output_dir, path, "invalidated result"
                )
        archived = record["archived_campaign_artifacts"]
        if not isinstance(archived, list):
            raise RuntimeError("archived campaign artifact paths must be a list")
        for relative in archived:
            if not isinstance(relative, str):
                raise RuntimeError("archived campaign artifact path must be a string")
            if relative in archived_paths:
                raise RuntimeError("resume archive path is duplicated")
            archived_paths.add(relative)
            _validate_archived_path(output_dir, relative, "archived campaign artifact")


def _validate_archived_path(output_dir: Path, value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} path must be a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "resume_invalidated"
    ):
        raise RuntimeError(f"{field} path is unsafe")
    path = _confined_regular_path(output_dir, relative, field)
    _read_stable(path, field)


def verify_campaign_integrity(
    input_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int, int], dict[str, float]],
    list[Path],
]:
    input_dir = input_dir.resolve(strict=True)
    manifest_path = input_dir / campaign.screen.MANIFEST_FILENAME
    attestation_path = input_dir / campaign.screen.COMPLETION_ATTESTATION_FILENAME
    manifest, manifest_bytes = _read_json(manifest_path, "run manifest")
    protocol = campaign.frozen_protocol()
    protocol_digest = protocol_sha256()
    execution_mode = manifest.get("execution_mode")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != campaign.CAMPAIGN_KIND
        or Path(str(manifest.get("output_dir", ""))).resolve() != input_dir
        or manifest.get("protocol") != protocol
        or manifest.get("protocol_sha256") != protocol_digest
        or manifest.get("time_limit_seconds") != campaign.TIME_LIMIT_SECONDS
        or manifest.get("resolved_child_num_cpus") != campaign.EXPECTED_CHILD_CPUS
        or manifest.get("wave_schedule_sha256")
        != campaign.wave_schedule_sha256()
        or execution_mode not in {"concurrent", "sequential_fallback"}
        or manifest.get("start_method") != "spawn"
        or manifest.get("worker_count") != campaign.WORKER_COUNT
        or manifest.get("wave_count") != campaign.EXPECTED_WAVES
        or manifest.get("execution_grid_sha256")
        != campaign.execution_grid_sha256(str(execution_mode))
    ):
        raise RuntimeError("run manifest does not match the frozen shootout")
    grid_path = input_dir / campaign.WAVE_SCHEDULE_FILENAME
    grid, grid_bytes = _read_json(grid_path, "execution grid")
    if (
        grid != campaign.execution_grid_payload(str(execution_mode))
        or _sha256(_canonical_json(grid))
        != campaign.execution_grid_sha256(str(execution_mode))
        or manifest.get("execution_grid_artifact_sha256") != _sha256(grid_bytes)
    ):
        raise RuntimeError("execution grid does not match the frozen shootout")
    execution = _verify_execution_provenance(manifest, input_dir)
    repository = Path(__file__).resolve().parents[1]
    reused, reused_diagnostics = _verify_source_reuse_contract(repository, manifest)

    attestation, attestation_bytes = _read_json(
        attestation_path, "completion attestation"
    )
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind") != campaign.COMPLETION_KIND
        or attestation.get("result_count") != campaign.EXPECTED_JOBS
        or attestation.get("expected_result_count") != campaign.EXPECTED_JOBS
        or attestation.get("expected_child_fits") != campaign.EXPECTED_CHILD_FITS
        or attestation.get("expected_paired_comparisons")
        != campaign.EXPECTED_COORDINATES
        or attestation.get("protocol_sha256") != protocol_digest
        or attestation.get("wave_schedule_sha256")
        != campaign.wave_schedule_sha256()
        or attestation.get("execution_mode") != execution_mode
        or attestation.get("execution_grid_sha256")
        != manifest.get("execution_grid_sha256")
        or attestation.get("preflight_report_sha256")
        != manifest.get("preflight_report_sha256")
        or attestation.get("git_head") != manifest["source"]["git_head"]
        or attestation.get("manifest_sha256") != _sha256(manifest_bytes)
    ):
        raise RuntimeError("completion attestation does not match the shootout")
    campaign.validate_completed_owner_session(input_dir, attestation)
    owner_state_path = _confined_regular_path(
        input_dir,
        Path(campaign.OWNER_STATE_FILENAME),
        "owner session state",
    )
    owner_state_bytes = _read_stable(owner_state_path, "owner session state")
    owner_lock_path = _confined_regular_path(
        input_dir,
        Path(campaign.OWNER_LOCK_FILENAME),
        "owner lock",
    )
    owner_lock_bytes = _read_stable(owner_lock_path, "owner lock")
    if owner_lock_bytes:
        raise RuntimeError("owner lock contents are invalid")

    artifacts = hardened._as_mapping(
        attestation.get("result_artifacts"), "result artifacts"
    )
    if len(artifacts) != campaign.EXPECTED_JOBS:
        raise RuntimeError("attested result count does not match the shootout")
    observed = {
        str(path.relative_to(input_dir))
        for path in (input_dir / "experiments").rglob("results.pkl")
    }
    if observed != set(artifacts):
        raise RuntimeError("on-disk result set does not match the attestation")
    for relative, metadata in artifacts.items():
        if Path(relative).name != "results.pkl":
            raise RuntimeError("attested result has an unsafe filename")
        _artifact_bytes(
            input_dir,
            relative,
            hardened._as_mapping(metadata, f"result artifact {relative}"),
            f"result {relative}",
        )

    payload_metadata = hardened._as_mapping(
        attestation.get("analysis_payload_artifact"), "analysis payload artifact"
    )
    if (
        set(payload_metadata) != {"path", "sha256", "size_bytes"}
        or payload_metadata.get("path") != campaign.screen.ANALYSIS_PAYLOAD_FILENAME
    ):
        raise RuntimeError("analysis payload attestation is incomplete")
    payload_bytes = _artifact_bytes(
        input_dir,
        campaign.screen.ANALYSIS_PAYLOAD_FILENAME,
        payload_metadata,
        "safe analysis payload",
    )
    try:
        payload = dict(
            hardened._as_mapping(
                json.loads(
                    payload_bytes,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"nonfinite JSON constant: {token}")
                    ),
                ),
                "safe analysis payload",
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("safe analysis payload is not valid JSON") from exc
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != campaign.PAYLOAD_KIND
        or payload.get("protocol_sha256") != protocol_digest
        or payload.get("result_artifacts_sha256")
        != _sha256(_canonical_json(artifacts))
        or payload.get("wave_schedule_sha256") != campaign.wave_schedule_sha256()
        or payload.get("execution_grid_sha256")
        != manifest.get("execution_grid_sha256")
        or payload.get("preflight_report_sha256")
        != manifest.get("preflight_report_sha256")
    ):
        raise RuntimeError("safe analysis payload does not bind the shootout")
    outer_rows, child_rows = _validate_safe_payload(payload, artifacts, 18)
    payload["outer_rows"] = outer_rows
    payload["child_rows"] = child_rows

    preflight, preflight_digest = _attested_json_artifact(
        input_dir,
        attestation,
        field="preflight_report_artifact",
        filename=campaign.PREFLIGHT_REPORT_FILENAME,
        required=True,
    )
    preflight_decision = _validate_preflight_artifact(
        preflight, execution_mode=str(execution_mode), input_dir=input_dir
    )
    if (
        manifest.get("preflight_report_sha256") != preflight_digest
        or attestation.get("preflight_report_sha256", preflight_digest)
        != preflight_digest
    ):
        raise RuntimeError("preflight report digest does not bind the run")

    concurrency, concurrency_digest = _attested_json_artifact(
        input_dir,
        attestation,
        field="concurrency_history_artifact",
        filename=campaign.CONCURRENCY_HISTORY_FILENAME,
        required=True,
    )
    _validate_concurrency_artifact(
        concurrency, execution_mode=str(execution_mode), input_dir=input_dir
    )
    _validate_concurrency_result_bindings(concurrency, artifacts, input_dir)
    warmup, warmup_digest = _attested_json_artifact(
        input_dir,
        attestation,
        field="warmup_history_artifact",
        filename=campaign.screen.WARMUP_HISTORY_FILENAME,
        required=True,
    )
    latest_warmup_identities = _validate_warmup_sessions(
        warmup, execution_mode=str(execution_mode), output_dir=input_dir
    )
    _validate_optional_artifact_presence(
        input_dir,
        attestation,
        field="resume_history_artifact",
        filename=campaign.screen.RESUME_HISTORY_FILENAME,
    )
    resume, resume_digest = _attested_json_artifact(
        input_dir,
        attestation,
        field="resume_history_artifact",
        filename=campaign.screen.RESUME_HISTORY_FILENAME,
        required=False,
    )
    if resume is not None:
        _validate_shootout_resume_history(resume, input_dir)
    _validate_production_session_binding(
        warmup,
        latest_warmup_identities=latest_warmup_identities,
        concurrency=concurrency,
        resume_history=resume,
    )

    validation = hardened._as_mapping(
        attestation.get("validation"), "completion validation"
    )
    if (
        validation.get("result_count") != campaign.EXPECTED_JOBS
        or validation.get("child_fit_count") != campaign.EXPECTED_CHILD_FITS
        or validation.get("paired_comparison_count")
        != campaign.EXPECTED_COORDINATES
        or validation.get("native_representation_pair_count")
        != campaign.EXPECTED_PAIRED_CHILDREN
        or validation.get("resource_allocation")
        != {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        }
        or validation.get("stop_reason_counts", {}).get("time_limit", 0) != 0
    ):
        raise RuntimeError("completion validation does not match the shootout")

    protected = hardened._protected_campaign_paths(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    protected = [
        *protected,
        grid_path.resolve(strict=True),
        (input_dir / campaign.PREFLIGHT_REPORT_FILENAME).resolve(strict=True),
        (input_dir / campaign.CONCURRENCY_HISTORY_FILENAME).resolve(strict=True),
        owner_state_path.resolve(strict=True),
        owner_lock_path.resolve(strict=True),
    ]
    digests = {
        "manifest_sha256": _sha256(manifest_bytes),
        "attestation_sha256": _sha256(attestation_bytes),
        "analysis_payload_sha256": _sha256(payload_bytes),
        "execution_grid_artifact_sha256": _sha256(grid_bytes),
        "preflight_report_sha256": preflight_digest,
        "concurrency_history_sha256": concurrency_digest,
        "warmup_history_sha256": warmup_digest,
        "resume_history_sha256": resume_digest,
        "owner_state_sha256": _sha256(owner_state_bytes),
        "owner_lock_sha256": _sha256(owner_lock_bytes),
        "preflight_decision": preflight_decision,
        "reused_evidence": reused_diagnostics,
        **execution,
    }
    return manifest, attestation, payload, digests, reused, list(protected)


def _validate_safe_payload(
    payload: Mapping[str, Any], artifacts: Mapping[str, Any], child_cpus: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows = payload.get("outer_rows")
    child_rows = payload.get("child_rows")
    if not isinstance(outer_rows, list) or not isinstance(child_rows, list):
        raise RuntimeError("analysis payload rows must be lists")
    if len(outer_rows) != campaign.EXPECTED_JOBS:
        raise RuntimeError("analysis payload outer result count is wrong")
    if len(child_rows) != campaign.EXPECTED_CHILD_FITS:
        raise RuntimeError("analysis payload child result count is wrong")

    expected_outer_fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "test_rmse",
        "val_rmse",
        "train_time_s",
        "infer_time_s",
        "peak_memory_bytes",
        "framework",
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
        "source",
    }
    outer_index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    source_counts = Counter()
    for raw in outer_rows:
        row = dict(hardened._as_mapping(raw, "outer row"))
        if set(row) != expected_outer_fields:
            raise RuntimeError("safe outer row fields are not exact")
        key = (
            str(row["dataset"]),
            _exact_int(row["repeat"], "outer repeat"),
            _exact_int(row["fold"], "outer fold"),
            str(row["arm"]),
        )
        dataset, repeat, fold, arm = key
        if (
            key not in expected_outer_grid()
            or key in outer_index
            or row["task_id"] != campaign.TASKS[dataset]
            or row["registered_fold"] != 3 * repeat + fold
            or row["num_cpus"] != child_cpus
            or row["num_cpus_child"] != child_cpus
            or row["num_gpus"] != 0
            or row["num_gpus_child"] != 0
            or row["source"] not in artifacts
            or row["framework"] != f"DarkoFit_c1_screen_{arm}_BAG_L1"
        ):
            raise RuntimeError(f"safe outer row does not match shootout grid: {key}")
        campaign.screen._validate_result_source_binding(
            row["source"],
            dataset=dataset,
            repeat=repeat,
            fold=fold,
            arm=arm,
        )
        for metric in NEW_METRICS:
            _positive(row[metric], f"outer {metric}")
        source_counts[row["source"]] += 1
        outer_index[key] = row
    if (
        set(outer_index) != expected_outer_grid()
        or set(source_counts) != set(artifacts)
        or any(count != 1 for count in source_counts.values())
    ):
        raise RuntimeError("safe outer rows do not bind one-to-one to raw results")

    expected_child_fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "child",
        "child_fold",
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "early_stopping_rounds",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "tree_mode_selection",
        "stop_reason",
        "deadline_hit",
        "wall_clock_elapsed_seconds",
        "child_features",
        "representation",
        "refit_params",
        "num_cpus",
        "num_gpus",
        "source",
    }
    child_index = set()
    child_counts = Counter()
    normalized = []
    for raw in child_rows:
        row = dict(hardened._as_mapping(raw, "child row"))
        if set(row) != expected_child_fields:
            raise RuntimeError("safe child row fields are not exact")
        dataset = str(row["dataset"])
        repeat = _exact_int(row["repeat"], "child repeat")
        fold = _exact_int(row["fold"], "child fold coordinate")
        arm = str(row["arm"])
        child_fold = _exact_int(row["child_fold"], "child fold")
        outer_key = (dataset, repeat, fold, arm)
        key = (*outer_key, child_fold)
        if (
            key not in expected_child_grid()
            or key in child_index
            or row["task_id"] != campaign.TASKS[dataset]
            or row["registered_fold"] != 3 * repeat + fold
            or row["child"] != f"S1F{child_fold + 1}"
            or row["source"] != outer_index[outer_key]["source"]
            or row["num_cpus"] != child_cpus
            or row["num_gpus"] != 0
        ):
            raise RuntimeError(f"safe child row does not match shootout grid: {key}")
        requested = _exact_int(row["iterations_requested"], "iterations requested")
        attempted = _exact_int(row["iterations_attempted"], "iterations attempted")
        completed = _exact_int(row["rounds_completed"], "rounds completed")
        retained = _exact_int(row["rounds_retained"], "rounds retained")
        best = _exact_int(row["best_iteration"], "best iteration")
        if requested != 10_000 or not (
            0 <= retained == best <= completed <= attempted <= requested
        ):
            raise RuntimeError("safe child round counters are inconsistent")
        if float(row["resolved_learning_rate"]) != 0.1:
            raise RuntimeError("safe child learning-rate policy changed")
        campaign.screen._validate_early_stopping_rounds(
            row["early_stopping_rounds"], field="safe child early stopping"
        )
        requested_mode = campaign.screen.ARM_SPECS[arm]["config"]["tree_mode"]
        selected_mode = row["selected_tree_mode"]
        if (
            row["requested_tree_mode"] != requested_mode
            or selected_mode not in {"catboost", "lightgbm", "hybrid"}
            or (requested_mode != "auto" and selected_mode != requested_mode)
            or row["linear_residual_active"] is not False
            or row["selected_lane"] != "boosting"
        ):
            raise RuntimeError("safe child fitted lane does not match its arm")
        if arm == "auto":
            campaign.screen._validate_tree_mode_selection(
                row["tree_mode_selection"],
                expected_iterations=10_000,
                selected_tree_mode=selected_mode,
                deadline_hit=row["deadline_hit"],
                top_level=row,
                field="safe A10 child tree selection",
            )
        elif row["tree_mode_selection"] is not None:
            raise RuntimeError("B10 child carries automatic selection metadata")
        if row["stop_reason"] not in campaign.screen.VALID_STOP_REASONS:
            raise RuntimeError("safe child stop reason is invalid")
        campaign.screen.hardened.validate_stop_reason_causality(
            row["stop_reason"],
            requested=requested,
            attempted=attempted,
            completed=completed,
            field="safe child",
        )
        if row["stop_reason"] == "time_limit" or row["deadline_hit"] is not False:
            raise RuntimeError("shootout contains a deadline-hit child")
        _nonnegative(row["wall_clock_elapsed_seconds"], "child wall time")
        campaign.screen._feature_schema_sha256(
            row["child_features"], "safe child external features"
        )
        campaign.screen._validate_representation_metadata(
            row["representation"],
            arm=arm,
            dataset=dataset,
            field="safe child representation",
            child_features=row["child_features"],
        )
        campaign.screen._validate_refit_params(
            row["refit_params"],
            expected_iterations=best,
            selected_tree_mode=selected_mode,
            field="safe child refit params",
        )
        child_index.add(key)
        child_counts[outer_key] += 1
        normalized.append(row)
    if set(child_index) != expected_child_grid() or any(
        child_counts[key] != 8 for key in outer_index
    ):
        raise RuntimeError("safe child grid is incomplete")
    campaign.screen.validate_native_representation_pairs(normalized)
    return list(outer_index.values()), normalized


def pair_outer_rows(
    outer_rows: Sequence[Mapping[str, Any]],
    reused: Mapping[tuple[str, int, int], Mapping[str, float]],
) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], int(row["repeat"]), int(row["fold"]), row["arm"]): row
        for row in outer_rows
    }
    if set(index) != expected_outer_grid() or set(reused) != set(
        campaign.expected_coordinates()
    ):
        raise RuntimeError("outer or reused comparison grid is incomplete")
    paired = []
    for dataset, repeat, fold in campaign.expected_coordinates():
        b10 = index[(dataset, repeat, fold, "baseline")]
        a10 = index[(dataset, repeat, fold, "auto")]
        source = reused[(dataset, repeat, fold)]
        row: dict[str, Any] = {
            "dataset": dataset,
            "task_id": campaign.TASKS[dataset],
            "repeat": repeat,
            "fold": fold,
            "registered_fold": 3 * repeat + fold,
        }
        for metric in NEW_METRICS:
            row[f"B10_{metric}"] = _positive(b10[metric], f"B10 {metric}")
            row[f"A10_{metric}"] = _positive(a10[metric], f"A10 {metric}")
        for metric in QUALITY_METRICS:
            for arm in ("P", "M", "C"):
                row[f"{arm}_{metric}"] = _positive(
                    source[f"{arm}_{metric}"], f"{arm} {metric}"
                )
            for code, numerator, denominator in CONTRASTS:
                ratio = row[f"{numerator}_{metric}"] / row[f"{denominator}_{metric}"]
                if not math.isfinite(ratio) or ratio <= 0.0:
                    raise RuntimeError(f"nonfinite {code} {metric} ratio")
                prefix = code.lower().replace("/", "_over_")
                row[f"{prefix}_{metric}_ratio"] = ratio
                row[f"{prefix}_{metric}_log_ratio"] = math.log(ratio)
                row[f"{prefix}_{metric}_pct"] = 100.0 * (ratio - 1.0)
            if not math.isclose(
                row[f"a10_over_p_{metric}_ratio"],
                row[f"a10_over_b10_{metric}_ratio"]
                * row[f"b10_over_p_{metric}_ratio"],
                rel_tol=2e-15,
                abs_tol=0.0,
            ):
                raise RuntimeError("attribution chain does not reconcile")
        paired.append(row)
    if len(paired) != campaign.EXPECTED_COORDINATES:
        raise RuntimeError("paired outer comparison count is wrong")
    return paired


def pair_child_rows(child_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row["dataset"],
            int(row["repeat"]),
            int(row["fold"]),
            row["arm"],
            int(row["child_fold"]),
        ): row
        for row in child_rows
    }
    if len(index) != campaign.EXPECTED_CHILD_FITS:
        raise RuntimeError("child metadata is incomplete or duplicated")
    paired = []
    for dataset, repeat, fold in campaign.expected_coordinates():
        for child_fold in range(8):
            b10 = index[(dataset, repeat, fold, "baseline", child_fold)]
            a10 = index[(dataset, repeat, fold, "auto", child_fold)]
            paired.append(
                {
                    "dataset": dataset,
                    "task_id": campaign.TASKS[dataset],
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": 3 * repeat + fold,
                    "child": f"S1F{child_fold + 1}",
                    "child_fold": child_fold,
                    "B10_best_iteration": b10["best_iteration"],
                    "A10_best_iteration": a10["best_iteration"],
                    "B10_iterations_requested": b10["iterations_requested"],
                    "A10_iterations_requested": a10["iterations_requested"],
                    "B10_iterations_attempted": b10["iterations_attempted"],
                    "A10_iterations_attempted": a10["iterations_attempted"],
                    "B10_rounds_completed": b10["rounds_completed"],
                    "A10_rounds_completed": a10["rounds_completed"],
                    "B10_rounds_retained": b10["rounds_retained"],
                    "A10_rounds_retained": a10["rounds_retained"],
                    "B10_resolved_learning_rate": b10["resolved_learning_rate"],
                    "A10_resolved_learning_rate": a10["resolved_learning_rate"],
                    "B10_stop_reason": b10["stop_reason"],
                    "A10_stop_reason": a10["stop_reason"],
                    "B10_selected_tree_mode": b10["selected_tree_mode"],
                    "A10_selected_tree_mode": a10["selected_tree_mode"],
                    "B10_selected_lane": b10["selected_lane"],
                    "A10_selected_lane": a10["selected_lane"],
                    "B10_deadline_hit": b10["deadline_hit"],
                    "A10_deadline_hit": a10["deadline_hit"],
                    "A10_candidate_count": a10["tree_mode_selection"][
                        "candidate_count"
                    ],
                    "A10_fitted_candidate_count": a10["tree_mode_selection"][
                        "fitted_candidate_count"
                    ],
                    "A10_selected_candidate_index": a10["tree_mode_selection"][
                        "selected_candidate_index"
                    ],
                    "B10_wall_clock_elapsed_seconds": b10[
                        "wall_clock_elapsed_seconds"
                    ],
                    "A10_wall_clock_elapsed_seconds": a10[
                        "wall_clock_elapsed_seconds"
                    ],
                }
            )
    if len(paired) != campaign.EXPECTED_PAIRED_CHILDREN:
        raise RuntimeError("paired child comparison count is wrong")
    return paired


def _ratio_prefix(code: str) -> str:
    return code.lower().replace("/", "_over_")


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _win_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "wins": sum(value < 1.0 for value in values),
        "losses": sum(value > 1.0 for value in values),
        "ties": sum(value == 1.0 for value in values),
    }


def analyze(
    paired_rows: Sequence[Mapping[str, Any]],
    paired_children: Sequence[Mapping[str, Any]],
    *,
    execution_mode: str = "concurrent",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(paired_rows) != 39 or len(paired_children) != 312:
        raise RuntimeError("analysis inputs do not match the frozen comparison counts")
    if execution_mode not in {"concurrent", "sequential_fallback"}:
        raise RuntimeError("analysis execution mode is invalid")
    per_dataset: list[dict[str, Any]] = []
    contrasts: dict[str, Any] = {}
    for code, _numerator, _denominator in CONTRASTS:
        prefix = _ratio_prefix(code)
        contrast_summary: dict[str, Any] = {}
        for metric in QUALITY_METRICS:
            field = f"{prefix}_{metric}_ratio"
            dataset_logs = {}
            for dataset in campaign.TASKS:
                rows = [row for row in paired_rows if row["dataset"] == dataset]
                if len(rows) != 3:
                    raise RuntimeError(f"dataset split grid is incomplete: {dataset}")
                dataset_logs[dataset] = math.fsum(
                    math.log(_positive(row[field], field)) for row in rows
                ) / 3.0
            equal_dataset_log = math.fsum(dataset_logs.values()) / len(dataset_logs)
            split_ratios = [float(row[field]) for row in paired_rows]
            dataset_ratios = [math.exp(dataset_logs[name]) for name in campaign.TASKS]
            worst_split = max(paired_rows, key=lambda row: float(row[field]))
            worst_dataset = max(campaign.TASKS, key=lambda name: dataset_logs[name])
            contrast_summary[metric] = {
                **_ratio_summary(equal_dataset_log),
                "split_counts": _win_counts(split_ratios),
                "dataset_counts": _win_counts(dataset_ratios),
                "worst_split": {
                    "dataset": worst_split["dataset"],
                    "repeat": worst_split["repeat"],
                    "fold": worst_split["fold"],
                    "ratio": worst_split[field],
                },
                "worst_dataset": {
                    "dataset": worst_dataset,
                    "ratio": math.exp(dataset_logs[worst_dataset]),
                },
            }
            for dataset in campaign.TASKS:
                selected = [
                    item for item in paired_rows if item["dataset"] == dataset
                ]
                worst = max(selected, key=lambda item: float(item[field]))
                row = next(
                    (
                        item
                        for item in per_dataset
                        if item["contrast"] == code and item["dataset"] == dataset
                    ),
                    None,
                )
                if row is None:
                    row = {
                        "contrast": code,
                        "numerator": code.split("/")[0],
                        "denominator": code.split("/")[1],
                        "dataset": dataset,
                        "task_id": campaign.TASKS[dataset],
                        "split_count": 3,
                        "test_rmse_ratio": None,
                        "test_rmse_pct": None,
                        "test_split_wins": None,
                        "test_split_losses": None,
                        "test_split_ties": None,
                        "test_worst_split": None,
                        "test_worst_split_ratio": None,
                        "val_rmse_ratio": None,
                        "val_rmse_pct": None,
                        "val_split_wins": None,
                        "val_split_losses": None,
                        "val_split_ties": None,
                        "val_worst_split": None,
                        "val_worst_split_ratio": None,
                    }
                    per_dataset.append(row)
                counts = _win_counts(
                    [
                        float(item[field])
                        for item in paired_rows
                        if item["dataset"] == dataset
                    ]
                )
                metric_name = "test" if metric == "test_rmse" else "val"
                summary = _ratio_summary(dataset_logs[dataset])
                row[f"{metric}_ratio"] = summary["ratio"]
                row[f"{metric}_pct"] = summary["pct"]
                row[f"{metric_name}_split_wins"] = counts["wins"]
                row[f"{metric_name}_split_losses"] = counts["losses"]
                row[f"{metric_name}_split_ties"] = counts["ties"]
                row[f"{metric_name}_worst_split"] = (
                    f"r{worst['repeat']}f{worst['fold']}"
                )
                row[f"{metric_name}_worst_split_ratio"] = worst[field]
        contrasts[code] = contrast_summary

    a10_m_datasets = {
        row["dataset"]: float(row["test_rmse_ratio"])
        for row in per_dataset
        if row["contrast"] == "A10/M"
    }
    lodo = []
    for omitted in campaign.TASKS:
        retained = [
            math.log(ratio)
            for dataset, ratio in a10_m_datasets.items()
            if dataset != omitted
        ]
        lodo.append(
            {
                "omitted_dataset": omitted,
                **_ratio_summary(math.fsum(retained) / 12.0),
            }
        )
    worst_lodo = max(lodo, key=lambda item: item["ratio"])
    gates = {
        "complete_39_coordinate_and_312_child_grid": True,
        "zero_failures_imputations_deadlines_or_time_limit_stops": True,
        "parity_g_a10_over_m_at_most_1": contrasts["A10/M"]["test_rmse"][
            "ratio"
        ]
        <= GATE_THRESHOLDS["parity_a10_over_m_max"],
        "worst_dataset_a10_over_p_at_most_1_02": contrasts["A10/P"][
            "test_rmse"
        ]["worst_dataset"]["ratio"]
        <= GATE_THRESHOLDS["worst_dataset_a10_over_p_max"],
        "worst_lodo_a10_over_m_at_most_1_01": worst_lodo["ratio"]
        <= GATE_THRESHOLDS["worst_lodo_a10_over_m_max"],
    }
    gates["all_development_gates_pass"] = all(gates.values())
    auto_mode_counts = Counter(
        row["A10_selected_tree_mode"] for row in paired_children
    )
    summary = {
        "protocol": "frozen TabArena B10/A10 regression accuracy shootout",
        "execution_mode": execution_mode,
        "public_arm_labels": {
            "P": "DarkoFit product default (reused)",
            "B10": "fixed catboost-mode 10,000-round base",
            "A10": "validation-selected auto-mode 10,000-round candidate",
            "M": "ChimeraBoost 0.14.1 default (reused)",
            "C": "CatBoost 1.2.10 default (reused reference)",
        },
        "counts": {
            "new_outer_jobs": 78,
            "new_child_fits": 624,
            "paired_outer_coordinates": 39,
            "paired_child_comparisons": 312,
            "datasets": 13,
            "coordinates_per_dataset": 3,
        },
        "aggregation": {
            "split": "paired outer-test RMSE ratio",
            "dataset": "geometric mean over r0f0/r1f1/r2f2",
            "campaign": "equal-dataset geometric mean over 13 datasets",
            "missing_pair_policy": "invalidate; never drop or impute",
        },
        "thresholds": dict(GATE_THRESHOLDS),
        "contrasts": contrasts,
        "lodo_a10_over_m": lodo,
        "worst_lodo_a10_over_m": worst_lodo,
        "a10_selected_tree_mode_counts": dict(sorted(auto_mode_counts.items())),
        "gates": gates,
        "decision": (
            "freeze_iteration_1_candidate"
            if gates["all_development_gates_pass"]
            else "iteration_1_failed_one_or_more_frozen_gates"
        ),
        "test_use_disclosure": (
            "This is a spent 13-dataset development panel and is not independent "
            "confirmation. No parameter was tuned from these shootout results."
        ),
        "timing_disclosure": (
            "A10/B10 wall-clock values are contention-exposed under the paired "
            "two-worker schedule and are descriptive, not a causal arm contrast."
            if execution_mode == "concurrent"
            else "A10/B10 wall-clock values were collected in the two-segment "
            "sequential fallback schedule. They are descriptive and retain "
            "slot/order exposure, not a causal arm contrast."
        ),
    }
    return summary, per_dataset


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty {field}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_report(
    summary: Mapping[str, Any], per_dataset: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "# DarkoFit B10/A10 regression accuracy shootout",
        "",
        summary["test_use_disclosure"],
        "",
        f"Execution mode: **{summary['execution_mode']}**.",
        "",
        "Negative percentages favor the numerator. Quality uses paired outer-test "
        "RMSE, three-split geometric means within each dataset, then an "
        "equal-dataset geometric mean.",
        "",
        "| Contrast | Test RMSE | Validation RMSE | Dataset W/L/T | Split W/L/T |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for code, _numerator, _denominator in CONTRASTS:
        item = summary["contrasts"][code]
        test_counts = item["test_rmse"]["dataset_counts"]
        split_counts = item["test_rmse"]["split_counts"]
        lines.append(
            f"| {code} | {item['test_rmse']['pct']:+.3f}% | "
            f"{item['val_rmse']['pct']:+.3f}% | "
            f"{test_counts['wins']}/{test_counts['losses']}/{test_counts['ties']} | "
            f"{split_counts['wins']}/{split_counts['losses']}/{split_counts['ties']} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, passed in summary["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## A10/M by dataset",
            "",
            "| Dataset | Test RMSE | Validation RMSE | Test split W/L/T | "
            "Worst test split |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in per_dataset:
        if row["contrast"] != "A10/M":
            continue
        lines.append(
            f"| {row['dataset']} | {row['test_rmse_pct']:+.3f}% | "
            f"{row['val_rmse_pct']:+.3f}% | "
            f"{row['test_split_wins']}/{row['test_split_losses']}/"
            f"{row['test_split_ties']} | {row['test_worst_split']} "
            f"({100.0 * (row['test_worst_split_ratio'] - 1.0):+.3f}%) |"
        )
    lines.extend(
        [
            "",
            "## A10/M leave-one-dataset-out",
            "",
            "| Omitted dataset | Ratio | Change |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in summary["lodo_a10_over_m"]:
        lines.append(
            f"| {item['omitted_dataset']} | {item['ratio']:.6f} | "
            f"{item['pct']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            f"Decision: **{summary['decision']}**.",
            "",
            summary["timing_disclosure"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Integrity integration is intentionally below the pure analysis layer so
    # tests can exercise the frozen estimator without campaign artifacts.
    args = parse_args(argv)
    manifest, attestation, payload, digests, reused, protected = (
        verify_campaign_integrity(args.input_dir)
    )
    paired = pair_outer_rows(payload["outer_rows"], reused)
    paired_children = pair_child_rows(payload["child_rows"])
    summary, per_dataset = analyze(
        paired,
        paired_children,
        execution_mode=str(manifest["execution_mode"]),
    )
    summary["integrity_diagnostics"] = {
        "new_outer_jobs_verified": 78,
        "new_child_fits_verified": 624,
        "paired_outer_coordinates_verified": 39,
        "paired_child_comparisons_verified": 312,
        "missing_results": 0,
        "duplicate_results": 0,
        "failed_results": 0,
        "imputed_results": 0,
        "deadline_hit_children": 0,
        "time_limit_stop_children": 0,
        "a10_internal_candidates_required_per_child": 3,
        "a10_internal_candidates_verified": 936,
        "reused_source_artifacts_verified": len(SOURCE_ARTIFACTS),
        "source_runtime_and_hardware_lock_verified": True,
    }
    summary["provenance"] = {
        **digests,
        "git_head": manifest["source"]["git_head"],
        "completed_at_utc": attestation.get("completed_at_utc"),
        "protocol_sha256": protocol_sha256(),
    }
    outputs = {
        key: args.input_dir.resolve(strict=True) / name
        for key, name in zip(OUTPUT_KEYS, OUTPUT_NAMES)
    }
    outputs = hardened._canonical_output_targets(
        args.input_dir.resolve(strict=True),
        outputs,
        protected_paths=protected,
        target_names=OUTPUT_KEYS,
    )
    payloads = {
        "split_csv": _csv_bytes(paired, "paired split CSV"),
        "dataset_csv": _csv_bytes(per_dataset, "per-dataset CSV"),
        "child_csv": _csv_bytes(paired_children, "paired child CSV"),
        "summary_json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report_md": render_report(summary, per_dataset).encode("utf-8"),
    }
    hardened._atomic_write_group(
        [(outputs[name], payloads[name]) for name in OUTPUT_KEYS],
        post_write_check=lambda: verify_campaign_integrity(args.input_dir),
    )
    print(
        f"analyzed 39 B10/A10 coordinates and 312 paired children; "
        f"decision={summary['decision']}; wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
