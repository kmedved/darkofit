#!/usr/bin/env python3
"""Measure exact-component factoring bounds for a safe ensemble archive.

This is a read-only feasibility screen.  It rebuilds non-loadable size-model
archives in memory; it does not change DarkoFit serialization or claim that a
simulated layout is a finished persistence format.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


GATE_ARCHIVE_TO_SINGLE = 4.0
R3_ARCHIVE_TO_SINGLE = 5.534767493867151
R3_MEMBER_COUNT = 8
R3_REQUIRED_SHARE_PER_DUPLICATE = (
    R3_ARCHIVE_TO_SINGLE - GATE_ARCHIVE_TO_SINGLE
) / (R3_MEMBER_COUNT - 1)

_PREPROCESSOR_PREFIXES = ("prep__", "bin__")
_PREPROCESSOR_HEADER_KEYS = ("n_input_features", "prep", "feature_names_in")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_npz(source) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("archive is not a safe NPZ payload") from exc
    with archive as data:
        arrays = {name: data[name].copy() for name in data.files}
    try:
        header = json.loads(str(arrays["header"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("archive header is invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("archive header must be an object")
    return arrays, header


def _np_save_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(value), allow_pickle=False)
    return output.getvalue()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    payload = _np_save_bytes(value)
    standalone = _npz_bytes({"component": np.asarray(value)})
    return {
        "sha256": _sha256_bytes(payload),
        "npy_bytes": len(payload),
        "standalone_npz_bytes": len(standalone),
        "dtype": str(np.asarray(value).dtype),
        "shape": [int(size) for size in np.asarray(value).shape],
    }


def _component_section(name: str) -> str:
    if name.startswith(_PREPROCESSOR_PREFIXES):
        return "preprocessor"
    if name.startswith("trees__"):
        return "trees"
    if name.startswith(("cat", "enc")) or name in {"classes", "classes_kinds"}:
        return "categorical_or_target"
    if name.startswith("wrapper__"):
        return "wrapper"
    if name.startswith("shap__"):
        return "shap"
    return "fitted_model_state"


def _member_payloads(
    outer_arrays: Mapping[str, np.ndarray], member_count: int
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    arrays = []
    headers = []
    for index in range(member_count):
        name = f"member_{index:04d}"
        if name not in outer_arrays:
            raise ValueError(f"ensemble member payload {name!r} is missing")
        values = np.asarray(outer_arrays[name])
        if values.ndim != 1 or values.dtype != np.uint8 or values.size == 0:
            raise ValueError(f"ensemble member payload {name!r} is invalid")
        member_arrays, header = _read_npz(io.BytesIO(values.tobytes()))
        arrays.append(member_arrays)
        headers.append(header)
    return arrays, headers


def _preprocessor_header(header: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: header[name]
        for name in _PREPROCESSOR_HEADER_KEYS
        if name in header
    }


def _simulate_factored_archive(
    outer_arrays: Mapping[str, np.ndarray],
    member_arrays: list[dict[str, np.ndarray]],
    member_headers: list[dict[str, Any]],
    shared_names: set[str],
    *,
    factor_preprocessor_header: bool,
) -> int:
    simulated = {name: value.copy() for name, value in outer_arrays.items()}
    for index, (arrays, header) in enumerate(zip(member_arrays, member_headers)):
        reduced = {
            name: value.copy()
            for name, value in arrays.items()
            if name not in shared_names and name != "header"
        }
        reduced_header = json.loads(json.dumps(header))
        if factor_preprocessor_header:
            for name in _PREPROCESSOR_HEADER_KEYS:
                reduced_header.pop(name, None)
            reduced_header["shared_preprocessor_ref"] = "canonical_v1"
        reduced["header"] = np.array(json.dumps(reduced_header))
        payload = _npz_bytes(reduced)
        simulated[f"member_{index:04d}"] = np.frombuffer(
            payload, dtype=np.uint8
        ).copy()

    first = member_arrays[0]
    for name in sorted(shared_names):
        simulated[f"shared__{name}"] = first[name].copy()
    if factor_preprocessor_header:
        simulated["shared__preprocessor_header"] = np.array(
            _canonical_json(_preprocessor_header(member_headers[0]))
        )
    simulated["shared__component_manifest"] = np.array(
        _canonical_json({
            "non_loadable_size_model": True,
            "shared_names": sorted(shared_names),
            "preprocessor_header_factored": factor_preprocessor_header,
        })
    )
    return len(_npz_bytes(simulated))


def analyze_archive(
    ensemble_path: str | Path,
    *,
    single_path: str | Path | None = None,
    gate: float = GATE_ARCHIVE_TO_SINGLE,
    reference_ratio: float | None = None,
    reference_member_count: int | None = None,
) -> dict[str, Any]:
    ensemble = Path(ensemble_path).expanduser().resolve()
    if not ensemble.is_file() or ensemble.is_symlink():
        raise ValueError("ensemble archive must be a regular file")
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("gate must be a positive finite number")
    if reference_ratio is not None and (
        not np.isfinite(reference_ratio) or reference_ratio <= 0.0
    ):
        raise ValueError("reference_ratio must be a positive finite number")
    if reference_member_count is not None and (
        isinstance(reference_member_count, bool)
        or not isinstance(reference_member_count, int)
        or reference_member_count < 2
    ):
        raise ValueError("reference_member_count must be an integer at least 2")
    if (reference_ratio is None) != (reference_member_count is None):
        raise ValueError(
            "reference_ratio and reference_member_count must be supplied together"
        )

    outer_arrays, outer_header = _read_npz(ensemble)
    if outer_header.get("archive_kind") != "darkofit_ensemble":
        raise ValueError("archive is not a DarkoFit ensemble")
    member_count = outer_header.get("member_count")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise ValueError("ensemble member count is invalid")
    members, member_headers = _member_payloads(outer_arrays, member_count)

    name_sets = [set(member).difference({"header"}) for member in members]
    union_names = set.union(*name_sets)
    exact_common = set()
    components = []
    for name in sorted(union_names):
        present_everywhere = all(name in member for member in members)
        fingerprints = (
            [_array_fingerprint(member[name]) for member in members]
            if present_everywhere
            else []
        )
        byte_identical = (
            present_everywhere
            and len({item["sha256"] for item in fingerprints}) == 1
        )
        if byte_identical:
            exact_common.add(name)
        components.append({
            "name": name,
            "section": _component_section(name),
            "present_in_all_members": present_everywhere,
            "byte_identical_across_members": byte_identical,
            "member_fingerprints": fingerprints,
        })

    preprocessor_sets = [
        {name for name in names if name.startswith(_PREPROCESSOR_PREFIXES)}
        for names in name_sets
    ]
    preprocessor_names = set.union(*preprocessor_sets)
    preprocessor_schema_identical = all(
        names == preprocessor_sets[0] for names in preprocessor_sets[1:]
    )
    preprocessor_arrays_identical = (
        bool(preprocessor_names)
        and preprocessor_schema_identical
        and preprocessor_names.issubset(exact_common)
    )
    prep_headers = [_preprocessor_header(header) for header in member_headers]
    preprocessor_headers_identical = (
        bool(prep_headers[0].get("prep"))
        and "n_input_features" in prep_headers[0]
        and len({_canonical_json(value) for value in prep_headers}) == 1
    )
    metadata = outer_header.get("metadata", {})
    numeric_target_free = (
        isinstance(metadata, Mapping)
        and metadata.get("shared_preprocessing") == "numeric_target_free"
    )
    canonical_eligible = (
        numeric_target_free
        and preprocessor_arrays_identical
        and preprocessor_headers_identical
    )

    current_bytes = ensemble.stat().st_size
    repacked_current_bytes = len(_npz_bytes(outer_arrays))
    canonical_bytes = None
    if canonical_eligible:
        canonical_bytes = _simulate_factored_archive(
            outer_arrays,
            members,
            member_headers,
            preprocessor_names,
            factor_preprocessor_header=True,
        )
    all_exact_bytes = _simulate_factored_archive(
        outer_arrays,
        members,
        member_headers,
        exact_common,
        factor_preprocessor_header=canonical_eligible,
    )
    all_exact_bytes = min(current_bytes, all_exact_bytes)

    single = None
    if single_path is not None:
        single = Path(single_path).expanduser().resolve()
        if not single.is_file() or single.is_symlink():
            raise ValueError("single archive must be a regular file")
        single_arrays, single_header = _read_npz(single)
        if single_header.get("archive_kind") == "darkofit_ensemble":
            raise ValueError("single archive must contain one fitted model")
        del single_arrays

    single_bytes = None if single is None else single.stat().st_size
    current_ratio = None if single_bytes is None else current_bytes / single_bytes
    canonical_ratio = (
        None
        if single_bytes is None or canonical_bytes is None
        else canonical_bytes / single_bytes
    )
    all_exact_ratio = (
        None if single_bytes is None else all_exact_bytes / single_bytes
    )
    required_savings = (
        None
        if single_bytes is None or current_ratio <= gate
        else (current_bytes - gate * single_bytes) / (member_count - 1)
    )
    canonical_savings = (
        None
        if canonical_bytes is None
        else (current_bytes - canonical_bytes) / (member_count - 1)
    )
    all_exact_savings = (current_bytes - all_exact_bytes) / (member_count - 1)

    verdict = "component_census_only"
    if single_bytes is not None:
        if current_ratio <= gate:
            verdict = "already_within_gate"
        elif all_exact_ratio > gate:
            verdict = "kill_all_exact_entries_insufficient"
        elif canonical_ratio is not None and canonical_ratio <= gate:
            verdict = "advance_canonical_preprocessor_feasible"
        else:
            verdict = "kill_requires_out_of_scope_sections"

    reference_required_share = None
    reference_verdict = "not_evaluated"
    if reference_ratio is not None:
        reference_required_share = max(
            0.0,
            (reference_ratio - gate) / (reference_member_count - 1),
        )
        if single_bytes is None:
            reference_verdict = "not_evaluated_without_single_archive"
        elif reference_required_share == 0.0:
            reference_verdict = "reference_already_within_gate"
        elif all_exact_savings / single_bytes < reference_required_share:
            reference_verdict = "kill_reference_all_exact_insufficient"
        elif (
            canonical_savings is not None
            and canonical_savings / single_bytes >= reference_required_share
        ):
            reference_verdict = "advance_reference_canonical_plausible"
        else:
            reference_verdict = "kill_reference_requires_out_of_scope_sections"

    return {
        "schema_version": 1,
        "analysis": "ensemble_archive_component_census",
        "non_loadable_size_simulation": True,
        "ensemble": {
            "path": str(ensemble),
            "bytes": current_bytes,
            "repacked_size_model_bytes": repacked_current_bytes,
            "sha256": _sha256_path(ensemble),
            "member_count": member_count,
            "metadata_version": metadata.get("version")
            if isinstance(metadata, Mapping)
            else None,
            "private_prototype": metadata.get("private_prototype")
            if isinstance(metadata, Mapping)
            else None,
        },
        "single": None
        if single is None
        else {
            "path": str(single),
            "bytes": single_bytes,
            "sha256": _sha256_path(single),
        },
        "gate": {
            "archive_to_single_at_most": float(gate),
            "current_archive_to_single": current_ratio,
            "canonical_preprocessor_to_single": canonical_ratio,
            "all_exact_entries_to_single": all_exact_ratio,
            "required_savings_per_duplicate_bytes": required_savings,
            "required_savings_per_duplicate_to_single": None
            if required_savings is None
            else required_savings / single_bytes,
            "canonical_savings_per_duplicate_bytes": canonical_savings,
            "canonical_savings_per_duplicate_to_single": None
            if canonical_savings is None or single_bytes is None
            else canonical_savings / single_bytes,
            "all_exact_savings_per_duplicate_to_single": None
            if single_bytes is None
            else all_exact_savings / single_bytes,
            "verdict": verdict,
            "reference_screen": {
                "current_archive_to_single": reference_ratio,
                "member_count": reference_member_count,
                "required_share_per_duplicate_to_single": (
                    reference_required_share
                ),
                "verdict": reference_verdict,
            },
        },
        "canonical_preprocessor": {
            "numeric_target_free_provenance": numeric_target_free,
            "array_schema_identical": preprocessor_schema_identical,
            "arrays_byte_identical": preprocessor_arrays_identical,
            "headers_byte_identical": preprocessor_headers_identical,
            "eligible": canonical_eligible,
            "array_names": sorted(preprocessor_names),
            "simulated_archive_bytes": canonical_bytes,
        },
        "optimistic_all_exact_entries": {
            "array_names": sorted(exact_common),
            "simulated_archive_bytes": all_exact_bytes,
            "includes_out_of_scope_sections": bool(
                exact_common.difference(preprocessor_names)
            ),
        },
        "components": components,
        "r3_reference": {
            "archive_to_single": R3_ARCHIVE_TO_SINGLE,
            "member_count": R3_MEMBER_COUNT,
            "gate": GATE_ARCHIVE_TO_SINGLE,
            "required_share_per_duplicate_to_single": (
                R3_REQUIRED_SHARE_PER_DUPLICATE
            ),
        },
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ensemble", type=Path)
    parser.add_argument("--single", type=Path)
    parser.add_argument("--gate", type=float, default=GATE_ARCHIVE_TO_SINGLE)
    parser.add_argument("--reference-ratio", type=float)
    parser.add_argument("--reference-member-count", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = analyze_archive(
        args.ensemble,
        single_path=args.single,
        gate=args.gate,
        reference_ratio=args.reference_ratio,
        reference_member_count=args.reference_member_count,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
