#!/usr/bin/env python3
"""Record group-centered-cross engagement on the exact M6 v3 grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import run_automatic_linear_selector_v2_m6_engagement as foundation
except ImportError:  # pragma: no cover
    from benchmarks import (
        run_automatic_linear_selector_v2_m6_engagement as foundation,
    )


IDENTITY = "group-centered-categorical-crosses-v1-m6-engagement-20260722"
MECHANISM_ID = "group_centered_categorical_crosses_v1"
RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
PROTOCOL_PATH = RUNNER_PATH.with_name(
    "group_centered_categorical_crosses_v1_m6_engagement_companion.md"
)
CONTRACT_PATH = RUNNER_PATH.with_name(
    "group_centered_categorical_crosses_v1_development_contract.md"
)
FOUNDATION_PATH = RUNNER_PATH.with_name(
    "run_automatic_linear_selector_v2_m6_engagement.py"
)
FOUNDATION_SHA256 = (
    "50d69b1c372b4e6849796c85f42395651d19ebabbe184f993e5addfa5e864969"
)
WORKER_PREFIX = "GROUP_CENTERED_CROSS_ENGAGEMENT="

_CROSS_FIELDS = {
    "version",
    "eligible",
    "reason",
    "selected",
    "pairs",
    "split",
    "control_validation_rmse",
    "augmented_validation_rmse",
    "relative_validation_improvement",
    "selection_total_seconds",
    "final_pairs",
    "final_preprocessing",
}
_ELIGIBLE_ONLY_FIELDS = {
    "fit_random_state_seed",
    "selection_fits",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cross_metadata_wrapper(**kwargs) -> dict[str, Any]:
    row = foundation._ORIGINAL_EVIDENCE_ROW_METADATA(**kwargs)
    metadata = json.loads(row["model_metadata"])
    metadata["group_centered_categorical_crosses"] = getattr(
        kwargs["model"], "group_centered_categorical_crosses_", None
    )
    row["model_metadata"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    )
    return row


def _cross_record(row: Mapping[str, Any], *, task: str) -> dict[str, Any]:
    metadata = json.loads(str(row["model_metadata"]))
    record = metadata.get("group_centered_categorical_crosses")
    if task == "regression":
        if not isinstance(record, dict):
            raise RuntimeError("regression cross engagement metadata is absent")
        expected = set(_CROSS_FIELDS)
        if record.get("eligible") is True:
            expected.update(_ELIGIBLE_ONLY_FIELDS)
        if (
            set(record) != expected
            or record.get("version") != 1
            or not isinstance(record.get("eligible"), bool)
            or not isinstance(record.get("selected"), bool)
            or not isinstance(record.get("reason"), str)
            or not record["reason"]
            or not isinstance(record.get("pairs"), list)
            or not isinstance(record.get("final_pairs"), list)
            or record["selected"] != bool(record["final_pairs"])
        ):
            raise RuntimeError("regression cross engagement metadata is invalid")
        return record
    if record is not None:
        raise RuntimeError("classification unexpectedly gained cross state")
    return {
        "eligible": False,
        "selected": False,
        "reason": "classification_not_applicable",
    }


def _configure_foundation() -> None:
    if file_sha256(FOUNDATION_PATH) != FOUNDATION_SHA256:
        raise RuntimeError("engagement companion foundation hash drifted")
    foundation.IDENTITY = IDENTITY
    foundation.MECHANISM_ID = MECHANISM_ID
    foundation.RUNNER_PATH = RUNNER_PATH
    foundation.REPO_ROOT = REPO_ROOT
    foundation.PROTOCOL_PATH = PROTOCOL_PATH
    foundation.SELECTOR_CONTRACT_PATH = CONTRACT_PATH
    foundation.WORKER_PREFIX = WORKER_PREFIX
    foundation._selector_metadata_wrapper = _cross_metadata_wrapper
    foundation._selector_record = _cross_record


def expected_identities() -> tuple[tuple[str, str, int, str], ...]:
    return foundation.expected_identities()


def run(args):
    _configure_foundation()
    return foundation.run(args)


def main(argv: Sequence[str] | None = None) -> None:
    _configure_foundation()
    foundation.main(argv)


if __name__ == "__main__":
    main()
