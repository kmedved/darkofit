#!/usr/bin/env python3
"""Run Protein attribution attempt 2 with terminal attempt-1 lineage."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Iterator, Sequence

RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
BENCH = RUNNER_PATH.parent
_root_text = str(ROOT)
if _root_text in sys.path:
    sys.path.remove(_root_text)
sys.path.insert(0, _root_text)

from benchmarks import run_automatic_linear_selector_v2_protein_attribution as _base


PROTOCOL_PATH = (
    BENCH / "automatic_linear_selector_v2_protein_attribution_attempt2_protocol.md"
)
TEST_PATH = (
    ROOT / "tests/test_automatic_linear_selector_v2_protein_attribution_attempt2.py"
)

CONTRACT_ID = (
    "automatic-linear-selector-v2-protein-attribution-attempt2-20260722"
)
ATTEMPT_INDEX = 2
R1_COMMIT = "d938d99bbc6324a0d8d34129a4c3b1c0ba2da5a9"
ATTEMPT1_CONTRACT_ID = "automatic-linear-selector-v2-protein-attribution-20260722"
ATTEMPT1_PROTOCOL_PATH = (
    BENCH / "automatic_linear_selector_v2_protein_attribution_protocol.md"
)
ATTEMPT1_MANIFEST_PATH = (
    BENCH
    / "automatic_linear_selector_v2_protein_attribution_attempt1_20260722_manifest.json"
)
ATTEMPT1_RESULT_PATH = (
    BENCH
    / "automatic_linear_selector_v2_protein_attribution_attempt1_20260722_result.json"
)
ATTEMPT1_NOTE_PATH = (
    BENCH / "automatic_linear_selector_v2_protein_attribution_attempt1_result.md"
)
R1_PLAN_PATH = ROOT / "BEAT_CHIMERABOOST_PLAN.md"
BASE_RUNNER_PATH = BENCH / "run_automatic_linear_selector_v2_protein_attribution.py"
BASE_TEST_PATH = ROOT / "tests/test_automatic_linear_selector_v2_protein_attribution.py"
RELEASE_RAW_PATH = BENCH / "v011_compute_ladder_v3_raw.json"
OPENML_VERSION = "0.15.1"
EXPECTED_SPLIT_FINGERPRINTS = {
    (0, 0): "88e9e1cb742d160a72bd4fc1977613f41f2a48bb713ebe1ec4bd17365d165e9a",
    (1, 1): "2d7f22d94f5308ee464da1eea65e64bf98923453222fdd0aee99377e8323d173",
    (2, 2): "e9e44958b94bbcb462434041a9c0df76eae8cfd6d893fbac3ce8bd9a9b6ad819",
}

EXPECTED_HASHES = {
    _base.DEVELOPMENT_CONTRACT_PATH: (
        "fe2d476417e8e8087a3c7342eee0d5cb82a6b8a4ee3f360a1806ee4c0922163b"
    ),
    _base.M6_RESULT_PATH: (
        "7445b70ca3bc727bb24f8990ceef590ca933eb1dd45ccefe9ee5788eff211948"
    ),
    _base.M6_MANIFEST_PATH: (
        "601f069896cdf664fcab470abe8c3643f0c0aacf5f79572a6663e304af3d7782"
    ),
    _base.RELEASE_CONTRACT_PATH: (
        "61e788f06b88eefcc2e3c08a38402bf93246e7334980a77061b46763650b581a"
    ),
    _base.RELEASE_RUNNER_PATH: (
        "db5b47af68fa0d74458c9d48d0c441caee8621cf1922542df2a27668118d14fb"
    ),
    ATTEMPT1_PROTOCOL_PATH: (
        "e231ab25297cb61280ed72716a423d2ec86c71403a5521d40c3ea5d346580d8f"
    ),
    ATTEMPT1_MANIFEST_PATH: (
        "4b4471cdba3beab6cc9dc2cce8d1c8835bfa01cebc986321b9541f89e191def4"
    ),
    ATTEMPT1_RESULT_PATH: (
        "e4bb44356c90d18e88c252bc2a9c8d197303e4a4cb750daacee6eda3c104ab0f"
    ),
    ATTEMPT1_NOTE_PATH: (
        "9c1283bfbeda19fdd32a320c113220913e684da91a2eaa0564bf8114e26cddad"
    ),
    R1_PLAN_PATH: (
        "b620b4e8cc522c02b08076607bc3c064d873c5d3a2997676343e628537ed0119"
    ),
    BASE_RUNNER_PATH: (
        "cfb21064b48c8acc73587118a88004eb3c65e74f3552408a1133c9243257711d"
    ),
    BASE_TEST_PATH: (
        "0229511ab9d83122f406348b72c33c5f7ed3f6bde620b26ec349dc19b850d6e7"
    ),
    RELEASE_RAW_PATH: (
        "96f594da1a0ea885aa55d45636049d97b9b6e1a7f56d85679dfe879420636f79"
    ),
}

_BASE_VALIDATE_BOUND_EVIDENCE = _base.validate_bound_evidence
_BASE_DATA_LOADER_PREFLIGHT = _base._data_loader_preflight


def _validate_attempt1_lineage() -> dict[str, Any]:
    manifest = json.loads(ATTEMPT1_MANIFEST_PATH.read_text())
    result = json.loads(ATTEMPT1_RESULT_PATH.read_text())
    candidate = manifest.get("sources", {}).get("candidate", {})
    analysis = result.get("analysis", {})
    if (
        manifest.get("contract_id") != ATTEMPT1_CONTRACT_ID
        or manifest.get("attempt_index") != 1
        or manifest.get("attempt_spent") is not True
        or manifest.get("status") != "launched_before_worker_zero"
        or candidate.get("head") != _base.CANDIDATE_COMMIT
        or result.get("contract_id") != ATTEMPT1_CONTRACT_ID
        or result.get("attempt_index") != 1
        or result.get("attempt_spent") is not True
        or analysis.get("disposition") != "terminal_execution_failure"
        or analysis.get("completed_worker_count") != 0
        or "No module named 'autogluon'" not in analysis.get("error", "")
        or result.get("shipping_or_default_claim_eligible") is not False
        or result.get("fresh_or_lockbox_accessed") is not False
        or "raw" in result.get("artifacts", {})
    ):
        raise RuntimeError("attempt-1 terminal lineage is invalid")
    return {
        "contract_id": ATTEMPT1_CONTRACT_ID,
        "attempt_index": 1,
        "attempt_spent": True,
        "completed_worker_count": 0,
        "scientific_outcome_observed": False,
        "disposition": "terminal_execution_failure",
        "candidate_commit": candidate["head"],
    }


def validate_bound_evidence() -> dict[str, Any]:
    bindings = _BASE_VALIDATE_BOUND_EVIDENCE()
    lineage = _validate_attempt1_lineage()
    plan_text = R1_PLAN_PATH.read_text()
    release_raw_text = RELEASE_RAW_PATH.read_text()
    if (
        _base._git(ROOT, "rev-parse", R1_COMMIT) != R1_COMMIT
        or "Revision R1" not in plan_text
        or "Protein attribution attempt 2" not in plan_text
        or any(
            fingerprint not in release_raw_text
            for fingerprint in EXPECTED_SPLIT_FINGERPRINTS.values()
        )
    ):
        raise RuntimeError("R1 attempt-2 authorization binding is invalid")
    return {
        **bindings,
        "attempt1_terminal_lineage": lineage,
        "r1_authorization_commit": R1_COMMIT,
    }


def _load_split(
    repeat: int,
    fold: int,
    tabarena_source: Path,
) -> dict[str, Any]:
    """Load the exact TabArena OpenML split without importing its full context."""
    del tabarena_source
    import openml

    if distribution_version("openml") != OPENML_VERSION:
        raise RuntimeError("attempt-2 OpenML version drifted")
    task = openml.tasks.get_task(
        _base.TASK_ID,
        download_splits=False,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset = task.get_dataset()
    if (
        int(task.task_id) != _base.TASK_ID
        or task.task_type_id.name != "SUPERVISED_REGRESSION"
        or task.target_name != "ResidualSize"
        or dataset.name != _base.DATASET
    ):
        raise RuntimeError("attempt-2 OpenML task identity drifted")
    X, y, _, _ = dataset.get_data(task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(
        fold=fold,
        repeat=repeat,
        sample=0,
    )
    data = {
        "task_id": int(task.task_id),
        "X_train": X.loc[train_indices],
        "y_train": y[train_indices],
        "X_test": X.loc[test_indices],
        "y_test": y[test_indices],
    }
    observed = _base._split_fingerprints(data)["combined_sha256"]
    expected = EXPECTED_SPLIT_FINGERPRINTS.get((repeat, fold))
    if expected is None or observed != expected:
        raise RuntimeError("attempt-2 Protein split fingerprint drifted")
    return data


def _data_loader_preflight(
    candidate: Path,
    tabarena_source: Path,
) -> dict[str, Any]:
    result = _BASE_DATA_LOADER_PREFLIGHT(candidate, tabarena_source)
    result["loader"] = "direct_openml_equivalent_to_tabarena_openml_task_wrapper"
    result["python_executable"] = sys.executable
    result["packages"] = {
        name: distribution_version(name)
        for name in ("openml", "numpy", "pandas", "scikit-learn")
    }
    if result["packages"]["openml"] != OPENML_VERSION:
        raise RuntimeError("attempt-2 preflight OpenML version drifted")
    observed = {
        (item["repeat"], item["fold"]): item["combined_split_sha256"]
        for item in result["coordinates"]
    }
    if observed != EXPECTED_SPLIT_FINGERPRINTS:
        raise RuntimeError("attempt-2 preflight split set drifted")
    return result


@contextmanager
def _configured_base() -> Iterator[None]:
    patches = {
        "RUNNER_PATH": RUNNER_PATH,
        "ROOT": ROOT,
        "BENCH": BENCH,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "TEST_PATH": TEST_PATH,
        "CONTRACT_ID": CONTRACT_ID,
        "ATTEMPT_INDEX": ATTEMPT_INDEX,
        "EXPECTED_HASHES": EXPECTED_HASHES,
        "validate_bound_evidence": validate_bound_evidence,
        "_load_split": _load_split,
        "_data_loader_preflight": _data_loader_preflight,
    }
    originals = {name: getattr(_base, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(_base, name, value)


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_base():
        return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
