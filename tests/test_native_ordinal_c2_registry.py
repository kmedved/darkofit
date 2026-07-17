from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import build_native_ordinal_c2_registry as registry
from benchmarks import build_ctr23_contamination_registry as ctr


REGISTRY_FILE_SHA256 = (
    "34343d5296698ad7ac728fbef40961f384ca61923e6524afa8a2c7eeda7080d3"
)
REGISTRY_CONTENT_SHA256 = (
    "e7493131eb0cb1da00f1118c39f29130a44381e12f38bd2e2bd972132f953b28"
)


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=path, text=True
    ).strip()


def test_native_ordinal_c2_declarations_are_frozen_before_current_head():
    declarations = json.loads(
        registry.DECLARATIONS.read_text(encoding="utf-8")
    )
    assert declarations["darkofit_prefreeze_head"] == (
        "a74299e67307f44675c4f2b73d581a633885387b"
    )
    assert len(declarations["development_tasks"]) == 8
    assert len(declarations["confirmation_tasks"]) == 5
    assert declarations["coordinate_folds"] == [0, 1, 2]
    dimensions = declarations["official_split_dimensions"]
    assert set(dimensions) == {
        str(row["task_id"])
        for tier in ("development_tasks", "confirmation_tasks")
        for row in declarations[tier]
    }
    assert dimensions["363631"] == {
        "repeats": 10,
        "folds": 3,
        "samples": 1,
    }
    assert dimensions["361622"] == {
        "repeats": 10,
        "folds": 10,
        "samples": 1,
    }
    assert sum(value == {
        "repeats": 1,
        "folds": 10,
        "samples": 1,
    } for value in dimensions.values()) == 11
    assert sum(
        bool(row["ordinal_features"])
        for row in declarations["development_tasks"]
    ) == 4
    assert all(
        len(row["ordinal_features"]) == 1
        for row in declarations["confirmation_tasks"]
    )


def test_native_ordinal_c2_registry_is_immutable_and_target_blind():
    payload = registry.DEFAULT_OUTPUT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == REGISTRY_FILE_SHA256
    artifact = json.loads(payload)
    assert payload == ctr.canonical_json_bytes(artifact)
    content_hash = artifact.pop("registry_sha256")
    assert content_hash == REGISTRY_CONTENT_SHA256
    assert ctr.sha256_json(artifact) == REGISTRY_CONTENT_SHA256
    artifact["registry_sha256"] = content_hash

    assert artifact["sources"] == {
        "darkofit_execution_head": (
            "00d28f9c0d6ca731d92caeca9f04e8d938008405"
        ),
        "darkofit_prefreeze_head": (
            "a74299e67307f44675c4f2b73d581a633885387b"
        ),
        "chimeraboost_head": (
            "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
        ),
    }
    assert artifact["coordinate_counts"] == {
        "development": 24,
        "confirmation": 15,
    }
    assert artifact["development_engaged_task_count"] == 4
    assert artifact["confirmation_lineage_count"] == 5
    assert all(
        row["status"] == "eligible" and not row["exclusion_reasons"]
        for row in artifact["confirmation_tasks"]
    )
    assert artifact["power_analysis"]["pass_probability"] == 1.0
    assert artifact["power_analysis"]["passes"] is True
    assert artifact["selection_used_target_statistics"] is False
    assert artifact["development_outcomes_inspected"] is False
    assert artifact["confirmation_outcomes_inspected"] is False
    assert artifact["lockbox_touched"] is False
    assert artifact["confirmation_run_authorized"] is False
    assert all(
        row["target_values_inspected"] is False
        and row["target_statistics_used"] is False
        for tier in ("development_tasks", "confirmation_tasks")
        for row in artifact[tier]
    )

    assert artifact["builder_source_sha256"] == hashlib.sha256(
        Path(registry.__file__).read_bytes()
    ).hexdigest()
    assert artifact["protocol_sha256"] == hashlib.sha256(
        registry.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["declarations_sha256"] == hashlib.sha256(
        registry.DECLARATIONS.read_bytes()
    ).hexdigest()
    for relative_path, expected_hash in artifact["source_artifacts"].items():
        assert hashlib.sha256(
            (registry.ROOT / relative_path).read_bytes()
        ).hexdigest() == expected_hash


def test_exact_five_task_bootstrap_upper_matches_brute_force():
    values = np.log(
        np.asarray([[0.90, 0.95, 1.00, 1.02, 0.98]], dtype=np.float64)
    )
    observed = float(registry._bootstrap_upper(values)[0])
    brute = sorted(
        float(np.mean(values[0, list(indices)]))
        for indices in itertools.product(range(5), repeat=5)
    )
    expected = brute[math.ceil(0.95 * len(brute)) - 1]
    assert observed == pytest.approx(expected, rel=0.0, abs=2e-18)


def test_power_analysis_is_deterministic_and_covers_full_confirmation_gate():
    first = registry._power_analysis()
    second = registry._power_analysis()
    assert first == second
    assert first["simulations"] == 200_000
    assert first["simulated_lineages"] == 5
    assert first["splits_per_lineage"] == 3
    assert first["effect_retention"] == 0.25
    assert first["pass_probability"] == 1.0
    assert first["passes"] is True
    assert first["gates"]["task_bootstrap_method"] == (
        "exact_multinomial_count_vectors_higher"
    )


def test_model_categorical_policy_unions_metadata_and_nonnumeric_columns():
    frame = pd.DataFrame({
        "numeric": [1.0, 2.0],
        "marked_numeric_codes": [1, 2],
        "unmarked_labels": ["low", "high"],
        "numeric_strings": ["3.0", "4.0"],
    })
    categorical, inferred = registry._model_categorical_indices(
        frame, [False, True, False, False]
    )
    assert inferred == [2]
    assert categorical == [1, 2]


def test_structured_repository_scan_ignores_unkeyed_numeric_collisions(
    tmp_path: Path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "noise.json").write_text(
        '{"elapsed_ns": 363221, "bytes": 46165}\n',
        encoding="utf-8",
    )
    _git(repository, "add", "noise.json")
    _git(repository, "commit", "-qm", "noise")
    revision = _git(repository, "rev-parse", "HEAD")
    declaration = {
        "task_id": 363221,
        "dataset_id": 46165,
        "dataset_name": "nwtco",
    }
    assert registry._structured_repository_hits(
        repository, revision, declaration
    ) == []

    (repository / "exposure.json").write_text(
        '{"openml_task_id": 363221, "dataset_name": "nwtco"}\n',
        encoding="utf-8",
    )
    _git(repository, "add", "exposure.json")
    _git(repository, "commit", "-qm", "exposure")
    revision = _git(repository, "rev-parse", "HEAD")
    hits = registry._structured_repository_hits(
        repository, revision, declaration
    )
    assert {row["kind"] for row in hits} == {
        "task_id",
        "quoted_dataset_name",
    }


def test_registry_atomic_create_is_create_only(tmp_path: Path):
    output = tmp_path / "registry.json"
    registry._atomic_create(output, b"first")
    with pytest.raises(FileExistsError):
        registry._atomic_create(output, b"second")
    assert output.read_bytes() == b"first"
