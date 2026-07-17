from __future__ import annotations

import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import build_native_ordinal_c2_registry as registry


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
