#!/usr/bin/env python3
"""Run the spent-data RSSI linear-leaf parity diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ID = 363132
DATASET_ID = 45718
TASK_NAME = "3D_Estimation_using_RSSI_of_WLAN_dataset"
TARGET_NAME = "Receiver_Height"
OUTER_REPEAT = 0
OUTER_FOLD = 0
OUTER_SAMPLE = 0
RANDOM_STATE = 4
THREADS = 6
EXPECTED_SHAPE = (5760, 7)
EXPECTED_SPLIT_DIMENSIONS = (1, 10, 1)
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_CROSS_PAIR_COUNT = 30
PROTOCOL = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis_protocol.md"
REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
PRIOR_RESULT = ROOT / "benchmarks" / "fresh_selector_confirmation.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis.json"

ARMS = (
    "darko_default",
    "darko_matched_auto10_linear",
    "darko_matched_auto20_linear",
    "darko_shared_constant",
    "darko_shared_linear",
    "chimera_shared_constant",
    "chimera_shared_linear",
    "chimera_full_selector",
    "chimera_capped_selector",
    "chimera_full_product",
    "chimera_product",
)

EXACT_FIELDS = (
    "borders_sha256",
    "validation_history_sha256",
    "model_sha256",
    "prediction_sha256",
    "fitted_tree_count",
    "best_validation_rmse",
    "test_rmse",
)
_CHIMERA_MODULE_NAME = "_darkofit_rssi_frozen_chimeraboost"
_CHIMERA_IMPORT_LOCK = threading.Lock()
_CHIMERA_MODULE = None
_CHIMERA_MODULES = None
_CHIMERA_REGRESSOR = None
FROZEN_EVIDENCE_SHA256 = (
    "aabf23b858511efe39a3a3be663d89883d5315ecd77c2fb9451eb297017d3ddf"
)
FROZEN_ARTIFACT_SHA256 = (
    "a6f408909ecb12fb3bad25f68f03e83bfaaa8fd81f0c0d9c9eb166bbd4754066"
)


def _private_chimera_provenance_is_valid(
    package,
    initializer,
    module,
    regressor,
    private_modules,
):
    if not isinstance(private_modules, dict):
        return False
    source_name = getattr(module, "__file__", None)
    regressor_name = getattr(regressor, "__module__", None)
    regressor_module = (
        private_modules.get(regressor_name)
        if isinstance(regressor_name, str)
        else None
    )
    regressor_source_name = getattr(regressor_module, "__file__", None)
    regressor_init_code = getattr(
        getattr(regressor, "__init__", None),
        "__code__",
        None,
    )
    if (
        private_modules.get(_CHIMERA_MODULE_NAME) is not module
        or not isinstance(source_name, (str, os.PathLike))
        or Path(source_name).resolve() != initializer.resolve()
        or not isinstance(regressor, type)
        or not isinstance(regressor_name, str)
        or not regressor_name.startswith(f"{_CHIMERA_MODULE_NAME}.")
        or regressor_module is None
        or getattr(module, "ChimeraBoostRegressor", None) is not regressor
        or not isinstance(regressor_source_name, (str, os.PathLike))
        or not Path(regressor_source_name).resolve().is_relative_to(
            package.resolve()
        )
        or regressor_init_code is None
        or Path(regressor_init_code.co_filename).resolve()
        != Path(regressor_source_name).resolve()
    ):
        return False
    for name, loaded in private_modules.items():
        loaded_spec = getattr(loaded, "__spec__", None)
        loaded_loader = getattr(loaded_spec, "loader", None)
        loaded_file = getattr(loaded, "__file__", None)
        loaded_origin = getattr(loaded_spec, "origin", None)
        if (
            sys.modules.get(name) is not loaded
            or getattr(loaded, "__name__", None) != name
            or getattr(loaded_spec, "name", None) != name
            or type(loaded_loader) is not importlib.machinery.SourceFileLoader
            or getattr(loaded_loader, "name", None) != name
            or not isinstance(loaded_file, (str, os.PathLike))
            or not isinstance(loaded_origin, (str, os.PathLike))
            or not isinstance(
                getattr(loaded_loader, "path", None),
                (str, os.PathLike),
            )
            or Path(loaded_file).resolve() != Path(loaded_origin).resolve()
            or Path(loaded_file).resolve()
            != Path(loaded_loader.path).resolve()
            or not Path(loaded_file).resolve().is_relative_to(
                package.resolve()
            )
        ):
            return False
    return True


def _chimera_regressor_class():
    """Load the frozen sibling package without changing global import order."""
    global _CHIMERA_MODULE, _CHIMERA_MODULES, _CHIMERA_REGRESSOR

    package = CHIMERA_ROOT / "chimeraboost"
    initializer = package / "__init__.py"
    with _CHIMERA_IMPORT_LOCK:
        installed = sys.modules.get(_CHIMERA_MODULE_NAME)
        if _CHIMERA_MODULE is None:
            occupied = [
                name
                for name in sys.modules
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            ]
            if occupied:
                raise RuntimeError(
                    "imported chimeraboost from the wrong checkout: "
                    "private module slot was already occupied"
                )
            spec = importlib.util.spec_from_file_location(
                _CHIMERA_MODULE_NAME,
                initializer,
                submodule_search_locations=[str(package)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"cannot load frozen chimeraboost package: {initializer}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[_CHIMERA_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                for name in tuple(sys.modules):
                    if name == _CHIMERA_MODULE_NAME or name.startswith(
                        f"{_CHIMERA_MODULE_NAME}."
                    ):
                        sys.modules.pop(name, None)
                raise
            regressor = getattr(module, "ChimeraBoostRegressor", None)
            private_modules = {
                name: loaded
                for name, loaded in sys.modules.items()
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            }
            if not _private_chimera_provenance_is_valid(
                package,
                initializer,
                module,
                regressor,
                private_modules,
            ):
                for name in tuple(sys.modules):
                    if name == _CHIMERA_MODULE_NAME or name.startswith(
                        f"{_CHIMERA_MODULE_NAME}."
                    ):
                        sys.modules.pop(name, None)
                raise RuntimeError(
                    "imported chimeraboost from the wrong checkout"
                )
            _CHIMERA_MODULE = module
            _CHIMERA_MODULES = private_modules
            _CHIMERA_REGRESSOR = regressor
        elif (
            installed is not _CHIMERA_MODULE
            or _CHIMERA_MODULES is None
            or {
                name
                for name in sys.modules
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            }
            != set(_CHIMERA_MODULES)
            or any(
                sys.modules.get(name) is not loaded
                for name, loaded in _CHIMERA_MODULES.items()
            )
            or not _private_chimera_provenance_is_valid(
                package,
                initializer,
                _CHIMERA_MODULE,
                _CHIMERA_REGRESSOR,
                _CHIMERA_MODULES,
            )
        ):
            raise RuntimeError(
                "imported chimeraboost from the wrong checkout: "
                "private module slot changed"
            )
        return _CHIMERA_REGRESSOR


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_mutable_symlink_output_parents(path: Path) -> None:
    for directory in (path.parent, *path.parent.parents):
        if directory.is_symlink() and os.access(directory.parent, os.W_OK):
            raise RuntimeError(
                f"refusing symlink output directory: {directory}"
            )


def _create_missing_directories(
    directory: Path,
    created: list[tuple[Path, tuple[int, int]]],
) -> None:
    missing = []
    current = directory
    while not current.exists():
        if current.is_symlink():
            raise RuntimeError(
                f"refusing symlink output directory: {current}"
            )
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(
            f"output parent is not a directory: {current}"
        )
    for current in reversed(missing):
        try:
            current.mkdir()
        except FileExistsError:
            if not current.is_dir() or current.is_symlink():
                raise
        else:
            metadata = current.lstat()
            created.append(
                (current, (metadata.st_dev, metadata.st_ino))
            )


def _remove_owned_empty_directories(
    directories: list[tuple[Path, tuple[int, int]]],
) -> None:
    for directory, identity in reversed(directories):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                directory.rmdir()
        except OSError:
            pass


def _assert_output_parent_identity(
    path: Path,
    identity: tuple[int, int],
) -> None:
    _reject_mutable_symlink_output_parents(path)
    try:
        current = path.parent.lstat()
    except OSError as exc:
        raise RuntimeError(f"output parent changed: {path.parent}") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RuntimeError(f"output parent changed: {path.parent}")


def _open_output_parent(path: Path) -> tuple[int, tuple[int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(path.parent, flags)
    try:
        current = os.fstat(descriptor)
        identity = (current.st_dev, current.st_ino)
        if not stat.S_ISDIR(current.st_mode):
            raise RuntimeError(
                f"output parent is not a directory: {path.parent}"
            )
        _assert_output_parent_identity(path, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _temporary_at(
    directory_descriptor: int,
    output_name: str,
) -> tuple[int, str, tuple[int, int]]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".{output_name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        current = os.fstat(descriptor)
        return descriptor, name, (current.st_dev, current.st_ino)
    raise FileExistsError(
        f"unable to reserve temporary output for {output_name}"
    )


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _exists_at(directory_descriptor: int, name: str) -> bool:
    try:
        _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    return True


def _unlink_if_owned_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    try:
        current = _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        os.unlink(name, dir_fd=directory_descriptor)


def _atomic_create(path: Path, value: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    _reject_mutable_symlink_output_parents(path)
    created_directories: list[tuple[Path, tuple[int, int]]] = []
    parent_descriptor = None
    parent_identity = None
    temporary_name = None
    temporary_descriptor = None
    identity = None
    created = False
    try:
        _create_missing_directories(path.parent, created_directories)
        parent_descriptor, parent_identity = _open_output_parent(path)
        if _exists_at(parent_descriptor, path.name):
            raise FileExistsError(
                f"refusing to replace existing output: {path}"
            )
        temporary_descriptor, temporary_name, identity = _temporary_at(
            parent_descriptor,
            path.name,
        )
        with os.fdopen(
            temporary_descriptor,
            "wb",
            closefd=False,
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            # Keep the original inode open until publication and cleanup
            # finish.  Otherwise Linux may immediately reuse its inode after
            # an attacker unlinks the temporary path, making an unrelated
            # replacement appear to have our recorded identity.
            _assert_output_parent_identity(path, parent_identity)
            current = _stat_at(parent_descriptor, temporary_name)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise RuntimeError(
                    "temporary output changed before publication: "
                    f"{path.parent / temporary_name}"
                )
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            created = True
            _assert_output_parent_identity(path, parent_identity)
            current = _stat_at(parent_descriptor, path.name)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise RuntimeError(f"published output changed: {path}")
            _unlink_if_owned_at(
                parent_descriptor,
                temporary_name,
                identity,
            )
            _assert_output_parent_identity(path, parent_identity)
    except BaseException:
        if parent_descriptor is not None and identity is not None:
            if created:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        path.name,
                        identity,
                    )
                except OSError:
                    pass
            if temporary_name is not None:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        temporary_name,
                        identity,
                    )
                except OSError:
                    pass
        _remove_owned_empty_directories(created_directories)
        raise
    finally:
        if temporary_descriptor is not None:
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _array_sha256(value, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state(path: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    head = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source tree is not clean: {path}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(
            f"unexpected source head for {path}: {head} != {expected_head}"
        )
    return {
        "path": str(path),
        "head": head,
        "branch": _git(path, "branch", "--show-current"),
        "clean": True,
    }


def _verify_spent_boundary() -> dict[str, Any]:
    registry_payload = REGISTRY.read_bytes()
    registry = json.loads(registry_payload)
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != 1
        or not isinstance(registry.get("coordinates"), list)
    ):
        raise RuntimeError("RSSI spent-coordinate registry has an unknown schema")
    coordinates = {
        (
            int(row["task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
        )
        for row in registry["coordinates"]
    }
    coordinate = (TASK_ID, OUTER_REPEAT, OUTER_FOLD, OUTER_SAMPLE)
    if coordinate not in coordinates:
        raise RuntimeError("RSSI diagnostic coordinate is not declared spent")

    prior_payload = PRIOR_RESULT.read_bytes()
    prior = json.loads(prior_payload)
    if (
        not isinstance(prior, dict)
        or prior.get("schema_version") != 1
        or not isinstance(prior.get("results"), list)
    ):
        raise RuntimeError("RSSI prior-result artifact has an unknown schema")
    scored = False
    for result in prior["results"]:
        if int(result["task_id"]) != TASK_ID:
            continue
        if any(
            int(row["fold"]) == OUTER_FOLD
            and int(row.get("repeat", 0)) == OUTER_REPEAT
            and int(row.get("sample", 0)) == OUTER_SAMPLE
            for row in result["folds"]
        ):
            scored = True
            break
    if not scored:
        raise RuntimeError("RSSI diagnostic coordinate lacks a prior outcome")
    return {
        "coordinate": {
            "task_id": TASK_ID,
            "repeat": OUTER_REPEAT,
            "fold": OUTER_FOLD,
            "sample": OUTER_SAMPLE,
        },
        "registry_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "prior_result_sha256": hashlib.sha256(prior_payload).hexdigest(),
        "prior_outcome_exists": True,
        "fresh_claim_eligible": False,
    }


def _load_data():
    import openml

    task = openml.tasks.get_task(TASK_ID, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if int(dataset.dataset_id) != DATASET_ID or str(dataset.name) != TASK_NAME:
        raise RuntimeError("RSSI dataset identity changed")
    if str(task.target_name) != TARGET_NAME:
        raise RuntimeError("RSSI target changed")
    if X.shape != EXPECTED_SHAPE or task.get_split_dimensions() != EXPECTED_SPLIT_DIMENSIONS:
        raise RuntimeError("RSSI task shape or split dimensions changed")
    if any(bool(value) for value in categorical):
        raise RuntimeError("RSSI task unexpectedly contains categoricals")
    X_array = np.asarray(X, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
        raise RuntimeError("RSSI task unexpectedly contains non-finite values")

    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=OUTER_REPEAT,
        fold=OUTER_FOLD,
        sample=OUTER_SAMPLE,
    )
    inner_train, inner_validation = next(
        ShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE,
        ).split(X.iloc[outer_train])
    )
    fit_indices = np.asarray(outer_train)[inner_train]
    validation_indices = np.asarray(outer_train)[inner_validation]
    metadata = {
        "task_id": TASK_ID,
        "dataset_id": DATASET_ID,
        "dataset_name": TASK_NAME,
        "target_name": TARGET_NAME,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X_array, "<f8"),
        "y_sha256": _array_sha256(y_array, "<f8"),
        "outer_train_index_sha256": _array_sha256(outer_train, "<i8"),
        "outer_test_index_sha256": _array_sha256(outer_test, "<i8"),
        "shared_fit_index_sha256": _array_sha256(fit_indices, "<i8"),
        "shared_validation_index_sha256": _array_sha256(
            validation_indices, "<i8"
        ),
        "outer_train_rows": int(len(outer_train)),
        "shared_fit_rows": int(len(fit_indices)),
        "shared_validation_rows": int(len(validation_indices)),
        "outer_test_rows": int(len(outer_test)),
    }
    return (
        X,
        y,
        np.asarray(outer_train),
        np.asarray(outer_test),
        fit_indices,
        validation_indices,
        metadata,
    )


def _update_array_hash(digest, label: str, value) -> None:
    digest.update(label.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _model_sha256(core) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, "init", np.asarray([core.init_], dtype="<f8"))
    for index, tree in enumerate(core.trees_):
        digest.update(str(index).encode("ascii"))
        _update_array_hash(digest, "splits_feat", tree.splits_feat)
        _update_array_hash(digest, "splits_thr", tree.splits_thr)
        _update_array_hash(digest, "values", tree.values)
        _update_array_hash(
            digest,
            "linear_features",
            getattr(tree, "linear_features", getattr(tree, "lin_feats", None)),
        )
        _update_array_hash(
            digest,
            "linear_coefficients",
            getattr(
                tree, "linear_coefficients", getattr(tree, "lin_coef", None)
            ),
        )
    return digest.hexdigest()


def _record(
    arm: str,
    model,
    prediction,
    y_test,
    fit_seconds: float,
    *,
    library: str,
) -> dict[str, Any]:
    core = model.model_
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != np.asarray(y_test).shape or not np.all(
        np.isfinite(prediction)
    ):
        raise RuntimeError(f"{arm} produced invalid predictions")
    history = np.asarray(getattr(core, "valid_history_", ()), dtype=np.float64)
    borders = np.concatenate(
        [
            np.asarray(border, dtype=np.float64)
            for border in core.prep_.binner_.borders_
        ]
    )
    tree = core.trees_[0]
    linear_features = getattr(
        tree, "linear_features", getattr(tree, "lin_feats", None)
    )
    selected_linear = getattr(model, "linear_leaves_selected_", None)
    selected_cross = getattr(model, "cross_features_selected_", None)
    return {
        "arm": arm,
        "library": library,
        "fit_seconds": float(fit_seconds),
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "best_validation_rmse": (
            None if history.size == 0 else float(np.min(history))
        ),
        "test_rmse": float(
            mean_squared_error(
                np.asarray(y_test, dtype=np.float64), prediction
            )
            ** 0.5
        ),
        "prediction_sha256": _array_sha256(prediction, "<f8"),
        "validation_history_sha256": _array_sha256(history, "<f8"),
        "borders_sha256": _array_sha256(borders, "<f8"),
        "model_sha256": _model_sha256(core),
        "first_tree_splits_feat": np.asarray(tree.splits_feat).tolist(),
        "first_tree_splits_thr": np.asarray(tree.splits_thr).tolist(),
        "first_tree_linear_features": (
            None
            if linear_features is None
            else np.asarray(linear_features).tolist()
        ),
        "linear_leaves_selected": (
            None if selected_linear is None else bool(selected_linear)
        ),
        "cross_features_selected": (
            None if selected_cross is None else bool(selected_cross)
        ),
        "cross_pair_count": int(
            len(getattr(model, "cross_pairs_", None) or ())
        ),
    }


def _fit_darko(
    arm: str,
    X,
    y,
    outer_train,
    fit_indices,
    validation_indices,
    outer_test,
):
    from darkofit import DarkoRegressor

    common = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "min_child_weight": 1.0,
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
    }
    eval_set = None
    train_indices = outer_train
    if arm == "darko_default":
        params = {
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
        }
    elif arm == "darko_matched_auto10_linear":
        params = dict(
            common,
            linear_leaves=True,
            early_stopping=True,
            validation_fraction=0.10,
            use_best_model=True,
            refit=False,
        )
    elif arm == "darko_matched_auto20_linear":
        params = dict(
            common,
            linear_leaves=True,
            early_stopping=True,
            validation_fraction=0.20,
            use_best_model=True,
            refit=False,
        )
    elif arm in {"darko_shared_constant", "darko_shared_linear"}:
        params = dict(
            common,
            linear_leaves=arm.endswith("linear"),
            early_stopping=True,
            use_best_model=True,
            refit=False,
        )
        train_indices = fit_indices
        eval_set = (X.iloc[validation_indices], y.iloc[validation_indices])
    else:
        raise ValueError(f"unknown DarkoFit arm: {arm}")
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X.iloc[train_indices], y.iloc[train_indices], eval_set=eval_set)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = model.predict(X.iloc[outer_test])
    return _record(
        arm,
        model,
        prediction,
        y.iloc[outer_test],
        fit_seconds,
        library="darkofit",
    )


def _fit_chimera(
    arm: str,
    X,
    y,
    outer_train,
    fit_indices,
    validation_indices,
    outer_test,
):
    ChimeraBoostRegressor = _chimera_regressor_class()

    common = {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "min_child_weight": 1.0,
        "random_state": RANDOM_STATE,
        "thread_count": THREADS,
        "early_stopping": True,
    }
    train_indices = fit_indices
    eval_set = (X.iloc[validation_indices], y.iloc[validation_indices])
    if arm == "chimera_shared_constant":
        params = dict(
            common,
            linear_leaves=False,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_shared_linear":
        params = dict(
            common,
            linear_leaves=True,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_full_selector":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=False,
            selection_rounds=None,
        )
    elif arm == "chimera_capped_selector":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=False,
            selection_rounds=100,
        )
    elif arm == "chimera_full_product":
        params = dict(
            common,
            linear_leaves=None,
            cross_features=None,
            selection_rounds=None,
        )
    elif arm == "chimera_product":
        params = {
            "random_state": RANDOM_STATE,
            "thread_count": THREADS,
        }
        train_indices = outer_train
        eval_set = None
    else:
        raise ValueError(f"unknown ChimeraBoost arm: {arm}")
    model = ChimeraBoostRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(X.iloc[train_indices], y.iloc[train_indices], eval_set=eval_set)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = model.predict(X.iloc[outer_test])
    return _record(
        arm,
        model,
        prediction,
        y.iloc[outer_test],
        fit_seconds,
        library="chimeraboost",
    )


def _exact_pair(records: dict[str, dict[str, Any]], left: str, right: str):
    mismatches = [
        field
        for field in EXACT_FIELDS
        if records[left][field] != records[right][field]
    ]
    if mismatches:
        raise RuntimeError(
            f"{left} and {right} differ in exact fields: {mismatches}"
        )
    return {
        "left": left,
        "right": right,
        "exact_fields": list(EXACT_FIELDS),
        "exact": True,
    }


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return _is_hex_digest(value, 40)


def _is_sha256(value: Any) -> bool:
    return _is_hex_digest(value, 64)


def _valid_aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _analysis_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(
            left,
            right,
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _analysis_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _analysis_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("arm"), str)
        for row in rows
    ):
        raise RuntimeError("RSSI diagnosis arm ledger is invalid")
    records = {row["arm"]: row for row in rows}
    missing = sorted(set(ARMS) - set(records))
    extra = sorted(set(records) - set(ARMS))
    if missing or extra or len(rows) != len(ARMS):
        raise RuntimeError(f"arm set mismatch: missing={missing}, extra={extra}")
    for arm, row in records.items():
        expected_library = "darkofit" if arm.startswith("darko_") else "chimeraboost"
        if row.get("library") != expected_library:
            raise RuntimeError(f"{arm} has an invalid library ledger")
        numeric_fields = (
            row.get("fit_seconds"),
            row.get("test_rmse"),
            row.get("resolved_learning_rate"),
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in numeric_fields
        ):
            raise RuntimeError(f"{arm} has invalid numeric evidence")
        fit_seconds, test_rmse, learning_rate = map(float, numeric_fields)
        tree_count = row.get("fitted_tree_count")
        cross_pair_count = row.get("cross_pair_count")
        if (
            not math.isfinite(fit_seconds)
            or fit_seconds <= 0.0
            or not math.isfinite(test_rmse)
            or test_rmse <= 0.0
            or not math.isfinite(learning_rate)
            or learning_rate <= 0.0
            or not isinstance(tree_count, int)
            or isinstance(tree_count, bool)
            or tree_count <= 0
            or not isinstance(cross_pair_count, int)
            or isinstance(cross_pair_count, bool)
            or cross_pair_count < 0
        ):
            raise RuntimeError(f"{arm} has invalid numeric evidence")
        best = row.get("best_validation_rmse")
        if arm != "darko_default" and (
            not isinstance(best, (int, float))
            or isinstance(best, bool)
            or not math.isfinite(float(best))
            or float(best) <= 0.0
        ):
            raise RuntimeError(f"{arm} has invalid validation evidence")
        if arm == "darko_default" and best is not None:
            raise RuntimeError(f"{arm} has invalid validation evidence")
        for field in (
            "borders_sha256",
            "validation_history_sha256",
            "model_sha256",
            "prediction_sha256",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                raise RuntimeError(f"{arm} has an invalid fingerprint ledger")

    parity = [
        _exact_pair(records, "darko_shared_constant", "chimera_shared_constant"),
        _exact_pair(records, "darko_shared_linear", "chimera_shared_linear"),
        _exact_pair(
            records, "darko_matched_auto20_linear", "darko_shared_linear"
        ),
    ]
    for arm in ("chimera_full_selector", "chimera_capped_selector"):
        if not isinstance(records[arm].get("linear_leaves_selected"), bool):
            raise RuntimeError(f"{arm} lacks a resolved linear-leaf decision")
    for arm in ("chimera_full_product", "chimera_product"):
        if not isinstance(
            records[arm].get("linear_leaves_selected"), bool
        ) or not isinstance(records[arm].get("cross_features_selected"), bool):
            raise RuntimeError(f"{arm} lacks a resolved product decision")
    for arm, row in records.items():
        linear_selected = row.get("linear_leaves_selected")
        cross_selected = row.get("cross_features_selected")
        pair_count = row["cross_pair_count"]
        if arm.startswith("darko_") or arm in {
            "chimera_shared_constant",
            "chimera_shared_linear",
        }:
            expected_decisions = (None, None)
        elif arm in {"chimera_full_selector", "chimera_capped_selector"}:
            expected_decisions = (linear_selected, None)
        else:
            expected_decisions = (linear_selected, cross_selected)
        if (
            (linear_selected, cross_selected) != expected_decisions
            or (
                cross_selected is True
                and pair_count != EXPECTED_CROSS_PAIR_COUNT
            )
            or (cross_selected is not True and pair_count != 0)
        ):
            raise RuntimeError(f"{arm} has an invalid selector ledger")
    constant_best = records["chimera_shared_constant"]["best_validation_rmse"]
    linear_best = records["chimera_shared_linear"]["best_validation_rmse"]
    full_winner = "linear" if linear_best < constant_best else "constant"
    full_selected = (
        "linear"
        if records["chimera_full_selector"]["linear_leaves_selected"]
        else "constant"
    )
    if full_selected != full_winner:
        raise RuntimeError("full selector disagrees with forced full-budget race")
    capped_selected = (
        "linear"
        if records["chimera_capped_selector"]["linear_leaves_selected"]
        else "constant"
    )
    _exact_pair(
        records,
        "chimera_full_selector",
        f"chimera_shared_{full_selected}",
    )
    _exact_pair(
        records,
        "chimera_capped_selector",
        f"chimera_shared_{capped_selected}",
    )
    if (
        records["chimera_full_product"]["linear_leaves_selected"]
        != (full_selected == "linear")
        or records["chimera_product"]["linear_leaves_selected"]
        != (capped_selected == "linear")
    ):
        raise RuntimeError("product selector disagrees with its linear-leaf race")
    if not records["chimera_full_product"]["cross_features_selected"]:
        _exact_pair(
            records,
            "chimera_full_product",
            "chimera_full_selector",
        )

    product = records["chimera_product"]["test_rmse"]
    default = records["darko_default"]["test_rmse"]
    shared_constant = records["darko_shared_constant"]["test_rmse"]
    shared_linear = records["darko_shared_linear"]["test_rmse"]
    return {
        "claim_tier": "development_diagnostic_only",
        "fresh_claim_eligible": False,
        "parity_checks": parity,
        "forced_full_budget_validation_winner": full_winner,
        "full_selector_winner": full_selected,
        "capped_selector_winner": capped_selected,
        "capped_selector_disagrees_with_full": capped_selected != full_winner,
        "chimera_product_linear_selected": records["chimera_product"][
            "linear_leaves_selected"
        ],
        "chimera_product_cross_selected": records["chimera_product"][
            "cross_features_selected"
        ],
        "chimera_full_product_cross_selected": records[
            "chimera_full_product"
        ]["cross_features_selected"],
        "test_rmse_ratios": {
            "darko_default_over_chimera_product": float(default / product),
            "darko_shared_constant_over_chimera_product": float(
                shared_constant / product
            ),
            "darko_shared_linear_over_chimera_product": float(
                shared_linear / product
            ),
            "shared_linear_over_shared_constant": float(
                shared_linear / shared_constant
            ),
            "darko_auto10_linear_over_shared20_linear": float(
                records["darko_matched_auto10_linear"]["test_rmse"]
                / shared_linear
            ),
        },
        "diagnosis": [
            "matched_constant_engine_parity",
            "matched_linear_engine_parity",
            (
                "capped_linear_selection_misselection"
                if capped_selected != full_winner
                else "capped_linear_selection_agrees"
            ),
            (
                "product_cross_features_selected"
                if records["chimera_product"]["cross_features_selected"]
                else "product_cross_features_not_selected"
            ),
        ],
    }


def validate_artifact(
    artifact: Any,
    *,
    require_frozen: bool = True,
) -> dict[str, Any]:
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or isinstance(artifact.get("schema_version"), bool)
        or not _valid_aware_timestamp(artifact.get("created_at"))
    ):
        raise RuntimeError("RSSI diagnosis artifact has an unknown schema")
    protocol = artifact.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("name") != "rssi_linear_leaf_diagnosis"
        or protocol.get("path") != str(PROTOCOL.relative_to(ROOT))
        or protocol.get("sha256") != _sha256(PROTOCOL)
        or protocol.get("arms") != list(ARMS)
        or protocol.get("timing_claim_eligible") is not False
    ):
        raise RuntimeError("RSSI diagnosis protocol ledger changed")
    if artifact.get("spent_boundary") != _verify_spent_boundary():
        raise RuntimeError("RSSI diagnosis spent-boundary ledger changed")

    sources = artifact.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "darkofit",
        "chimeraboost",
    }:
        raise RuntimeError("RSSI diagnosis source ledger changed")
    for name, source in sources.items():
        if (
            not isinstance(source, dict)
            or source.get("clean") is not True
            or not isinstance(source.get("path"), str)
            or not Path(source["path"]).is_absolute()
            or not isinstance(source.get("branch"), str)
            or not source["branch"]
            or not _is_git_sha(source.get("head"))
        ):
            raise RuntimeError(f"RSSI diagnosis {name} source ledger changed")
    if sources["chimeraboost"]["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("RSSI diagnosis ChimeraBoost source ledger changed")

    data = artifact.get("data")
    integer_fields = {
        "task_id": TASK_ID,
        "dataset_id": DATASET_ID,
        "rows": EXPECTED_SHAPE[0],
        "features": EXPECTED_SHAPE[1],
    }
    if (
        not isinstance(data, dict)
        or data.get("dataset_name") != TASK_NAME
        or data.get("target_name") != TARGET_NAME
        or any(
            not isinstance(data.get(field), int)
            or isinstance(data.get(field), bool)
            or data[field] != expected
            for field, expected in integer_fields.items()
        )
        or any(
            not _is_sha256(data.get(field))
            for field in (
                "X_sha256",
                "y_sha256",
                "outer_train_index_sha256",
                "outer_test_index_sha256",
                "shared_fit_index_sha256",
                "shared_validation_index_sha256",
            )
        )
    ):
        raise RuntimeError("RSSI diagnosis data ledger changed")
    row_fields = (
        "outer_train_rows",
        "outer_test_rows",
        "shared_fit_rows",
        "shared_validation_rows",
    )
    if any(
        not isinstance(data.get(field), int)
        or isinstance(data.get(field), bool)
        or data[field] <= 0
        for field in row_fields
    ):
        raise RuntimeError("RSSI diagnosis split-row ledger changed")
    if (
        data["outer_train_rows"] + data["outer_test_rows"] != data["rows"]
        or data["shared_fit_rows"] + data["shared_validation_rows"]
        != data["outer_train_rows"]
    ):
        raise RuntimeError("RSSI diagnosis split-row ledger changed")

    rows = artifact.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("RSSI diagnosis result ledger changed")
    for row in rows:
        if not isinstance(row, dict) or any(
            not _is_sha256(row.get(field))
            for field in (
                "borders_sha256",
                "validation_history_sha256",
                "model_sha256",
                "prediction_sha256",
            )
        ):
            raise RuntimeError("RSSI diagnosis fingerprint ledger changed")
        split_features = row.get("first_tree_splits_feat")
        split_thresholds = row.get("first_tree_splits_thr")
        linear_features = row.get("first_tree_linear_features")
        if (
            not isinstance(split_features, list)
            or not split_features
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in split_features
            )
            or not isinstance(split_thresholds, list)
            or len(split_thresholds) != len(split_features)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in split_thresholds
            )
            or (
                linear_features is not None
                and (
                    not isinstance(linear_features, list)
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        for value in linear_features
                    )
                )
            )
        ):
            raise RuntimeError("RSSI diagnosis first-tree ledger changed")
    frozen_evidence = {
        "protocol": protocol,
        "source_revisions": {
            name: {
                "head": source["head"],
                "branch": source["branch"],
                "clean": source["clean"],
            }
            for name, source in sources.items()
        },
        "spent_boundary": artifact["spent_boundary"],
        "data": data,
        "results": [
            {
                key: value
                for key, value in row.items()
                if key != "fit_seconds"
            }
            for row in rows
        ],
    }
    if _json_sha256(frozen_evidence) != FROZEN_EVIDENCE_SHA256:
        raise RuntimeError("RSSI diagnosis frozen evidence ledger changed")
    analysis = analyze(rows)
    if not _analysis_equal(artifact.get("analysis"), analysis):
        raise RuntimeError("RSSI diagnosis stored analysis is not reproducible")
    if (
        require_frozen
        and _json_sha256(artifact) != FROZEN_ARTIFACT_SHA256
    ):
        raise RuntimeError("RSSI diagnosis frozen artifact changed")
    return analysis


def run(output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    protocol_sha256 = _sha256(PROTOCOL)
    darko_source = _source_state(ROOT)
    chimera_source = _source_state(
        CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
    )
    spent = _verify_spent_boundary()
    (
        X,
        y,
        outer_train,
        outer_test,
        fit_indices,
        validation_indices,
        data,
    ) = _load_data()

    rows = []
    for arm in ARMS:
        if arm.startswith("darko_"):
            row = _fit_darko(
                arm,
                X,
                y,
                outer_train,
                fit_indices,
                validation_indices,
                outer_test,
            )
        else:
            row = _fit_chimera(
                arm,
                X,
                y,
                outer_train,
                fit_indices,
                validation_indices,
                outer_test,
            )
        rows.append(row)

    analysis = analyze(rows)
    if _source_state(ROOT) != darko_source:
        raise RuntimeError("DarkoFit source state changed during diagnosis")
    if (
        _source_state(
            CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
        )
        != chimera_source
    ):
        raise RuntimeError("ChimeraBoost source state changed during diagnosis")
    if _sha256(PROTOCOL) != protocol_sha256:
        raise RuntimeError("RSSI diagnosis protocol changed during diagnosis")
    if _verify_spent_boundary() != spent:
        raise RuntimeError("RSSI spent boundary changed during diagnosis")

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "rssi_linear_leaf_diagnosis",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": protocol_sha256,
            "arms": list(ARMS),
            "timing_claim_eligible": False,
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "spent_boundary": spent,
        "data": data,
        "results": rows,
        "analysis": analysis,
    }
    validate_artifact(artifact, require_frozen=False)
    _atomic_create(
        output,
        (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
