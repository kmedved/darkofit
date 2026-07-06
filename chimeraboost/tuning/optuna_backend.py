"""Small Optuna integration layer.

The rest of ChimeraBoost should stay importable without Optuna installed.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def import_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "ChimeraBoost tuning requires Optuna. Install it with "
            "`pip install chimeraboost[tuning]`."
        ) from exc
    return optuna


@dataclass(frozen=True)
class StorageConfig:
    storage: Any
    storage_kind: str
    storage_url: str | None


def default_journal_path(study_name):
    name = study_name or f"chimeraboost-study-{os.getpid()}"
    stem = Path(str(name)).name
    if stem in {"", ".", ".."}:
        stem = f"chimeraboost-study-{os.getpid()}"
    tempdir = Path(tempfile.gettempdir()).resolve()
    path = (tempdir / f"{stem}.optuna-journal.log").resolve()
    if path.parent != tempdir:
        path = tempdir / f"chimeraboost-study-{os.getpid()}.optuna-journal.log"
    return str(path)


def make_storage(storage, *, n_workers=1, study_name=None, resume=True):
    """Return Optuna storage plus user-facing metadata.

    ``journal:///path`` is the preferred same-machine multiprocessing form.
    String RDB URLs are passed through to Optuna unchanged. In-memory storage is
    allowed only when a single process can own the study.
    """
    optuna = import_optuna()
    if storage is None:
        if n_workers and int(n_workers) > 1:
            path = default_journal_path(study_name)
            _remove_journal_if_fresh(path, resume)
            backend = _journal_file_backend(path)
            return StorageConfig(
                optuna.storages.JournalStorage(backend),
                "journal",
                f"journal://{path}",
            )
        return StorageConfig(None, "in_memory", None)

    if isinstance(storage, str) and storage.startswith("journal://"):
        path = storage[len("journal://"):]
        backend = _journal_file_backend(path)
        return StorageConfig(
            optuna.storages.JournalStorage(backend),
            "journal",
            storage,
        )

    if isinstance(storage, str) and storage.startswith("sqlite") and n_workers > 1:
        raise ValueError(
            "SQLite storage is not supported with n_workers > 1; use "
            "journal:///path or a server RDB storage."
        )

    if not isinstance(storage, str) and n_workers > 1:
        raise ValueError(
            "n_workers > 1 requires reconstructable Optuna storage; use "
            "None, journal:///path, or a server RDB storage URL instead of a "
            "custom storage object."
        )

    kind = "rdb" if isinstance(storage, str) else type(storage).__name__
    return StorageConfig(storage, kind, storage if isinstance(storage, str) else None)


def create_study(*, storage_config, study_name, resume, sampler, pruner,
                 sampler_seed=None):
    optuna = import_optuna()
    if sampler is None and sampler_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    if pruner is None:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(
        direction="minimize",
        storage=storage_config.storage,
        study_name=study_name,
        load_if_exists=bool(resume),
        sampler=sampler,
        pruner=pruner,
    )


def load_study(*, storage_config, study_name):
    optuna = import_optuna()
    return optuna.load_study(study_name=study_name, storage=storage_config.storage)


def _journal_file_backend(path):
    from optuna.storages.journal import JournalFileBackend

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return JournalFileBackend(path)


def _remove_journal_if_fresh(path, resume):
    if resume:
        return
    journal_path = Path(path)
    if journal_path.exists() and journal_path.is_file():
        journal_path.unlink()
