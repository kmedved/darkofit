"""Contract tests for the ChimeraBoost creator basketball benchmark."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import run_basketball_creator_benchmark as basketball  # noqa: E402


def _toy_frame():
    rows = []
    for team_index, team in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        for player in range(2):
            row = {feature: team_index + player + 1 for feature in basketball.FEATURES}
            row.update(
                {
                    "Tm": team,
                    "MP": 600 + player,
                    "G": 20,
                    "GS": 10 if player else 9,
                    "MPG": -1.0,
                    "starter": -1,
                }
            )
            rows.append(row)
    rows.append(
        {
            **{feature: 0 for feature in basketball.FEATURES},
            "Tm": "ZZZ",
            "MP": 500,
            "G": 20,
            "GS": 20,
            "MPG": -1.0,
            "starter": -1,
        }
    )
    return pd.DataFrame(rows)


def test_creator_cv_is_explicit_unshuffled_ten_fold():
    cv = basketball.creator_cv()

    assert cv.n_splits == 10
    assert cv.shuffle is False
    assert cv.random_state is None


def test_build_estimator_exact_default_kwargs(monkeypatch, tmp_path):
    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_darkofit = type(sys)("darkofit")
    fake_darkofit.DarkoRegressor = FakeRegressor
    fake_chimeraboost = type(sys)("chimeraboost")
    fake_chimeraboost.ChimeraBoostRegressor = FakeRegressor
    fake_catboost = type(sys)("catboost")
    fake_catboost.CatBoostRegressor = FakeRegressor
    monkeypatch.setitem(sys.modules, "darkofit", fake_darkofit)
    monkeypatch.setitem(sys.modules, "chimeraboost", fake_chimeraboost)
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    darko = basketball.build_estimator("darkofit_default", tmp_path)
    chimera = basketball.build_estimator("chimeraboost_default", tmp_path)
    ensemble = basketball.build_estimator("chimeraboost_ensemble5", tmp_path)
    catboost = basketball.build_estimator("catboost_default", tmp_path)

    assert darko.kwargs == {"random_state": 4}
    assert chimera.kwargs == {"random_state": 4}
    assert ensemble.kwargs == {"random_state": 4, "n_ensembles": 5}
    assert catboost.kwargs == {
        "random_state": 4,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }

    steady_catboost = basketball.build_estimator(
        "catboost_default", tmp_path, lane="steady"
    )
    assert steady_catboost.kwargs["thread_count"] == max(
        1, basketball.os.cpu_count() or 1
    )


def test_prepare_creator_data_uses_alphabetical_team_holdout(monkeypatch):
    frame = _toy_frame()
    monkeypatch.setattr(basketball, "X_TRAIN_SHA256", "skip")
    monkeypatch.setattr(basketball, "Y_TRAIN_SHA256", "skip")

    original_sha = basketball.sha256_bytes
    digests = iter(("skip", "skip"))
    monkeypatch.setattr(basketball, "sha256_bytes", lambda _: next(digests))
    X, y, metadata = basketball.prepare_creator_data(frame)
    monkeypatch.setattr(basketball, "sha256_bytes", original_sha)

    assert metadata["test_teams"] == ["AAA", "BBB"]
    assert metadata["train_team_count"] == 4
    assert metadata["test_team_count"] == 2
    assert metadata["train_rows"] == 8
    assert metadata["test_rows"] == 4
    assert list(X.columns) == list(basketball.FEATURES)
    np.testing.assert_allclose(y.to_numpy(), [30.0, 30.05] * 4)


def test_prepare_creator_data_rejects_processed_fingerprint(monkeypatch):
    frame = _toy_frame()
    monkeypatch.setattr(basketball, "X_TRAIN_SHA256", "not-the-real-hash")

    with pytest.raises(RuntimeError, match="processed creator data fingerprint"):
        basketball.prepare_creator_data(frame)


def test_parse_args_freezes_lane_defaults(tmp_path):
    args = basketball.parse_args(
        [
            "--lane",
            "author",
            "--data-cache",
            str(tmp_path / "data.csv"),
            "--chimeraboost-repo",
            str(tmp_path / "chimera"),
        ]
    )

    assert args.arms == list(basketball.ARM_ORDER)
    assert args.lane == "author"
    assert args.output.name == "results-author.json"
    assert args.allow_dirty_source is False
    assert args.allow_chimeraboost_drift is False


def test_worker_lane_policy_and_mean(monkeypatch, tmp_path):
    class FakeEstimator:
        def get_params(self, deep=False):
            return {}

    frame = _toy_frame()
    X = frame.loc[:9, basketball.FEATURES]
    y = pd.Series(np.linspace(0.0, 1.0, len(X)))
    estimator = FakeEstimator()
    seen = {}

    monkeypatch.setattr(
        basketball,
        "load_raw_data",
        lambda _: (frame, {"sha256": basketball.DATA_SHA256}),
    )
    monkeypatch.setattr(
        basketball,
        "prepare_creator_data",
        lambda _: (X, y, {"train_rows": len(X)}),
    )
    monkeypatch.setattr(basketball, "build_estimator", lambda *_: estimator)
    monkeypatch.setattr(basketball, "_assert_estimator_source", lambda *_: None)
    monkeypatch.setattr(
        basketball,
        "_module_details",
        lambda _: {"package": "fake"},
    )

    def fake_cross_val_score(model, features, target, **kwargs):
        seen.update(kwargs)
        return np.linspace(0.1, 1.0, 10)

    monkeypatch.setattr(basketball, "cross_val_score", fake_cross_val_score)
    monkeypatch.setattr(
        basketball,
        "_jsonable",
        lambda _: {},
    )
    result = basketball.run_worker(
        "darkofit_default", "author", tmp_path / "data.csv", tmp_path
    )

    assert seen["scoring"] == "r2"
    assert seen["n_jobs"] == -1
    assert isinstance(seen["cv"], basketball.KFold)
    assert seen["error_score"] == "raise"
    assert result["mean_r2"] == pytest.approx(0.55)
    assert len(result["fold_scores"]) == 10


def test_worker_process_forces_chimeraboost_import_warmup_off(
    monkeypatch, tmp_path
):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs["env"])
        payload = {
            "arm": "chimeraboost_default",
            "mean_r2": 0.5,
            "wall_seconds": 1.0,
        }
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": basketball.WORKER_RESULT_PREFIX + str_json(payload),
                "stderr": "",
            },
        )()

    def str_json(value):
        import json

        return json.dumps(value)

    monkeypatch.setenv("CHIMERABOOST_WARMUP", "background")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("NUMBA_CPU_NAME", "generic")
    monkeypatch.setenv("NUMBA_CPU_FEATURES", "-neon")
    monkeypatch.setenv("NUMBA_THREADING_LAYER_PRIORITY", "omp tbb workqueue")
    monkeypatch.setenv("NUMBA_BOUNDSCHECK", "1")
    monkeypatch.setenv("JOBLIB_START_METHOD", "fork")
    monkeypatch.setenv("KMP_AFFINITY", "compact")
    monkeypatch.setattr(basketball.subprocess, "run", fake_run)
    args = basketball.parse_args(
        [
            "--chimeraboost-repo",
            str(tmp_path / "chimera"),
            "--data-cache",
            str(tmp_path / "data.csv"),
        ]
    )

    basketball._run_worker_process(args, "chimeraboost_default")

    assert seen["CHIMERABOOST_WARMUP"] == "0"
    for key in basketball.THREAD_LIMIT_ENV_KEYS:
        assert seen[key] == "1"
    assert seen["NUMBA_DISABLE_JIT"] == "0"
    assert seen["ENABLE_IPC"] == "1"
    assert seen["JOBLIB_MULTIPROCESSING"] == "1"
    assert seen["LOKY_PICKLER"] == "cloudpickle"
    assert seen["PYTHONHASHSEED"] == "0"
    assert "NUMBA_CPU_NAME" not in seen
    assert "NUMBA_CPU_FEATURES" not in seen
    assert "NUMBA_THREADING_LAYER_PRIORITY" not in seen
    assert "NUMBA_BOUNDSCHECK" not in seen
    assert "JOBLIB_START_METHOD" not in seen
    assert "KMP_AFFINITY" not in seen
    assert "CHIMERABOOST_WARMUP" in basketball.THREAD_ENV_KEYS


def test_source_boundary_rejects_changed_head():
    expected = {
        "darkofit": {
            "path": "/darko",
            "head": "aaa",
            "branch": "main",
            "clean": True,
            "status": [],
        },
        "chimeraboost": {
            "path": "/chimera",
            "head": "bbb",
            "branch": "main",
            "clean": True,
            "status": [],
        },
    }
    observed = {
        name: dict(state) for name, state in expected.items()
    }
    observed["darkofit"]["head"] = "ccc"

    with pytest.raises(RuntimeError, match="changed after arm"):
        basketball._assert_sources_unchanged(
            expected, observed, boundary="after arm"
        )


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        (
            "https://user:secret-token@github.com/owner/repo.git?token=other",
            "https://github.com/owner/repo.git",
        ),
        ("git@github.com:owner/repo.git", "github.com:owner/repo.git"),
        (
            "helper::https://user:secret@github.com/owner/repo.git",
            "helper::<redacted>",
        ),
        (
            "https://user:secret@[2001:db8::1]/owner/repo.git",
            "https://[2001:db8::1]/owner/repo.git",
        ),
        ("/local/repo", "/local/repo"),
    ],
)
def test_git_remote_sanitization(remote, expected):
    assert basketball.sanitize_git_remote(remote) == expected


def test_estimator_source_must_live_in_attested_checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "installed" / "darkofit" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.touch()
    fake_darkofit = type(sys)("darkofit")
    fake_darkofit.__file__ = str(outside)
    monkeypatch.setitem(sys.modules, "darkofit", fake_darkofit)
    monkeypatch.setattr(basketball, "REPO_ROOT", checkout)
    monkeypatch.setattr(
        basketball,
        "_git_output",
        lambda *args, **kwargs: str(checkout),
    )

    with pytest.raises(RuntimeError, match="outside the attested checkout"):
        basketball._assert_estimator_source("darkofit_default", tmp_path)


def test_baseline_eligibility_marks_escape_hatches():
    args = basketball.parse_args([])
    sources = {
        "darkofit": {"clean": True, "head": "darko"},
        "chimeraboost": {
            "clean": True,
            "head": basketball.CHIMERABOOST_BASELINE_REVISION,
        },
    }

    baseline = basketball._baseline_eligibility(args, sources)
    assert baseline["eligible"] is True
    assert baseline["reasons"] == []

    args.allow_chimeraboost_drift = True
    exploratory = basketball._baseline_eligibility(args, sources)
    assert exploratory["eligible"] is False
    assert exploratory["reasons"] == ["chimeraboost_drift_override_enabled"]
    assert exploratory["overrides"]["allow_chimeraboost_drift"] is True


def test_atomic_write_rejects_symlink_destination(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    link = tmp_path / "result.json"
    link.symlink_to(victim)

    with pytest.raises(RuntimeError, match="symlink destination"):
        basketball._atomic_write_bytes(link, b"replace")

    assert victim.read_text(encoding="utf-8") == "keep"
    assert link.is_symlink()


def test_parse_args_preserves_lexical_output_symlink(tmp_path):
    victim = tmp_path / "victim.json"
    victim.touch()
    link = tmp_path / "output.json"
    link.symlink_to(victim)

    args = basketball.parse_args(["--output", str(link)])

    assert args.output == link
    assert args.output.is_symlink()
