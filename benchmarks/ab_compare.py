"""Bitwise A/B comparison harness for ChimeraBoost revisions.

Run from any checkout with two repo paths:

    python benchmarks/ab_compare.py /tmp/chimera-base /path/to/chimeraboost

The harness executes each repo in a separate Python subprocess, saves
predictions and ``model_.train_history_`` for a deterministic case matrix, and
then compares every saved array with ``np.array_equal``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


CASES = [
    {"id": "reg_catboost_ordered", "task": "regression", "kwargs": {"tree_mode": "catboost", "ordered_boosting": True}},
    {"id": "reg_catboost_plain", "task": "regression", "kwargs": {"tree_mode": "catboost", "ordered_boosting": False}},
    {"id": "reg_catboost_mae", "task": "regression", "kwargs": {"tree_mode": "catboost", "loss": "MAE"}},
    {"id": "reg_lightgbm", "task": "regression", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7}},
    {"id": "reg_hybrid", "task": "regression", "kwargs": {"tree_mode": "hybrid", "num_leaves": 7}},
    {"id": "reg_depthwise", "task": "regression", "kwargs": {"tree_mode": "depthwise"}},
    {"id": "reg_weighted", "task": "regression", "weighted": True, "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7}},
    {"id": "reg_goss", "task": "regression", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7, "sampling": "goss", "top_rate": 0.25, "other_rate": 0.25}},
    {"id": "reg_mvs", "task": "regression", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7, "sampling": "mvs", "subsample": 0.65}},
    {"id": "reg_bayesian", "task": "regression", "kwargs": {"tree_mode": "catboost", "bootstrap_type": "bayesian", "bagging_temperature": 1.0}},
    {"id": "bin_lightgbm", "task": "binary", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7}},
    {"id": "bin_weighted", "task": "binary", "weighted": True, "kwargs": {"tree_mode": "hybrid", "num_leaves": 7}},
    {"id": "mc_per_class_catboost", "task": "multiclass", "kwargs": {"tree_mode": "catboost"}},
    {"id": "mc_per_class_lightgbm", "task": "multiclass", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7, "multiclass_tree_strategy": "per_class"}},
    {"id": "mc_shared_vector", "task": "multiclass", "kwargs": {"tree_mode": "lightgbm", "num_leaves": 7, "multiclass_tree_strategy": "shared_vector", "ordered_boosting": False}},
    {"id": "mc_depthwise", "task": "multiclass", "kwargs": {"tree_mode": "depthwise"}},
    {"id": "mc_weighted", "task": "multiclass", "weighted": True, "kwargs": {"tree_mode": "catboost"}},
]


RUNNER = r"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

repo = Path(sys.argv[1]).resolve()
case_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
sys.path.insert(0, str(repo))

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

case = json.loads(case_path.read_text())
rng = np.random.default_rng(12345)
task = case["task"]
kwargs = dict(case["kwargs"])
common = dict(iterations=18, depth=3, max_bins=48, random_state=7)
common.update(kwargs)

if task == "regression":
    X, y = make_regression(
        n_samples=900, n_features=10, n_informative=7, noise=4.0,
        random_state=101,
    )
    Est = ChimeraBoostRegressor
elif task == "binary":
    X, y = make_classification(
        n_samples=900, n_features=12, n_informative=7, n_redundant=2,
        flip_y=0.03, random_state=102,
    )
    Est = ChimeraBoostClassifier
else:
    X, y = make_classification(
        n_samples=900, n_features=12, n_informative=8, n_redundant=2,
        n_classes=3, n_clusters_per_class=1, flip_y=0.03, random_state=103,
    )
    Est = ChimeraBoostClassifier

stratify = y if task != "regression" else None
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=5, stratify=stratify
)
sample_weight = None
if case.get("weighted", False):
    sample_weight = rng.uniform(0.5, 2.0, size=ytr.shape[0])

model = Est(**common)
model.fit(Xtr, ytr, sample_weight=sample_weight)
if task == "regression":
    pred = model.predict(Xte)
else:
    pred = model.predict_proba(Xte)
history = np.asarray(model.model_.train_history_, dtype=np.float64)
np.savez_compressed(out_path, pred=np.asarray(pred), train_history=history)
"""


def _run_repo(repo, cases, out_dir, python):
    repo = Path(repo).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    for case in cases:
        case_path = out_dir / f"{case['id']}.json"
        out_path = out_dir / f"{case['id']}.npz"
        case_path.write_text(json.dumps(case))
        subprocess.run(
            [python, "-c", RUNNER, str(repo), str(case_path), str(out_path)],
            check=True,
            cwd=str(repo),
            env=env,
        )
        outputs[case["id"]] = out_path
    return outputs


def _compare_outputs(left, right):
    failures = []
    for case_id in sorted(left):
        with np.load(left[case_id], allow_pickle=False) as a, np.load(
            right[case_id], allow_pickle=False
        ) as b:
            for key in ("pred", "train_history"):
                if not np.array_equal(a[key], b[key]):
                    failures.append((case_id, key, a[key].shape, b[key].shape))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        out_root = args.out or Path(tmp)
        out_root.mkdir(parents=True, exist_ok=True)
        left = _run_repo(args.baseline, CASES, out_root / "baseline", args.python)
        right = _run_repo(args.candidate, CASES, out_root / "candidate", args.python)
        failures = _compare_outputs(left, right)
    if failures:
        for case_id, key, left_shape, right_shape in failures:
            print(
                f"DIFF {case_id} {key}: baseline {left_shape}, "
                f"candidate {right_shape}",
                file=sys.stderr,
            )
        return 1
    print(f"ab_compare clean: {len(CASES)} cases bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
