"""Public entry points: deterministic build, harness builders, hashing.

Keys look like "syn:df1/031". Content is a pure function of (VERSION, id):
harness seed and --scale are deliberately ignored so per-dataset pairing in
compare_runs.py always compares identical data, and a generator change forces
a new version namespace instead of silently changing bytes under an old key.

Modified by the DarkoFit project from ChimeraBoost 0.15.0 commit 851ab7f.
"""
import functools
import hashlib
import json
from dataclasses import asdict

import numpy as np

from . import emit as _emit
from . import recipe as _recipe

_PREFIX = "syn:"


def key_for(dataset_id, version=_recipe.VERSION):
    return f"{_PREFIX}{version}/{dataset_id:03d}"


def parse_key(key):
    if not key.startswith(_PREFIX) or "/" not in key:
        raise ValueError(f"not a synthgen key: {key!r}")
    version, _, num = key[len(_PREFIX):].partition("/")
    if version != _recipe.VERSION:
        raise ValueError(
            f"key {key!r} is generator version {version!r} but this code is "
            f"{_recipe.VERSION!r} -- regenerate the suite (no silent cross-version reuse)")
    return version, int(num)


def _streams(dataset_id):
    ss = np.random.SeedSequence([_recipe.VERSION_SEED, dataset_id])
    return ss.spawn(2)  # (recipe stream, emit stream)


def sample_recipe(dataset_id):
    rs_recipe, _ = _streams(dataset_id)
    return _recipe.sample_recipe(dataset_id, np.random.default_rng(rs_recipe))


@functools.lru_cache(maxsize=2)
def build_dataset(key):
    """(X, y, cat_idx_or_None, task, meta) for a synthgen key. Deterministic."""
    _, dataset_id = parse_key(key)
    rs_recipe, rs_emit = _streams(dataset_id)
    rec = _recipe.sample_recipe(dataset_id, np.random.default_rng(rs_recipe))
    return _emit.emit_dataset(rec, rs_emit)


def make_builder(key):
    """Harness-convention builder(scale, rng) -> (X, y, cat, task). Both args
    are ignored on purpose (see module docstring)."""
    def _builder(scale, rng, _key=key):  # noqa: ARG001 - harness signature
        X, y, cat, task, _meta = build_dataset(_key)
        return X, y, cat, task
    return _builder


def recipe_meta(key):
    """Recipe/realized factors for the datasets-meta JSON (LRU-cheap)."""
    return dict(build_dataset(key)[4])


def _canonical_recipe_value(value):
    if isinstance(value, dict):
        return {
            key: _canonical_recipe_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_recipe_value(item) for item in value]
    if isinstance(value, float):
        rounded = round(value, 12)
        return 0.0 if rounded == 0.0 else rounded
    return value


def hash_recipe(key):
    """Cross-platform SHA-256 of the sampled recipe and versioned seed state.

    Dataset-content hashes intentionally include floating transforms and are
    therefore reference-platform tripwires. This narrower hash covers the
    NumPy distribution draws that select every recipe factor without treating
    architecture-level floating reduction differences as generator drift.
    """
    _, dataset_id = parse_key(key)
    recipe = sample_recipe(dataset_id)
    payload = {
        "hash_schema": "synthgen-recipe-v1",
        "key": key,
        "recipe": _canonical_recipe_value(asdict(recipe)),
        "seed_entropy": [_recipe.VERSION_SEED, dataset_id],
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def task_of(key):
    """Task type from the recipe alone (no data generation)."""
    _, dataset_id = parse_key(key)
    return sample_recipe(dataset_id).task


def hash_dataset(key):
    """sha256 of canonicalized content -- the RNG-stream-drift tripwire."""
    X, y, cat, task, meta = build_dataset(key)
    h = hashlib.sha256()
    cat_set = set(cat or [])
    for j in range(X.shape[1]):
        if j in cat_set:
            h.update("\x1f".join(str(v) for v in X[:, j]).encode())
        else:
            h.update(np.round(X[:, j].astype(np.float64), 6).tobytes())
    if task == "regression":
        h.update(np.round(np.asarray(y, dtype=np.float64), 6).tobytes())
    else:
        h.update(np.asarray(y, dtype=np.int64).tobytes())
    h.update(repr(sorted(meta.items())).encode())
    return h.hexdigest()
