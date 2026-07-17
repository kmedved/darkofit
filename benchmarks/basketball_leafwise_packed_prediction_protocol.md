# Basketball leafwise packed-prediction confirmation protocol

## Question and scope

Can DarkoFit eliminate the measured scalar leafwise prediction dispatch
overhead with a bounded packed explicit-node route while preserving every
prediction bit, large-batch throughput, fitted model, serialized model, and
basketball/cold-player behavior?

This is one prediction-only engine lever:

```text
bounded 64-row packed scalar explicit-node prediction
```

It changes no fit, tree, split, leaf value, loss, preprocessing, learning-rate
policy, early stopping, refit behavior, categorical representation, model
default, public API, or serialization schema. It does not reopen the rejected
constant-leaf oblivious work-router campaign. That earlier campaign changed
the serial/parallel cutoff for `FlatObliviousEnsemble`; this campaign concerns
only scalar `FlatNonObliviousEnsemble` prediction for fitted
`tree_mode="lightgbm"` models.

Basketball remains the primary and fatal boundary. A failure stops this route
without a TabArena, CTR23 development, or lockbox run.

## Profile and opportunity score

Exploration used creator fold 0 only, a fixed 1,000-tree scalar leafwise model,
and the frozen Python 3.12 basketball runtime. The current path calls
`NonObliviousTree.add_predict` once per tree. A ten-prediction `cProfile` run
on the 525-row fold attributed 0.081 of 0.082 seconds to 10,000
`add_predict` calls; 10,000 repeated `get_num_threads`/Numba dispatch checks
accounted for 0.006 seconds within that path.

Direct reciprocal nanosecond timing was used because `hyperfine` and `py-spy`
are not installed on the frozen host. All arms were warmed, repeated, and
required array-exact output. With 1,000 trees and two resolved threads, a
64-row packed kernel had these exploratory candidate/current core ratios:

| Rows | Basketball view | Candidate/current |
| ---: | --- | ---: |
| 127 | serial-sized control | 0.471 |
| 525 | creator fold 0 | 0.657 |
| 585 | cold players | 0.607 |
| 2,409 | held teams | 0.635 |
| 8,192 | repeated fold rows | 0.507 |
| 32,768 | repeated fold rows | 0.914 |
| 65,536 | repeated fold rows | 1.092 |
| 100,000 | repeated fold rows | 1.194 |

The existing 256-row packed blocks produced a three-block scheduling cliff at
the 525-row shape and were 1.038 times current. Reducing the explicit-node
block to 64 rows removed that cliff. At four, eight, and eighteen threads the
64-row route became progressively stronger; the only measured regression
region was the large-batch, low-thread case.

The opportunity score is `5 * 5 / 2 = 12.5`: very high impact in the exact
leafwise inference path that blocked the A10 performance claim, very high
confidence because the packed representation and exact kernels already ship,
and low implementation effort. This exceeds the required score of 2.0.

## Frozen source, comparator, license, and data

- Pre-protocol DarkoFit source:
  `96413f2c71faf4fd4b2caf05c411c661e5958f21`.
- Pre-protocol `darkofit/` package Git tree:
  `1a60b529c5f5d09920d81338406b491fb7275e3a`.
- Pre-protocol file SHA-256 values:
  - `darkofit/flat_model.py`:
    `98327432772d63ed01b372071ef84904dfba6a209f71f2912ea51b09ce9bf93d`;
  - `darkofit/booster.py`:
    `eb5f363d5558c97cf708634b83e16a283d9b8a03979ae88187b0b0055114da5c`;
  - `darkofit/tree.py`:
    `40362d75f599268c4067a0fd61db3f5c3841d18e088c3cc261ef702218d1f508`.
- The local ChimeraBoost checkout, `origin/main`, and `upstream/main` are clean
  and equal at v0.15.0 commit
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`. ChimeraBoost is provenance
  context, not a timing arm: it has no leafwise tree kind to compare
  algorithm-for-algorithm.
- Both repositories carry the same Apache-2.0 `LICENSE` bytes, SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
  The candidate adapts DarkoFit's already-shipped packed explicit-node kernel;
  it copies no literal ChimeraBoost source and requires no new `NOTICE`
  attribution.
- Basketball CSV SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- Creator feature fingerprint:
  `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`.
- Creator target SHA-256:
  `7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf`.
- Creator split fingerprint:
  `7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea`.
- Cold-player mask SHA-256:
  `e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19`.
- Shared support SHA-256 values:
  - `benchmarks/basketball_harness.py`:
    `40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1`;
  - `benchmarks/basketball_guardrails.py`:
    `4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52`;
  - `benchmarks/run_basketball_creator_benchmark.py`:
    `9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec`.
- Phase 0 golden manifest SHA-256:
  `2443509cded5e8ec3a725b50ce9fcc403be4c3ad9339260a9ccb85dec3ca17ff`.

The frozen runtime is Python `3.12.13` on
`macOS-26.5.2-arm64-arm-64bit`, Apple M5 Max, with 18 logical CPUs. Exact
package versions are NumPy `2.4.6`, Numba `0.66.0`, llvmlite `0.48.0`,
pandas `3.0.3`, scikit-learn `1.9.0`, SciPy `1.18.0`, joblib `1.5.3`, and
threadpoolctl `3.6.0`. The strict-test packages provisioned into that
environment are pytest `9.1.1`, iniconfig `2.3.0`, packaging `26.2`, pluggy
`1.6.0`, and Pygments `2.20.0`.

The formal run must use clean committed `main`, equal to `origin/main`,
descended from the pre-protocol commit. The runner must bind its final source,
the final candidate package tree, this protocol, support files, runtime,
dataset, split, and model configuration by hashes and explicit checks. The
artifact is create-only.

## Single candidate

Use a dedicated scalar explicit-node packed block size of 64 rows. The
existing oblivious, linear-oblivious, levelwise, per-class multiclass,
shared-vector multiclass, and distributional routes remain unchanged.

`flat_predict_preferred` may select a scalar
`FlatNonObliviousEnsemble` only when all conditions are true:

```text
numba_threads == 2
tree_count >= 5
row_count * tree_count >= 32768
row_count <= 32768
class_ids is None
packed values are one-dimensional
resolved_tree_mode == "lightgbm"
```

The dimensional conditions exclude per-class and shared-vector multiclass and
all distributional heads. The explicit fitted-mode condition excludes hybrid
models, which can also produce `FlatNonObliviousEnsemble`. The tree-count and
work floors preserve small-forest startup behavior. The row ceiling preserves
the current per-tree large-batch path where it measured faster.

The work threshold, block size, and row ceiling are frozen before candidate
implementation. Do not tune them on confirmation fold 1.

Selection must bypass the existing independent
`_PARALLEL_MIN_ROWS == 8192` check. Add one scalar-only internal entry point
on `FlatNonObliviousEnsemble` that calls a new, separate
`_flat_nonoblivious_scalar_add_parallel` kernel with its own frozen 64-row
block constant. Do not change `_ROW_BLOCK`, `_flat_nonoblivious_add_parallel`,
or any caller of that existing 256-row kernel. The selected scalar path in
`GradientBoosting.predict_raw` must call the new entry point; the existing
`FlatNonObliviousEnsemble.add_predict` threshold and every class-major,
multiclass, distributional, oblivious, linear-oblivious, and levelwise entry
point remain unchanged. Formal route instrumentation must separately record
both the outer `flat_predict_preferred` decision and entry into the new
scalar-only parallel kernel. Observing only the outer packed/fallback
decision is insufficient.

The exact two-thread condition deliberately limits authorization to the
profiled and formally tested basketball lane. One, four, eight, eighteen, and
all other resolved thread counts retain the pre-change path. A broader
parallel route requires a separate protocol.

## Frozen model and cases

Fit the complete 5,241-row creator training view once:

```text
DarkoRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=1,
    max_bins=128,
    tree_mode="lightgbm",
    ordered_boosting=False,
    use_best_model=False,
    early_stopping=False,
    thread_count=18,
    random_state=4,
    diagnostic_warnings="never",
)
```

The fitted core must resolve two threads, retain 1,000
`NonObliviousTree` instances, and contain no selection, refit, linear leaf,
linear residual, categorical cross, or calibration path.

Two independent pre-change fits under the frozen runtime produced identical
oracles. A candidate fit must reproduce all four exactly:

- canonical fitted-state SHA-256:
  `b588ddf2e09857479421bd490d394f7667d29e998b5d931dcff089455672604e`;
- serialized wrapper `.npz` SHA-256:
  `eb16c2f24f884f9661debd029897e3e7b1403d9e4189d86ff4f2c7ac4aeaf5bc`;
- serialized archive length: `606056` bytes; and
- full-training prediction SHA-256:
  `1544903cc4f52b361dc21327f43f848e9471337c88e44b5d2578f11b9bf515d1`.

The formal runner must copy this exact oracle function and bind its
implementation by source hash:

```python
def canonical_fitted_state_sha256(model):
    array_fields = (
        "features", "thresholds", "left_child", "right_child",
        "leaf_index", "values", "splits_feat", "splits_thr", "gains",
    )

    def add_array(digest, name, value):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))

    core = model.model_
    digest = hashlib.sha256()
    metadata = {
        "tree_count": len(core.trees_),
        "tree_types": [type(tree).__name__ for tree in core.trees_],
        "best_iteration": int(core.best_iteration_),
        "learning_rate": float(core.lr_),
        "stop_reason": str(core.stop_reason_),
        "tree_mode": str(core.tree_mode_),
        "ordered_boosting": bool(core.ordered_boosting_),
        "threads": int(core.n_threads_),
        "depth": int(core.depth),
        "l2_leaf_reg": float(core.l2_leaf_reg),
        "max_bins": int(core.max_bins),
    }
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    add_array(digest, "feature_importances", core.feature_importances_)
    for index, tree in enumerate(core.trees_):
        digest.update(
            f"tree:{index}:{tree.depth}:{tree.n_leaves}:"
            f"{tree.n_splits}".encode("utf-8")
        )
        for field in array_fields:
            add_array(digest, field, getattr(tree, field))
    return digest.hexdigest()
```

Exploration used creator fold 0. Formal confirmation uses:

- 127 unmodified creator fold-1 rows;
- all 524 unmodified creator fold-1 rows;
- all 585 genuinely cold-player rows;
- all 2,409 overlap-exposed held-team rows;
- repeated fold-1 rows at 8,192 and 32,768 rows; and
- fallback controls at 65,536 and 100,000 repeated rows.

Repeating rows is throughput evidence only. It does not create quality
evidence.

Record and verify the boundary table independently using prefixes of the same
fitted forest:

| Trees | Rows | Expected route |
| ---: | ---: | --- |
| 1 | 32,768 | per-tree fallback |
| 5 | 8,192 | packed |
| 16 | 2,409 | packed |
| 25 | 525 | per-tree fallback |
| 62 | 525 | per-tree fallback |
| 63 | 525 | packed |
| 258 | 127 | per-tree fallback |
| 259 | 127 | packed |
| 1,000 | 32,768 | packed |
| 1,000 | 65,536 | per-tree fallback at two threads |

Every expected route must be observed through instrumentation outside timed
sections; never infer dispatch from the desired formula.

## Behavior proof

For every real and repeated view:

- public candidate prediction must be array-exact to an independently
  executed per-tree reference loop;
- packed-core output must be array-exact to the same loop;
- row order, point ordering, ties, feature importance, fitted metadata,
  selected/final tree counts, learning rate, stop reason, and every tree array
  must remain unchanged;
- the final staged prediction must be array-exact to public prediction;
- the serialized `.npz` bytes before and after prediction must be identical;
- candidate fitted-state, archive bytes, archive length, and full-training
  prediction must equal the pinned pre-change oracles above;
- lazy packed cache identity and total packed array bytes must remain
  unchanged by repeated prediction;
- outputs, timing, memory, fitted state, and artifact values must be finite.

Expanded tests must require exact public-versus-loop parity for numeric,
categorical, and weighted RMSE; MAE; Quantile; binary classification; a
serialized-and-loaded leafwise model; one-, four-, and eighteen-thread
fallbacks; router boundary coordinates; fitted LightGBM versus hybrid mode;
and direct scalar packed kernels. The four- and eighteen-thread tests must
instrument the outer selector and prove that the scalar-only kernel is not
entered. Existing multiclass and all five distributional goldens must remain
exact. The strict command is:

```bash
DARKOFIT_STRICT_GOLDENS=1 PYTHONPATH=. \
  .cache/basketball-py312/bin/python -m pytest -q
```

Before that command, verify that the interpreter is exactly
`.cache/basketball-py312/bin/python`, reports Python `3.12.13`, and imports
the frozen runtime and strict-test package versions above. The general Python
3.13 development environment may be an additional gate, but it cannot
replace this frozen Python 3.12 strict gate.

Isomorphism proof:

- ordering preserved: yes, every row accumulates trees in original tree
  order;
- tie-breaking unchanged: yes, prediction bytes are unchanged;
- floating point: identical, with exact-array and strict-golden gates;
- RNG seeds: unchanged; prediction introduces no RNG;
- rollback: one source commit can be reverted without a model migration.

## Frozen timing and memory gates

Warm all public, reference-loop, packed-core, and fallback paths before
measurement. Core timing excludes validation and binning. Public timing
includes both equally. Use exactly 11 reciprocal blocks. Even-numbered blocks
run candidate then reference; odd-numbered blocks run reference then
candidate. Each measured arm in a block uses this frozen deterministic inner
call count:

```text
inner_calls = max(1, min(64, ceil(65536 / row_count)))
```

That yields 64 calls for 127, 524, and 585 rows; 28 for 2,409; 8 for
8,192; 2 for 32,768; and 1 for 65,536 and 100,000. Use the same count for
both arms and for core and public timing at a given row count. Record every
raw block duration, medians, IQR/median, minimum, maximum, and p50/p95/p99.

At two resolved threads:

- packed-core candidate/reference median ratio must be at most `0.80` on
  127, fold-1, cold-player, held-team, and 8,192-row cases;
- the 32,768-row packed-core ratio must be at most `0.98`;
- public candidate/reference median ratio must be at most `0.90` on fold-1,
  cold-player, and held-team cases;
- 65,536- and 100,000-row public calls must observe the per-tree fallback and
  candidate/reference ratios must be at most `1.10`;
- every gated timing series must have IQR/median at most `0.20`;
- candidate packed storage must equal the pre-change packed representation
  byte-for-byte and add zero persistent fitted-model bytes;
- maximum candidate transient traced memory may not exceed the matched
  reference by more than 256 KiB.

Fit time is directional and must not be claimed as changed. This is a
prediction-only campaign.

## Stop and advance rules

Any failed source, route, exactness, behavior, timing, stability, storage, or
memory gate rejects the candidate. Do not retune the work floor, tree floor,
block size, row ceiling, repetitions, or thresholds on fold 1. Restore the
pre-change router and preserve the artifact as a rejected experiment.

A complete pass authorizes the bounded internal scalar leafwise route because
its behavior proof is exact. It does not authorize a default model-policy
change, a categorical policy, a new public parameter, removal of fallback
kernels, or a universal A10 performance claim. The next evidence would be a
same-machine, report-only rerun of the existing spent 13-dataset A10 inference
panel; CTR23 and its lockbox remain sealed.
