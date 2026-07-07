"""Shared feature preprocessing for every ChimeraBoost estimator.

Turns a raw (possibly mixed numeric/categorical, possibly object-dtype) matrix
into integer bins ready for the tree builder, and remembers everything needed to
reproduce the same transform at predict time.

Categoricals are encoded with ordered target statistics. The encoder is fit
against a *list* of target vectors:
  * regression / binary -> one target (y, or the 0/1 label)
  * multiclass          -> K one-hot targets (one ordered-TS column per class)
This is why a single categorical column can expand into K numeric columns for
multiclass, exactly like CatBoost's per-class target statistics.

`feature_map_` maps each combined-matrix column back to its original input
column index, so importances can be aggregated in the user's feature space.
"""

import sys

import numpy as np

from .binning import Binner, DEFAULT_BIN_SAMPLE_COUNT
from .target_encoding import (
    OrderedTargetEncoder,
    factorize,
    _MISSING_CATEGORY,
    _is_missing_value,
)


def _normalize_target_ordered_cat_codes(value):
    if value is None or value is False:
        return "off"
    if value is True:
        raise ValueError(
            "target_ordered_cat_codes=True is ambiguous; use "
            "'leaky_full' to opt in to full-target raw-code ordering"
        )
    mode = str(value).lower().replace("-", "_")
    if mode in {"off", "none", "false"}:
        return "off"
    if mode == "leaky_full":
        return "leaky_full"
    raise ValueError("target_ordered_cat_codes must be 'off' or 'leaky_full'")


class FeaturePreprocessor:
    """Converts raw mixed-type input into integer bins for the tree builder.

    Numeric columns are quantile-binned; categorical columns are ordered-target
    encoded (one encoded column per target supplied to `fit_transform`) and then
    binned alongside the numerics. The fitted state needed to reproduce the
    transform at predict time is retained, along with `feature_map_` mapping each
    output column back to its original input column for importances.

    The numeric, raw-code, and target-stat blocks are binned directly into one
    output matrix; their float64 horizontal concatenation is never materialized.
    """

    def __init__(self, max_bins=254, cat_smoothing=1.0, random_state=None,
                 include_cat_codes=False, target_encoding_mode="ordered",
                 target_encoding_folds=20,
                 ts_permutations=1,
                 target_ordered_cat_codes="off",
                 bin_sample_count=DEFAULT_BIN_SAMPLE_COUNT):
        self.max_bins = int(max_bins)
        self.cat_smoothing = float(cat_smoothing)
        self.random_state = random_state
        self.include_cat_codes = bool(include_cat_codes)
        self.target_encoding_mode = target_encoding_mode
        self.target_encoding_folds = int(target_encoding_folds)
        self.ts_permutations = int(ts_permutations)
        if self.ts_permutations < 1:
            raise ValueError("ts_permutations must be at least 1")
        self.target_ordered_cat_codes = _normalize_target_ordered_cat_codes(
            target_ordered_cat_codes
        )
        self.bin_sample_count = bin_sample_count

    # ---- helpers -------------------------------------------------------------
    def _split_columns_fit(self, X, cat_features):
        """Split input into a numeric matrix and an integer-code matrix for the
        categorical columns, learning the category->code maps on the way."""
        n_features = X.shape[1]
        cat_set = set(cat_features or [])
        self.cat_features_ = sorted(cat_set)
        self.num_features_ = [f for f in range(n_features) if f not in cat_set]

        if not self.num_features_:
            num = np.empty((X.shape[0], 0))
        elif len(self.num_features_) == n_features:
            num = np.asarray(X, dtype=np.float64)
        else:
            num = np.asarray(X[:, self.num_features_], dtype=np.float64)

        self._cat_indexes_ = {}  # lazy pd.Index cache; reset on every fit
        if self.cat_features_:
            codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
            self.cat_maps_ = []
            self.cat_categories_ = []
            for j, f in enumerate(self.cat_features_):
                c, cats = factorize(X[:, f])
                codes[:, j] = c
                self.cat_maps_.append({v: i for i, v in enumerate(cats)})
                self.cat_categories_.append(cats)
        else:
            codes = np.empty((X.shape[0], 0), dtype=np.int64)
            self.cat_maps_ = []
            self.cat_categories_ = []
        return num, codes

    def _codes_for_transform(self, X):
        """Map categorical columns to the codes learned at fit time; unseen
        categories get -1 (the encoder then falls back to the prior)."""
        if not self.cat_features_:
            return np.empty((X.shape[0], 0), dtype=np.int64)
        codes = np.empty((X.shape[0], len(self.cat_features_)), dtype=np.int64)
        pd = sys.modules.get("pandas")
        for j, f in enumerate(self.cat_features_):
            col = X[:, f]
            if pd is not None:
                mapped = self._pandas_codes_for_column(pd, col, j)
                if mapped is not None:
                    codes[:, j] = mapped
                    continue
            m = self.cat_maps_[j]
            if _MISSING_CATEGORY in m:
                missing = _MISSING_CATEGORY
                codes[:, j] = np.fromiter(
                    (
                        m.get(
                            missing
                            if _is_missing_value(v)
                            else v,
                            -1,
                        )
                        for v in col
                    ),
                    dtype=np.int64,
                    count=X.shape[0],
                )
            else:
                codes[:, j] = np.fromiter(
                    (m.get(v, -1) for v in col),
                    dtype=np.int64,
                    count=X.shape[0],
                )
        return codes

    def _pandas_codes_for_column(self, pd, col, j):
        """Vectorized hashtable lookup of fitted category codes.

        ``Index.get_indexer`` returns -1 for unseen values, which is exactly
        the prior-fallback sentinel the dict path uses. Returns None on any
        failure so the caller can fall back to the per-element dict loop.
        """
        try:
            cat_map = self.cat_maps_[j]
            if (
                "__nan__" in cat_map
                and _MISSING_CATEGORY in cat_map
                and cat_map["__nan__"] == cat_map[_MISSING_CATEGORY]
            ):
                return None
            cache = getattr(self, "_cat_indexes_", None)
            if cache is None:
                cache = self._cat_indexes_ = {}
            index = cache.get(j)
            if index is None:
                index = pd.Index(self.cat_categories_[j])
                if not index.is_unique:
                    return None
                cache[j] = index
            s = pd.Series(col, dtype=object)
            if _MISSING_CATEGORY in cat_map:
                s = s.where(pd.notna(s), _MISSING_CATEGORY)
            return index.get_indexer(s)
        except Exception:
            return None

    def _fit_target_ordered_code_remaps(self, codes, target, sample_weight):
        target = np.asarray(target, dtype=np.float64)
        weight = (
            None if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        prior = (
            float(np.mean(target)) if weight is None
            else float(np.average(target, weights=weight))
        )
        remaps = []
        for j in range(codes.shape[1]):
            col = np.ascontiguousarray(codes[:, j])
            n_cat = int(col.max()) + 1 if col.size else 1
            sums = np.zeros(n_cat, dtype=np.float64)
            counts = np.zeros(n_cat, dtype=np.float64)
            if weight is None:
                np.add.at(sums, col, target)
                np.add.at(counts, col, 1.0)
            else:
                np.add.at(sums, col, weight * target)
                np.add.at(counts, col, weight)
            means = (
                sums + prior * self.cat_smoothing
            ) / (counts + self.cat_smoothing)
            order = np.lexsort((np.arange(n_cat, dtype=np.int64), means))
            remap = np.empty(n_cat, dtype=np.int64)
            remap[order] = np.arange(n_cat, dtype=np.int64)
            remaps.append(remap)
        return remaps

    def _raw_code_block(self, codes):
        if self.target_ordered_cat_codes != "leaky_full":
            raw_codes = codes.astype(np.float64)
            raw_codes[raw_codes < 0] = np.nan
            return raw_codes

        remaps = getattr(self, "cat_code_remaps_", [])
        out = np.full(codes.shape, np.nan, dtype=np.float64)
        for j, remap in enumerate(remaps):
            col = codes[:, j]
            valid = (col >= 0) & (col < len(remap))
            out[valid, j] = remap[col[valid]]
        return out

    # ---- fit / transform -----------------------------------------------------
    def fit_transform(self, X, encode_targets, cat_features, sample_weight=None):
        """encode_targets: list of 1D arrays used for ordered TS (len T)."""
        num, codes = self._split_columns_fit(X, cat_features)

        encoded_blocks = []
        code_blocks = []
        self.encoders_ = []
        self.cat_code_remaps_ = []
        if codes.shape[1]:
            if self.include_cat_codes:
                if self.target_ordered_cat_codes == "leaky_full":
                    if len(encode_targets) != 1:
                        raise ValueError(
                            "target_ordered_cat_codes='leaky_full' is "
                            "currently scalar-only"
                        )
                    self.cat_code_remaps_ = (
                        self._fit_target_ordered_code_remaps(
                            codes, encode_targets[0], sample_weight
                        )
                    )
                code_blocks.append(self._raw_code_block(codes))
            for t, target in enumerate(encode_targets):
                enc = OrderedTargetEncoder(
                    self.cat_smoothing,
                    None if self.random_state is None else self.random_state + t,
                    mode=self.target_encoding_mode,
                    n_folds=self.target_encoding_folds,
                    ts_permutations=self.ts_permutations,
                )
                encoded_blocks.append(
                    enc.fit_transform(codes, target, sample_weight=sample_weight)
                )
                self.encoders_.append(enc)

        self._build_feature_map(num.shape[1], codes.shape[1], len(encode_targets))

        self.binner_ = Binner(self.max_bins, sample_count=self.bin_sample_count,
                              random_state=self.random_state)
        X_binned = self.binner_.fit_transform_blocks(
            [num] + code_blocks + encoded_blocks,
            sample_weight=sample_weight,
        )
        self.n_bins_ = self.binner_.n_bins_
        return X_binned

    def transform(self, X):
        """Apply the fitted binning + categorical encoding to new data."""
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        expected = int(getattr(self, "n_input_features_", X.shape[1]))
        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features, but fitted preprocessor "
                f"expects {expected}"
            )
        if not self.num_features_:
            num = np.empty((X.shape[0], 0))
        elif not self.cat_features_ and len(self.num_features_) == X.shape[1]:
            num = np.asarray(X, dtype=np.float64)
        else:
            num = np.asarray(X[:, self.num_features_], dtype=np.float64)
        encoded_blocks = []
        code_blocks = []
        if self.cat_features_:
            codes = self._codes_for_transform(X)
            if self.include_cat_codes:
                code_blocks.append(self._raw_code_block(codes))
            for enc in self.encoders_:
                encoded_blocks.append(enc.transform(codes))
        return self.binner_.transform_blocks(
            [num] + code_blocks + encoded_blocks
        )

    # ---- internals -----------------------------------------------------------
    def _build_feature_map(self, n_num, n_cat, n_targets):
        """Combined column index -> original input column index."""
        fmap = list(self.num_features_)            # numeric block
        if self.include_cat_codes:
            fmap.extend(self.cat_features_)        # raw category-code block
        for _ in range(n_targets):                 # each TS target adds a block
            fmap.extend(self.cat_features_)        # one col per cat feature
        self.feature_map_ = np.array(fmap, dtype=np.int64)
        self.n_input_features_ = (
            (max(self.num_features_) if self.num_features_ else -1)
        )
        if self.cat_features_:
            self.n_input_features_ = max(self.n_input_features_,
                                         max(self.cat_features_))
        self.n_input_features_ += 1
