"""Ordered target statistics for categorical features.

This is CatBoost's key trick for categoricals. Instead of plain mean-target
encoding (which leaks the label of each row into its own feature and overfits),
we fix a random permutation of the rows and encode each row using only the rows
that come *before* it in that permutation:

    ctr_i = (sum_y_before(category_i) + prior * a) / (count_before(category_i) + a)

where `prior` is the global target mean and `a` is a smoothing weight. A row
never sees its own target, which removes the leakage / prediction shift that
makes naive target encoding so fragile.

At prediction time there is no "before", so we use the full training totals:

    ctr   = (sum_y_total(category) + prior * a) / (count_total(category) + a)

Unseen categories fall back to the prior.
"""

import numpy as np
import sys
from numba import njit


def _get_missing_category():
    return _MISSING_CATEGORY


class _MissingCategory:
    def __repr__(self):
        return "<ChimeraBoost missing category>"

    def __reduce__(self):
        return (_get_missing_category, ())


_MISSING_CATEGORY = _MissingCategory()


def _is_missing_value(value):
    if value is None:
        return True
    pd = sys.modules.get("pandas")
    if pd is not None:
        try:
            result = pd.isna(value)
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
        except (TypeError, ValueError):
            pass
    try:
        result = np.isnat(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    try:
        result = np.isnan(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    return False


def _factorize_with_loaded_pandas(col):
    """Use pandas' fast hashtable factorizer only if pandas is already loaded."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return None
    try:
        s = pd.Series(col, dtype=object)
        if s.hasnans:
            s = s.where(pd.notna(s), _MISSING_CATEGORY)
        codes, categories = pd.factorize(s, sort=False)
    except Exception:
        return None
    return codes.astype(np.int64), np.asarray(categories, dtype=object)


@njit(cache=True)
def _ordered_ts(codes, y, perm, n_cat, prior, a):
    """Single-permutation ordered target statistic.

    Returns the encoded column plus the full per-category totals (reused at
    predict time).
    """
    sums = np.zeros(n_cat)
    counts = np.zeros(n_cat)
    out = np.empty(codes.shape[0], dtype=np.float64)
    for pos in range(perm.shape[0]):
        i = perm[pos]
        c = codes[i]
        out[i] = (sums[c] + prior * a) / (counts[c] + a)
        sums[c] += y[i]
        counts[c] += 1.0
    return out, sums, counts


@njit(cache=True)
def _ordered_ts_weighted(codes, y, weight, perm, n_cat, prior, a):
    """Weighted ordered target statistic.

    Category totals accumulate weighted targets and weighted sample mass. Rows
    with zero weight receive an encoding from prior history but do not influence
    later rows or prediction-time totals.
    """
    sums = np.zeros(n_cat)
    counts = np.zeros(n_cat)
    out = np.empty(codes.shape[0], dtype=np.float64)
    for pos in range(perm.shape[0]):
        i = perm[pos]
        c = codes[i]
        out[i] = (sums[c] + prior * a) / (counts[c] + a)
        wi = weight[i]
        if wi > 0.0:
            sums[c] += wi * y[i]
            counts[c] += wi
    return out, sums, counts


@njit(cache=True)
def _ordered_ts_multi(codes, y, perms, n_cat, prior, a):
    """Average ordered statistics over multiple independent permutations."""
    n_samples = codes.shape[0]
    out = np.zeros(n_samples, dtype=np.float64)
    sums_total = np.zeros(n_cat)
    counts_total = np.zeros(n_cat)
    for p in range(perms.shape[0]):
        sums = np.zeros(n_cat)
        counts = np.zeros(n_cat)
        perm = perms[p]
        for pos in range(perm.shape[0]):
            i = perm[pos]
            c = codes[i]
            out[i] += (sums[c] + prior * a) / (counts[c] + a)
            sums[c] += y[i]
            counts[c] += 1.0
        if p == 0:
            for c in range(n_cat):
                sums_total[c] = sums[c]
                counts_total[c] = counts[c]
    out /= perms.shape[0]
    return out, sums_total, counts_total


@njit(cache=True)
def _ordered_ts_weighted_multi(codes, y, weight, perms, n_cat, prior, a):
    """Average weighted ordered statistics over independent permutations."""
    n_samples = codes.shape[0]
    out = np.zeros(n_samples, dtype=np.float64)
    sums_total = np.zeros(n_cat)
    counts_total = np.zeros(n_cat)
    for p in range(perms.shape[0]):
        sums = np.zeros(n_cat)
        counts = np.zeros(n_cat)
        perm = perms[p]
        for pos in range(perm.shape[0]):
            i = perm[pos]
            c = codes[i]
            out[i] += (sums[c] + prior * a) / (counts[c] + a)
            wi = weight[i]
            if wi > 0.0:
                sums[c] += wi * y[i]
                counts[c] += wi
        if p == 0:
            for c in range(n_cat):
                sums_total[c] = sums[c]
                counts_total[c] = counts[c]
    out /= perms.shape[0]
    return out, sums_total, counts_total


def _kfold_ts(codes, y, weight, folds, n_cat, prior, a):
    sums = np.zeros(n_cat)
    counts = np.zeros(n_cat)
    if weight is None:
        np.add.at(sums, codes, y)
        np.add.at(counts, codes, 1.0)
    else:
        np.add.at(sums, codes, weight * y)
        np.add.at(counts, codes, weight)

    out = np.empty(codes.shape[0], dtype=np.float64)
    for fold in folds:
        if fold.size == 0:
            continue
        fold_codes = codes[fold]
        fold_sums = np.zeros(n_cat)
        fold_counts = np.zeros(n_cat)
        if weight is None:
            np.add.at(fold_sums, fold_codes, y[fold])
            np.add.at(fold_counts, fold_codes, 1.0)
        else:
            np.add.at(fold_sums, fold_codes, weight[fold] * y[fold])
            np.add.at(fold_counts, fold_codes, weight[fold])
        out[fold] = (
            sums[fold_codes] - fold_sums[fold_codes] + prior * a
        ) / (counts[fold_codes] - fold_counts[fold_codes] + a)
    return out, sums, counts


class OrderedTargetEncoder:
    """Encodes one or more categorical columns into numeric ctr columns.

    Categorical inputs are expected as integer codes in [0, n_categories).
    Use `factorize` to turn arbitrary (string/object) columns into codes.
    """

    def __init__(self, smoothing=1.0, random_state=None, mode="ordered",
                 n_folds=20, ts_permutations=1):
        self.smoothing = float(smoothing)
        self.random_state = random_state
        self.mode = str(mode).lower().replace("-", "_")
        if self.mode not in {"ordered", "kfold"}:
            raise ValueError("mode must be 'ordered' or 'kfold'")
        self.n_folds = int(n_folds)
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.ts_permutations = int(ts_permutations)
        if self.ts_permutations < 1:
            raise ValueError("ts_permutations must be at least 1")
        self.prior_ = None
        self.sums_ = None       # list per column
        self.counts_ = None     # list per column
        self.n_cat_ = None      # list per column

    def fit_transform(self, codes_matrix, y, sample_weight=None):
        """codes_matrix: (n_samples, n_cat_features) int array of codes."""
        codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_cols = codes_matrix.shape
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n_samples)
        folds = np.array_split(perm, min(self.n_folds, max(n_samples, 1)))
        perms = None
        if self.mode == "ordered" and self.ts_permutations > 1:
            perms = np.empty((self.ts_permutations, n_samples), dtype=np.int64)
            perms[0] = perm
            for p in range(1, self.ts_permutations):
                perms[p] = rng.permutation(n_samples)

        if sample_weight is None:
            weight = None
            self.prior_ = float(np.mean(y))
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)
            self.prior_ = float(np.average(y, weights=weight))
        self.sums_, self.counts_, self.n_cat_ = [], [], []
        out = np.empty((n_samples, n_cols), dtype=np.float64)

        for j in range(n_cols):
            codes = np.ascontiguousarray(codes_matrix[:, j])
            n_cat = int(codes.max()) + 1 if codes.size else 1
            if self.mode == "kfold":
                enc, sums, counts = _kfold_ts(
                    codes, y, weight, folds, n_cat, self.prior_,
                    self.smoothing
                )
            elif weight is None:
                if self.ts_permutations == 1:
                    enc, sums, counts = _ordered_ts(
                        codes, y, perm, n_cat, self.prior_, self.smoothing
                    )
                else:
                    enc, sums, counts = _ordered_ts_multi(
                        codes, y, perms, n_cat, self.prior_, self.smoothing
                    )
            else:
                if self.ts_permutations == 1:
                    enc, sums, counts = _ordered_ts_weighted(
                        codes, y, weight, perm, n_cat, self.prior_,
                        self.smoothing
                    )
                else:
                    enc, sums, counts = _ordered_ts_weighted_multi(
                        codes, y, weight, perms, n_cat, self.prior_,
                        self.smoothing
                    )
            out[:, j] = enc
            self.sums_.append(sums)
            self.counts_.append(counts)
            self.n_cat_.append(n_cat)
        return out

    def transform(self, codes_matrix):
        codes_matrix = np.asarray(codes_matrix, dtype=np.int64)
        n_samples, n_cols = codes_matrix.shape
        out = np.empty((n_samples, n_cols), dtype=np.float64)
        a = self.smoothing
        for j in range(n_cols):
            codes = codes_matrix[:, j]
            sums, counts, n_cat = self.sums_[j], self.counts_[j], self.n_cat_[j]
            enc = np.full(n_samples, self.prior_, dtype=np.float64)
            valid = (codes >= 0) & (codes < n_cat)
            c = codes[valid]
            enc[valid] = (sums[c] + self.prior_ * a) / (counts[c] + a)
            out[:, j] = enc
        return out


def factorize(column):
    """Map an arbitrary 1D column to integer codes in [0, K).

    NaN / None map to a dedicated code. Returns (codes, categories).
    """
    col_raw = np.asarray(column)
    if col_raw.dtype != object:
        if np.issubdtype(col_raw.dtype, np.floating) and np.isnan(col_raw).any():
            col = col_raw.astype(object)
        else:
            try:
                categories, codes = np.unique(col_raw, return_inverse=True)
                return codes.astype(np.int64), categories.astype(object)
            except TypeError:
                col = np.asarray(column, dtype=object)
    else:
        col = np.asarray(column, dtype=object)

    pandas_result = _factorize_with_loaded_pandas(col)
    if pandas_result is not None:
        return pandas_result

    cats = {}
    codes = np.empty(col.shape[0], dtype=np.int64)
    for i, v in enumerate(col):
        # Normalize missing values to a single key.
        if _is_missing_value(v):
            v = _MISSING_CATEGORY
        if v not in cats:
            cats[v] = len(cats)
        codes[i] = cats[v]
    categories = np.empty(len(cats), dtype=object)
    for v, k in cats.items():
        categories[k] = v
    return codes, categories
