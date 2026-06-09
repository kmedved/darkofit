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

_MISSING_CATEGORY = "__nan__"


def _factorize_with_loaded_pandas(col):
    """Use pandas' fast hashtable factorizer only if pandas is already loaded."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return None
    try:
        s = pd.Series(col, dtype=object)
        if s.hasnans:
            s = s.where(pd.notna(s), _MISSING_CATEGORY)
        codes, categories = pd.factorize(s, sort=False, use_na_sentinel=False)
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


class OrderedTargetEncoder:
    """Encodes one or more categorical columns into numeric ctr columns.

    Categorical inputs are expected as integer codes in [0, n_categories).
    Use `factorize` to turn arbitrary (string/object) columns into codes.
    """

    def __init__(self, smoothing=1.0, random_state=None):
        self.smoothing = float(smoothing)
        self.random_state = random_state
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
            if weight is None:
                enc, sums, counts = _ordered_ts(
                    codes, y, perm, n_cat, self.prior_, self.smoothing
                )
            else:
                enc, sums, counts = _ordered_ts_weighted(
                    codes, y, weight, perm, n_cat, self.prior_, self.smoothing
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
        try:
            if np.issubdtype(col_raw.dtype, np.floating):
                missing = np.isnan(col_raw)
                if np.any(missing):
                    col_obj = col_raw.astype(object)
                    col_obj[missing] = _MISSING_CATEGORY
                    categories, codes = np.unique(col_obj, return_inverse=True)
                    return codes.astype(np.int64), categories.astype(object)
            categories, codes = np.unique(col_raw, return_inverse=True)
            return codes.astype(np.int64), categories.astype(object)
        except TypeError:
            pass

    col = np.asarray(column, dtype=object)
    pandas_result = _factorize_with_loaded_pandas(col)
    if pandas_result is not None:
        return pandas_result

    cats = {}
    codes = np.empty(col.shape[0], dtype=np.int64)
    for i, v in enumerate(col):
        # Normalize missing values to a single key.
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = _MISSING_CATEGORY
        if v not in cats:
            cats[v] = len(cats)
        codes[i] = cats[v]
    categories = np.empty(len(cats), dtype=object)
    for v, k in cats.items():
        categories[k] = v
    return codes, categories
