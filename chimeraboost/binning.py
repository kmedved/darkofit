"""Quantization of numeric features into integer bins.

Borders are learned once on the training data (quantile based). Every feature
is mapped to a small integer bin index, which is what the tree builder consumes.
NaNs are routed to a dedicated bin so a split can isolate missing values, the
way CatBoost/LightGBM do.

Bin layout per feature:
    real values -> 0 .. n_borders        (via searchsorted on borders)
    NaN         -> n_borders + 1          (the highest bin, "missing")
The histogram width for a feature is therefore (n_borders + 2).

Border-finding sorts every column, which dominates preprocessing time on
large fits, so it runs on a row subsample once the data exceeds
``sample_count`` (the same idea as LightGBM's ``bin_construct_sample_cnt``).
The transform itself is a numba kernel parallelized over features.
"""

import numpy as np
from numba import njit, prange

BIN_DTYPE = np.uint16

DEFAULT_BIN_SAMPLE_COUNT = 200_000


def _bin_dtype_for_n_bins(n_bins):
    """Smallest unsigned dtype that can represent every learned bin id."""
    max_bin = int(np.max(n_bins)) - 1 if len(n_bins) else 0
    if max_bin <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_bin <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _feature_borders(col, max_bins):
    """Quantile borders for one numeric column, ignoring NaNs."""
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    uniq = np.unique(finite)
    if uniq.size <= max_bins:
        # Few distinct values: put a border between each pair.
        return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    borders = np.quantile(finite, qs)
    return np.unique(borders).astype(np.float64)


@njit(cache=True, parallel=True)
def _bin_columns_into(X, borders_flat, border_offsets, out, col_offset):
    """Bin one float block into integer bin ids, parallel over features.

    border_offsets holds absolute [start, end) positions into borders_flat
    for each of this block's columns. Matches
    ``np.searchsorted(borders, v, side="right")`` for finite values; NaN and
    +/-inf go to the dedicated top bin, exactly like the numpy path this
    replaces.
    """
    n = X.shape[0]
    n_cols = X.shape[1]
    for f in prange(n_cols):
        lo = border_offsets[f]
        hi = border_offsets[f + 1]
        nan_bin = (hi - lo) + 1
        for i in range(n):
            v = X[i, f]
            if np.isfinite(v):
                left = lo
                right = hi
                while left < right:
                    mid = (left + right) >> 1
                    if borders_flat[mid] <= v:
                        left = mid + 1
                    else:
                        right = mid
                out[i, col_offset + f] = left - lo
            else:
                out[i, col_offset + f] = nan_bin


class Binner:
    """Learns per-feature borders and maps a float matrix to bins.

    ``sample_count`` caps how many rows border-finding looks at; ``None``
    always uses every row. Sampling only affects the learned borders -- the
    transform always bins every row.
    """

    def __init__(self, max_bins=128, sample_count=DEFAULT_BIN_SAMPLE_COUNT,
                 random_state=None):
        self.max_bins = int(max_bins)
        self.sample_count = None if sample_count is None else int(sample_count)
        self.random_state = random_state
        self.borders_ = None       # list of np.ndarray, one per feature
        self.n_bins_ = None        # np.ndarray int, width per feature

    def fit(self, X):
        """Learn quantile borders for each column from training data."""
        return self.fit_blocks([np.asarray(X, dtype=np.float64)])

    def fit_blocks(self, blocks):
        """Learn borders over the columns of several blocks, in order.

        ``blocks`` is a list of (n_samples, width_b) float64 matrices that
        together form the conceptual combined feature matrix, without ever
        materializing their horizontal concatenation.
        """
        n_samples = blocks[0].shape[0] if blocks else 0
        if (
            self.sample_count is not None
            and n_samples > self.sample_count
        ):
            rng = np.random.default_rng(self.random_state)
            sample_idx = np.sort(
                rng.choice(n_samples, self.sample_count, replace=False)
            )
        else:
            sample_idx = None

        self.borders_ = []
        for block in blocks:
            for j in range(block.shape[1]):
                col = block[:, j] if sample_idx is None else block[sample_idx, j]
                self.borders_.append(_feature_borders(col, self.max_bins))
        # +1 for the searchsorted upper bucket, +1 for the NaN bucket.
        self.n_bins_ = np.array(
            [len(b) + 2 for b in self.borders_], dtype=np.int64
        )
        self._block_widths_ = [block.shape[1] for block in blocks]
        if self.borders_:
            self._borders_flat_ = np.concatenate(self.borders_)
        else:
            self._borders_flat_ = np.empty(0, dtype=np.float64)
        self._border_offsets_ = np.zeros(len(self.borders_) + 1, dtype=np.int64)
        np.cumsum(
            [len(b) for b in self.borders_], out=self._border_offsets_[1:]
        )
        return self

    def transform(self, X):
        """Map a float matrix to integer bin indices; NaNs go to the top bin."""
        return self.transform_blocks([np.asarray(X, dtype=np.float64)])

    def transform_blocks(self, blocks):
        """Bin several blocks straight into one output matrix."""
        widths = [block.shape[1] for block in blocks]
        if widths != self._block_widths_:
            raise ValueError("blocks do not match the fitted column layout")
        n_samples = blocks[0].shape[0] if blocks else 0
        if any(block.shape[0] != n_samples for block in blocks):
            raise ValueError("blocks must share the same number of rows")
        dtype = _bin_dtype_for_n_bins(self.n_bins_)
        out = np.empty((n_samples, sum(widths)), dtype=dtype)
        col_offset = 0
        for block in blocks:
            width = block.shape[1]
            if width:
                block = np.ascontiguousarray(block, dtype=np.float64)
                _bin_columns_into(
                    block,
                    self._borders_flat_,
                    self._border_offsets_[col_offset:col_offset + width + 1],
                    out,
                    col_offset,
                )
            col_offset += width
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit_transform_blocks(self, blocks):
        return self.fit_blocks(blocks).transform_blocks(blocks)
