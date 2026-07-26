"""Fractional differencing with a fixed-width window (López de Prado, AFML §5).

Integer differencing (``diff``) makes a price series stationary but erases almost
all memory. *Fractional* differencing applies a real order ``d in [0, 1]`` so the
series can be made stationary while retaining as much memory (long-range
dependence) as possible -- the sweet spot for predictive features (G7 trend/
momentum orientation without discarding level information).

Fixed-width window (FFD)
------------------------
The differencing weights follow the recursion ``w_0 = 1`` and
``w_k = -w_{k-1} * (d - k + 1) / k``. They decay towards zero; FFD truncates the
window at the first ``k`` with ``|w_k| < threshold``, giving a **fixed-width,
backward-looking** kernel of length ``l``. The differenced value is

    y_t = sum_{k=0}^{l-1} w_k * x_{t-k}

so ``y_t`` depends only on bars at or before ``t`` (point-in-time safe, G4). The
first ``l-1`` observations lack a full window and are ``NaN``.

Anchors: ``d=0`` gives weights ``[1]`` (identity); ``d=1`` gives ``[1, -1]`` (the
first difference).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


def ffd_weights(d: float, threshold: float) -> FloatArray:
    """Fixed-width-window differencing weights for order ``d``.

    Returns ``[w_0, w_1, ...]`` (``w_0 == 1`` multiplies the current bar, ``w_k``
    multiplies lag ``k``), truncated at the first ``|w_k| < threshold``.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.asarray(weights, dtype=np.float64)


def frac_diff(series: pd.Series, d: float, threshold: float) -> pd.Series:
    """Fixed-width-window fractionally differenced ``series`` of order ``d``.

    ``threshold`` sets the window width (smaller keeps more, longer-memory lags).
    The result aligns to ``series.index``; the warm-up region without a full
    window is ``NaN``.
    """
    weights = ffd_weights(d, threshold)
    width = len(weights)
    x = series.to_numpy(dtype=np.float64)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    # w_0 multiplies x_t, w_k multiplies x_{t-k}: dot weights with the reversed window.
    reversed_weights = weights[::-1]
    for t in range(width - 1, n):
        out[t] = float(np.dot(reversed_weights, x[t - width + 1 : t + 1]))
    return pd.Series(out, index=series.index)
