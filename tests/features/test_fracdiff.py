"""Tests for fixed-width-window fractional differencing (AFML Chapter 5).

Verifies the two closed-form anchors (``d=0`` is identity, ``d=1`` is the first
difference) and demonstrates the memory-vs-stationarity trade-off with an
Augmented Dickey-Fuller test on an integrated (random-walk) fixture: raising
``d`` from 0 to 1 makes the series progressively more stationary (ADF statistic
falls) while shedding memory (correlation with the level falls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from futures_engine.features.fracdiff import frac_diff


def _random_walk(n: int = 2000, seed: int = 11, drift: float = 0.1) -> pd.Series:
    """An integrated (I(1)) series with drift -- robustly non-stationary."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC").tolist())
    return pd.Series(100.0 + np.cumsum(rng.normal(drift, 1.0, n)), index=idx)


def test_d_zero_is_identity() -> None:
    s = _random_walk(200)
    out = frac_diff(s, d=0.0, threshold=1e-5)
    pd.testing.assert_series_equal(out, s, check_names=False)


def test_d_one_is_first_difference() -> None:
    s = _random_walk(200)
    out = frac_diff(s, d=1.0, threshold=1e-5)
    pd.testing.assert_series_equal(out, s.diff(), check_names=False)


def test_output_aligned_to_input_index() -> None:
    s = _random_walk(120)
    out = frac_diff(s, d=0.4, threshold=1e-4)
    assert out.index.equals(s.index)
    assert isinstance(out, pd.Series)


def test_is_causal_backward_window() -> None:
    """A fixed-width backward window: truncating the tail cannot change past values."""
    s = _random_walk(300)
    full = frac_diff(s, d=0.4, threshold=1e-4)
    cut = 200
    truncated = frac_diff(s.iloc[:cut], d=0.4, threshold=1e-4)
    pd.testing.assert_series_equal(full.iloc[:cut].dropna(), truncated.dropna(), check_names=False)


def test_memory_vs_stationarity_tradeoff() -> None:
    s = _random_walk()
    level = s

    def adf_stat(d: float) -> float:
        fd = frac_diff(s, d=d, threshold=1e-5).dropna()
        return float(adfuller(fd.to_numpy(), maxlag=1, regression="c", autolag=None)[0])

    def memory(d: float) -> float:
        fd = frac_diff(s, d=d, threshold=1e-5)
        aligned = pd.concat([fd, level], axis=1).dropna()
        return float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])

    adf0, adf_half, adf1 = adf_stat(0.0), adf_stat(0.5), adf_stat(1.0)
    # More differencing -> more stationary -> more negative ADF statistic.
    assert adf0 > adf_half > adf1

    # Stationarity outcome at the 5% critical value (~-2.86): the raw level fails
    # to reject a unit root; the fully differenced series rejects it.
    crit_5pct = -2.86
    assert adf0 > crit_5pct
    assert adf1 < crit_5pct

    # Memory: lower d retains more correlation with the original level than d=1.
    assert memory(0.5) > memory(1.0)
