"""Tests for average-uniqueness sample weights (AFML ch.4).

A label's weight is its average uniqueness: over the bars its ``[t0, t1]`` span
covers, the mean of ``1 / concurrency``. Disjoint labels never share a bar, so
every weight is 1; overlapping labels share bars, so weights fall below 1. The
weights are a float Series in ``(0, 1]`` aligned to the labels' index, usable
directly as a LightGBM/sklearn ``sample_weight``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futures_engine.labels.triple_barrier import uniqueness_weights


def _labels(spans: list[tuple[int, int]], grid: pd.DatetimeIndex) -> pd.DataFrame:
    t0 = [grid[s] for s, _ in spans]
    t1 = [grid[e] for _, e in spans]
    return pd.DataFrame({"t1": t1}, index=pd.DatetimeIndex(t0))


def _grid(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))


def test_disjoint_labels_have_unit_weight() -> None:
    grid = _grid(6)
    labels = _labels([(0, 1), (2, 3), (4, 5)], grid)
    w = uniqueness_weights(labels, grid)
    assert np.allclose(w.to_numpy(), 1.0)


def test_overlapping_labels_have_sub_unit_weight() -> None:
    grid = _grid(4)
    # spans [0,2] and [1,3]: concurrency = [1, 2, 2, 1].
    labels = _labels([(0, 2), (1, 3)], grid)
    w = uniqueness_weights(labels, grid)
    assert (w < 1.0).all()
    # each label: mean(1/1, 1/2, 1/2) = 2/3.
    assert np.allclose(w.to_numpy(), 2.0 / 3.0)


def test_weights_are_in_unit_interval_and_aligned() -> None:
    grid = _grid(10)
    labels = _labels([(0, 4), (2, 6), (5, 9)], grid)
    w = uniqueness_weights(labels, grid)
    assert w.index.equals(labels.index)
    assert ((w > 0.0) & (w <= 1.0)).all()
    assert w.dtype == np.float64
    assert not w.isna().any()


def test_fully_overlapping_pair_weight_is_half() -> None:
    grid = _grid(5)
    # identical spans [0,4]: concurrency 2 everywhere -> uniqueness 1/2.
    labels = _labels([(0, 4), (0, 4)], grid)
    w = uniqueness_weights(labels, grid)
    assert np.allclose(w.to_numpy(), 0.5)
