"""Shared fixtures for feature/indicator tests.

``bars_fixture`` is a deterministic, seeded OHLCV frame in the engine's ``Bars``
shape (UTC DatetimeIndex; lowercase ``open, high, low, close, volume`` columns),
so the ported indicators can be guarded against the legacy formulas on a stable
reference series with no network or wall-clock reads (G15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 600, seed: int = 7, drift: float = 0.0003) -> pd.DataFrame:
    """Synthetic geometric-random-walk OHLCV with regime-switching volatility."""
    rng = np.random.default_rng(seed)
    vol = np.where(rng.random(n) < 0.1, 0.03, 0.012)
    rets = rng.normal(drift, vol)
    close = 100.0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.0, 0.008, n))
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    open_ = low + rng.random(n) * (high - low)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC").tolist())
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def bars_fixture() -> pd.DataFrame:
    return make_bars()
