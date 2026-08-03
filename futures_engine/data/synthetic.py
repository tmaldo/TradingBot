"""Shared, seeded synthetic MES bar generator + snapshot writer.

This is the single home for the deterministic GBM 1-minute MES generator that
was previously copied into the demo script and the backtest/research conftests.
It produces a *validation-grade* snapshot **with** :class:`ContinuousMeta` so the
audited pipeline (which refuses dev-grade or continuous-less futures data, G1/G3)
accepts it. The data is NOT point-in-time market data -- it is a plausible,
reproducible fixture for offline research/UI flows.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from futures_engine.core.types import BarInterval, ContinuousMeta, DatasetMeta
from futures_engine.data.store import PENDING_SNAPSHOT_HASH, SnapshotStore

_BARS_PER_DAY = 1380  # ~23h trading session of 1-minute bars
_START = datetime(2020, 1, 2, 0, 0, tzinfo=UTC)


def synthetic_mes_1min(n_bars: int, *, seed: int) -> pd.DataFrame:
    """Return ``n_bars`` deterministic GBM 1-minute MES bars for ``seed``.

    Mirrors the T5/T6 test generator: a seeded geometric-Brownian-motion close
    with no real drift, intraday range around it, integer volume, and a small
    overnight gap between sessions. Identical inputs always yield identical bars.
    """
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.0009, size=n_bars)
    close = 5000.0 * np.exp(np.cumsum(log_ret))
    span = np.abs(rng.normal(0.0, 0.5, size=n_bars)) + 0.25
    open_ = np.empty(n_bars)
    open_[0] = 5000.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(50, 5000, size=n_bars).astype(float)
    idx: list[datetime] = []
    current = _START
    for i in range(n_bars):
        idx.append(current)
        current += timedelta(hours=1) if (i + 1) % _BARS_PER_DAY == 0 else timedelta(minutes=1)
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def generate_synthetic_snapshot(
    store: SnapshotStore,
    *,
    symbol_root: str = "MES",
    n_bars: int = 2500,
    seed: int = 7,
    interval: BarInterval = "1m",
) -> str:
    """Generate a seeded synthetic snapshot into ``store``; return its hash.

    The written snapshot is validation-grade and carries :class:`ContinuousMeta`,
    so :func:`~futures_engine.data.store.require_validation_grade` passes and the
    audited pipeline will load it. Deterministic: the same
    ``(symbol_root, n_bars, seed, interval)`` yields the same content hash.
    """
    bars = synthetic_mes_1min(n_bars, seed=seed)
    meta = DatasetMeta(
        symbol_root=symbol_root,
        source="synthetic",
        interval=interval,
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        continuous=ContinuousMeta(
            roll_rule="volume",
            adjustment="panama_diff",
            roll_dates=[date(2020, 1, 2)],
            underlying_contracts=[f"{symbol_root}H20", f"{symbol_root}M20"],
        ),
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime.now(UTC),
        validation_grade=True,
    )
    return store.save(bars, meta)
