"""Shared fixtures for the research-harness tests.

Provides:

* ``MES`` instrument spec and a validation-grade / dev-grade snapshot helper so
  sweeps can be exercised against the real :class:`SnapshotStore` refusal path.
* :func:`synthetic_mes_1min` -- a seeded GBM 1-minute MES bar generator with
  realistic intraday vol and overnight session gaps, used to build the
  >=500-combo runtime fixture without committing a multi-hundred-MB parquet.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from futures_engine.core.types import ContinuousMeta, DatasetMeta, InstrumentSpec
from futures_engine.data.store import PENDING_SNAPSHOT_HASH, SnapshotStore

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)


@pytest.fixture
def mes_spec() -> InstrumentSpec:
    return MES


@pytest.fixture
def store(tmp_path) -> SnapshotStore:  # type: ignore[no-untyped-def]
    return SnapshotStore(tmp_path / "snapshots")


def _meta(bars: pd.DataFrame, *, validation_grade: bool) -> DatasetMeta:
    continuous = (
        ContinuousMeta(
            roll_rule="volume",
            adjustment="panama_diff",
            roll_dates=[date(2024, 3, 15)],
            underlying_contracts=["MESH24", "MESM24"],
        )
        if validation_grade
        else None
    )
    return DatasetMeta(
        symbol_root="MES",
        source="synthetic-test" if validation_grade else "yfinance-dev",
        interval="1m",
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        continuous=continuous,
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime(2026, 7, 25, tzinfo=UTC),
        validation_grade=validation_grade,
    )


def save_snapshot(store: SnapshotStore, bars: pd.DataFrame, *, validation_grade: bool) -> str:
    """Persist ``bars`` with a validation- or dev-grade meta; return the hash."""
    return store.save(bars, _meta(bars, validation_grade=validation_grade))


def synthetic_mes_1min(n_bars: int, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic GBM 1-minute MES bars with intraday vol and session gaps.

    Not point-in-time market data -- a seeded synthetic path purely for harness
    throughput/behaviour tests. ~23h trading days: 1380 one-minute bars per day
    with a weekend/overnight gap between days.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 1380
    mu = 0.0
    sigma = 0.0009  # ~ per-minute log-vol
    log_ret = rng.normal(mu, sigma, size=n_bars)
    close = 5000.0 * np.exp(np.cumsum(log_ret))
    # Intraday range around the close.
    span = np.abs(rng.normal(0.0, 0.5, size=n_bars)) + 0.25
    open_ = np.empty(n_bars)
    open_[0] = 5000.0
    open_[1:] = close[:-1]  # open at prior close (gap handled by day stamping)
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(50, 5000, size=n_bars).astype(float)

    # Timestamps: contiguous minutes within a day, +64h gap (weekend-ish) per day.
    start = datetime(2020, 1, 2, 0, 0, tzinfo=UTC)
    idx: list[datetime] = []
    current = start
    for i in range(n_bars):
        idx.append(current)
        if (i + 1) % bars_per_day == 0:
            current = current + timedelta(hours=1)  # overnight gap
        else:
            current = current + timedelta(minutes=1)
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
