"""Fixtures for the T6 event-driven backtest tests.

Reuses the same seeded synthetic MES generator and snapshot-meta pattern the T5
research tests use (a validation-grade continuous snapshot with ``ContinuousMeta``,
plus the dev-grade and no-``ContinuousMeta`` variants for the G3 loud-fail path).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from futures_engine.core.types import ContinuousMeta, DatasetMeta, InstrumentSpec
from futures_engine.data.store import PENDING_SNAPSHOT_HASH, SnapshotStore
from futures_engine.trials.logger import TrialLogger

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)


def synthetic_mes_1min(n_bars: int, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic GBM 1-minute MES bars (mirrors tests/research/conftest.py)."""
    rng = np.random.default_rng(seed)
    bars_per_day = 1380
    log_ret = rng.normal(0.0, 0.0009, size=n_bars)
    close = 5000.0 * np.exp(np.cumsum(log_ret))
    span = np.abs(rng.normal(0.0, 0.5, size=n_bars)) + 0.25
    open_ = np.empty(n_bars)
    open_[0] = 5000.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(50, 5000, size=n_bars).astype(float)
    start = datetime(2020, 1, 2, 0, 0, tzinfo=UTC)
    idx: list[datetime] = []
    current = start
    for i in range(n_bars):
        idx.append(current)
        current += timedelta(hours=1) if (i + 1) % bars_per_day == 0 else timedelta(minutes=1)
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def trending_mes_1min(n_bars: int, *, slope: float = 0.5, seed: int = 3) -> pd.DataFrame:
    """A strongly up-trending fixture (for the delay=1 worse-or-equal check)."""
    rng = np.random.default_rng(seed)
    close = 5000.0 + slope * np.arange(n_bars) + rng.normal(0.0, 0.05, size=n_bars)
    open_ = np.empty(n_bars)
    open_[0] = 5000.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    volume = np.full(n_bars, 100.0)
    index = pd.DatetimeIndex(
        pd.date_range("2021-01-04", periods=n_bars, freq="1min", tz="UTC"), name="timestamp"
    )
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _meta(bars: pd.DataFrame, *, validation_grade: bool, continuous: bool) -> DatasetMeta:
    cont = (
        ContinuousMeta(
            roll_rule="volume",
            adjustment="panama_diff",
            roll_dates=[date(2020, 3, 15)],
            underlying_contracts=["MESH20", "MESM20"],
        )
        if continuous
        else None
    )
    return DatasetMeta(
        symbol_root="MES",
        source="synthetic-test" if validation_grade else "yfinance-dev",
        interval="1m",
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        continuous=cont,
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime(2026, 7, 25, tzinfo=UTC),
        validation_grade=validation_grade,
    )


def save_snapshot(
    store: SnapshotStore,
    bars: pd.DataFrame,
    *,
    validation_grade: bool = True,
    continuous: bool = True,
) -> str:
    return store.save(bars, _meta(bars, validation_grade=validation_grade, continuous=continuous))


@pytest.fixture
def mes_spec() -> InstrumentSpec:
    return MES


@pytest.fixture
def store(tmp_path) -> SnapshotStore:  # type: ignore[no-untyped-def]
    return SnapshotStore(tmp_path / "snapshots")


@pytest.fixture
def logger(tmp_path) -> TrialLogger:  # type: ignore[no-untyped-def]
    return TrialLogger(tmp_path / "trials.db")


@pytest.fixture
def bars() -> pd.DataFrame:
    return synthetic_mes_1min(2000, seed=11)


@pytest.fixture
def valid_snapshot(store: SnapshotStore, bars: pd.DataFrame) -> str:
    return save_snapshot(store, bars, validation_grade=True, continuous=True)
