"""Regenerate the bundled validation-grade synthetic MES snapshot for the demo.

Deterministic (seeded): a GBM 1-minute MES path with no real drift, stored as a
content-addressed, validation-grade snapshot WITH ``ContinuousMeta`` under
``examples/mes_momentum_demo/snapshots/``. It is NOT point-in-time market data --
it is a plausible-but-not-winning fixture on which the honest demo yields NO-GO.

Run:  python examples/mes_momentum_demo/generate_snapshot.py
Prints the resulting snapshot hash to paste into ``config.yaml``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from futures_engine.core.types import ContinuousMeta, DatasetMeta
from futures_engine.data.store import PENDING_SNAPSHOT_HASH, SnapshotStore

N_BARS = 2500
SEED = 7


def synthetic_mes_1min(n_bars: int, *, seed: int) -> pd.DataFrame:
    """Deterministic GBM 1-minute MES bars (mirrors the T5/T6 test generator)."""
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


def build() -> str:
    bars = synthetic_mes_1min(N_BARS, seed=SEED)
    meta = DatasetMeta(
        symbol_root="MES",
        source="synthetic-demo",
        interval="1m",
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        continuous=ContinuousMeta(
            roll_rule="volume",
            adjustment="panama_diff",
            roll_dates=[date(2020, 1, 2)],
            underlying_contracts=["MESH20", "MESM20"],
        ),
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime(2026, 7, 25, tzinfo=UTC),
        validation_grade=True,
    )
    root = Path(__file__).parent / "snapshots"
    store = SnapshotStore(root)
    snapshot_hash = store.content_hash(bars, meta)
    if not (root / f"{snapshot_hash}.parquet").exists():
        store.save(bars, meta)
    return snapshot_hash


if __name__ == "__main__":
    print(build())
