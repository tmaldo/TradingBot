"""Tests for the shared synthetic generator + snapshot writer (U3 additive helper)."""

from __future__ import annotations

import pandas as pd

from futures_engine.core.types import BAR_COLUMNS
from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.data.synthetic import (
    generate_synthetic_snapshot,
    synthetic_mes_1min,
)


def test_synthetic_mes_1min_shape_and_columns() -> None:
    bars = synthetic_mes_1min(50, seed=7)
    assert len(bars) == 50
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert str(bars.index.tz) == "UTC"


def test_synthetic_mes_1min_is_deterministic_under_seed() -> None:
    a = synthetic_mes_1min(100, seed=42)
    b = synthetic_mes_1min(100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_mes_1min_differs_across_seeds() -> None:
    a = synthetic_mes_1min(100, seed=1)
    b = synthetic_mes_1min(100, seed=2)
    assert not a["close"].equals(b["close"])


def test_generate_snapshot_is_validation_grade_with_continuous_meta(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot_hash = generate_synthetic_snapshot(store, n_bars=200, seed=7)
    bars, meta = store.load(snapshot_hash)
    assert len(bars) == 200
    assert meta.validation_grade is True
    assert meta.continuous is not None
    # Must be accepted by the audited pipeline's guard.
    require_validation_grade(meta)


def test_generate_snapshot_is_deterministic_under_seed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store_a = SnapshotStore(tmp_path / "a")
    store_b = SnapshotStore(tmp_path / "b")
    hash_a = generate_synthetic_snapshot(store_a, n_bars=150, seed=11)
    hash_b = generate_synthetic_snapshot(store_b, n_bars=150, seed=11)
    assert hash_a == hash_b
