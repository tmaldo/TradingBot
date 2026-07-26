"""Tests for the immutable, content-addressed snapshot store (Global Constraint G4).

Highest-stakes behaviours (G16): the content hash must be a *pure* function of
bar content + essential meta (never ``as_of`` / mtime), identical data must hash
identically, any mutation must change the hash, ``load`` must return bit-identical
frames, and snapshots must be write-once.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from futures_engine.core.types import ContinuousMeta, DatasetMeta
from futures_engine.data.store import (
    PENDING_SNAPSHOT_HASH,
    DataIntegrityError,
    SnapshotExistsError,
    SnapshotStore,
    require_validation_grade,
)

_CONT = ContinuousMeta(
    roll_rule="volume",
    adjustment="panama_diff",
    roll_dates=[datetime(2024, 3, 7, tzinfo=UTC).date()],
    underlying_contracts=["MESH24", "MESM24"],
)


def _bars(closes: list[float]) -> pd.DataFrame:
    # freq=None mirrors real (irregular) vendor bars; it is not snapshot data.
    idx = pd.DatetimeIndex(
        pd.date_range("2024-01-02", periods=len(closes), freq="D", tz="UTC").tolist()
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0 + i for i in range(len(closes))],
        },
        index=idx,
    )


def _meta(*, snapshot_hash: str = PENDING_SNAPSHOT_HASH, **overrides: object) -> DatasetMeta:
    kwargs: dict[str, object] = {
        "symbol_root": "MES",
        "source": "databento",
        "interval": "1d",
        "start": datetime(2024, 1, 2, tzinfo=UTC),
        "end": datetime(2024, 1, 6, tzinfo=UTC),
        "continuous": _CONT,
        "snapshot_hash": snapshot_hash,
        "as_of": datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        "validation_grade": True,
    }
    kwargs.update(overrides)
    return DatasetMeta(**kwargs)


# --- content hash: determinism & sensitivity --------------------------------


def test_same_data_same_hash(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    h1 = store.content_hash(_bars([1.0, 2.0, 3.0]), _meta())
    h2 = store.content_hash(_bars([1.0, 2.0, 3.0]), _meta())
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_price_mutation_changes_hash(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    base = store.content_hash(_bars([1.0, 2.0, 3.0]), _meta())
    mutated = store.content_hash(_bars([1.0, 2.0, 3.000001]), _meta())
    assert base != mutated


def test_index_mutation_changes_hash(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bars = _bars([1.0, 2.0, 3.0])
    shifted = bars.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert store.content_hash(bars, _meta()) != store.content_hash(shifted, _meta())


def test_hash_ignores_as_of(tmp_path: Path) -> None:
    """The hash is stable across sessions: ``as_of`` must not influence it."""
    store = SnapshotStore(tmp_path)
    early = _meta(as_of=datetime(2020, 1, 1, tzinfo=UTC))
    late = _meta(as_of=datetime(2030, 1, 1, tzinfo=UTC))
    assert store.content_hash(_bars([1.0, 2.0]), early) == store.content_hash(
        _bars([1.0, 2.0]), late
    )


def test_hash_reflects_essential_meta(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    base = store.content_hash(_bars([1.0, 2.0]), _meta())
    other_source = store.content_hash(_bars([1.0, 2.0]), _meta(source="norgate"))
    assert base != other_source


# --- save / load round-trip & immutability ----------------------------------


def test_save_returns_hash_and_persists_files(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    h = store.save(_bars([1.0, 2.0, 3.0]), _meta())
    assert len(h) == 64
    assert (tmp_path / f"{h}.parquet").exists()
    assert (tmp_path / f"{h}.meta.json").exists()


def test_load_returns_bit_identical_frame(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bars = _bars([1.0, 2.0, 3.0, 4.0])
    h = store.save(bars, _meta())
    loaded, meta = store.load(h)
    pd.testing.assert_frame_equal(loaded, bars)
    assert meta.snapshot_hash == h
    assert meta.source == "databento"
    # the persisted meta carries the resolved hash, not the sentinel
    assert meta.snapshot_hash != PENDING_SNAPSHOT_HASH


def test_saved_meta_hash_matches_return(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bars = _bars([5.0, 6.0])
    h = store.save(bars, _meta())
    assert h == store.content_hash(bars, _meta())


def test_save_requires_pending_sentinel(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(DataIntegrityError, match="PENDING"):
        store.save(_bars([1.0, 2.0]), _meta(snapshot_hash="deadbeef"))


def test_snapshot_is_write_once(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bars = _bars([1.0, 2.0, 3.0])
    store.save(bars, _meta())
    with pytest.raises(SnapshotExistsError):
        store.save(bars, _meta())


def test_load_unknown_hash_raises(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("0" * 64)


def test_rejects_missing_bar_columns(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bad = _bars([1.0, 2.0]).drop(columns=["volume"])
    with pytest.raises(DataIntegrityError, match="column"):
        store.save(bad, _meta())


def test_rejects_non_utc_index(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bars = _bars([1.0, 2.0])
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="UTC"):
        store.save(bars, _meta())


# --- require_validation_grade (G1 / G3) --------------------------------------


def test_require_validation_grade_passes_for_graded_continuous() -> None:
    require_validation_grade(_meta())  # validation_grade=True, continuous set


def test_require_validation_grade_rejects_dev_grade() -> None:
    with pytest.raises(DataIntegrityError, match="validation"):
        require_validation_grade(_meta(validation_grade=False))


def test_require_validation_grade_rejects_missing_continuous() -> None:
    with pytest.raises(DataIntegrityError, match="continuous"):
        require_validation_grade(_meta(continuous=None))
