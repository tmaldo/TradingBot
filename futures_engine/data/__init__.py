"""Point-in-time data layer for the futures engine (task T1).

Public surface (imported by downstream research/backtest tasks):

- :class:`~futures_engine.data.provider.MarketDataProvider` /
  :class:`~futures_engine.data.provider.ContractInfo` -- the vendor-agnostic
  data access interface. All vendor SDK access lives behind adapters in
  :mod:`futures_engine.data.adapters` (Global Constraints G1/G2).
- :class:`~futures_engine.data.store.SnapshotStore` /
  :func:`~futures_engine.data.store.require_validation_grade` /
  :class:`~futures_engine.data.store.DataIntegrityError` -- immutable,
  content-addressed point-in-time snapshots (G4).
- :func:`~futures_engine.data.continuous.build_continuous` -- explicit
  continuous-contract stitching (G3).
- :mod:`futures_engine.data.audit` -- the look-ahead (point-in-time) audit
  registry and shift-test runner (G4).
"""

from __future__ import annotations

from futures_engine.data.continuous import build_continuous
from futures_engine.data.provider import ContractInfo, MarketDataProvider
from futures_engine.data.store import (
    PENDING_SNAPSHOT_HASH,
    DataIntegrityError,
    SnapshotExistsError,
    SnapshotStore,
    require_validation_grade,
)

__all__ = [
    "PENDING_SNAPSHOT_HASH",
    "ContractInfo",
    "DataIntegrityError",
    "MarketDataProvider",
    "SnapshotExistsError",
    "SnapshotStore",
    "build_continuous",
    "require_validation_grade",
]
