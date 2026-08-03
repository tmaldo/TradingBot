"""Thin orchestration for the cockpit Data screen (UI-G1).

This module holds no data logic of its own: it enumerates snapshots via
:meth:`SnapshotStore.list` and generates synthetic fixtures via the shared
:mod:`futures_engine.data.synthetic` helper. The FastAPI routes in
:mod:`futures_engine.ui.app` call these functions and render ``data.html``.

The "fetch real (Databento)" path is intentionally NOT implemented here -- it
lands in U7. This module only reports whether it is enabled (a ``DATABENTO_API_KEY``
is present); the key value is never read into a variable, logged, or rendered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from futures_engine.core.types import BarInterval
from futures_engine.data.store import SnapshotStore
from futures_engine.data.synthetic import generate_synthetic_snapshot


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """View model for one stored snapshot, populated from its ``DatasetMeta``."""

    hash: str
    symbol_root: str
    interval: str
    start: datetime
    end: datetime
    validation_grade: bool
    has_continuous_meta: bool


def list_snapshots(store_root: str | Path) -> list[SnapshotSummary]:
    """Return a :class:`SnapshotSummary` for every snapshot under ``store_root``.

    An empty (or freshly created) store yields an empty list -- never an error.
    """
    store = SnapshotStore(store_root)
    return [
        SnapshotSummary(
            hash=meta.snapshot_hash,
            symbol_root=meta.symbol_root,
            interval=meta.interval,
            start=meta.start,
            end=meta.end,
            validation_grade=meta.validation_grade,
            has_continuous_meta=meta.continuous is not None,
        )
        for meta in store.list()
    ]


def generate_synthetic(
    store_root: str | Path,
    *,
    symbol_root: str = "MES",
    n_bars: int = 2500,
    seed: int = 7,
    interval: BarInterval = "1m",
) -> str:
    """Write a validation-grade synthetic snapshot into ``store_root``; return its hash.

    Delegates to :func:`futures_engine.data.synthetic.generate_synthetic_snapshot`
    so the generator is not duplicated (UI-G1). The result carries
    :class:`ContinuousMeta` and passes ``require_validation_grade``.
    """
    store = SnapshotStore(store_root)
    return generate_synthetic_snapshot(
        store,
        symbol_root=symbol_root,
        n_bars=n_bars,
        seed=seed,
        interval=interval,
    )


def databento_enabled() -> bool:
    """Return whether a ``DATABENTO_API_KEY`` is present in the environment.

    Only the presence of the key is observed; its value is never read, stored,
    logged, or rendered. The live fetch path is U7.
    """
    return bool(os.environ.get("DATABENTO_API_KEY"))
