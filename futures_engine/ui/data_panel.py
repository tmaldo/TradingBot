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
from datetime import UTC, datetime
from pathlib import Path

from futures_engine.core.types import BarInterval, DatasetMeta
from futures_engine.data.adapters.databento_adapter import DatabentoAdapter
from futures_engine.data.continuous import Adjustment, RollRule, build_continuous
from futures_engine.data.store import PENDING_SNAPSHOT_HASH, SnapshotStore
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


def fetch_databento_snapshot(
    store_root: str | Path,
    *,
    symbol_root: str,
    start: datetime,
    end: datetime,
    roll_rule: RollRule,
    adjustment: Adjustment,
    api_key: str,
    interval: BarInterval = "1m",
    dataset: str = "GLBX.MDP3",
) -> str:
    """Fetch real MES/MNQ history from Databento into a validation-grade snapshot.

    Thin orchestration (UI-G1): the EXISTING :class:`DatabentoAdapter` does the
    audited fetch + parse (parent ``.FUT`` symbology, fixed-point->float64 x 1e-9,
    ``ts_event``->bar-open, scale 1e-9), and :func:`build_continuous` does explicit
    continuous stitching (G3). We only wire them together, then persist the result
    via :class:`SnapshotStore`.

    The snapshot is written **only** after every per-contract fetch and the
    continuous build have succeeded -- a mid-fetch error propagates and leaves no
    partial snapshot on disk. Returns the content hash.

    Key hygiene (UI-G4): ``api_key`` comes from the caller (``os.environ`` at the
    route) and is used solely to construct the adapter. It is never logged and
    never written into :class:`DatasetMeta`/the manifest -- the meta records only
    ``source="databento"`` plus the content hash.
    """
    adapter = DatabentoAdapter(api_key=api_key, dataset=dataset)
    contracts = adapter.list_contracts(symbol_root, start.date(), end.date())
    if not contracts:
        raise ValueError(f"Databento returned no contracts for {symbol_root!r} in the given range")
    per_contract = {c.symbol: adapter.fetch_bars(c.symbol, start, end, interval) for c in contracts}

    bars, cmeta = build_continuous(per_contract, contracts, roll_rule, adjustment)

    meta = DatasetMeta(
        symbol_root=symbol_root,
        source="databento",
        interval=interval,
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        continuous=cmeta,
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime.now(UTC),
        validation_grade=True,
    )
    # Reached only on a clean, complete fetch + build -> no partial snapshot.
    return SnapshotStore(store_root).save(bars, meta)


def databento_enabled() -> bool:
    """Return whether a ``DATABENTO_API_KEY`` is present in the environment.

    Only the presence of the key is observed; its value is never read, stored,
    logged, or rendered. The live fetch path is U7.
    """
    return bool(os.environ.get("DATABENTO_API_KEY"))
