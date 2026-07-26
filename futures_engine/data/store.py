"""Immutable, content-addressed point-in-time snapshot store (Global Constraint G4).

A snapshot is a frozen ``(bars, DatasetMeta)`` pair persisted as
``<hash>.parquet`` + ``<hash>.meta.json`` under a configurable root
(default ``data/snapshots`` beneath the repo, which is gitignored). Snapshots are
**write-once**: an attempted overwrite raises, so a run's inputs can never mutate
underneath it.

Canonical content-hash recipe (``snapshot_hash``)
-------------------------------------------------
The hash is a **pure function of bar content plus essential provenance** and is
deliberately independent of ``as_of``, file mtime, row/column memory layout, and
host byte order, so it is stable across sessions and platforms::

    payload = b"FE-SNAPSHOT-v1\\n"
        + b"index\\0"  + int64 little-endian UTC-nanoseconds of each row (row order)
        + b"\\ncols\\0" + for each recognised column present, in a fixed order:
                            name(utf-8) + b"\\0" + float64 little-endian values (row order) + b"\\0"
        + b"\\nmeta\\0" + canonical JSON (sorted keys, no whitespace) of essential meta:
                            symbol_root, source, interval, start, end,
                            validation_grade, continuous
    snapshot_hash = sha256(payload).hexdigest()          # 64 lowercase hex chars

Little-endian fixed-width dtypes make the digest identical on any platform; NaN
uses numpy's canonical quiet-NaN bit pattern, so missing values hash stably.
``start``/``end`` are serialised as ISO-8601 UTC strings. ``as_of`` and
``snapshot_hash`` are intentionally excluded.

Chicken-and-egg with ``DatasetMeta.snapshot_hash`` (which is a required,
non-empty field): callers pass meta with ``snapshot_hash == PENDING_SNAPSHOT_HASH``
to :meth:`SnapshotStore.save`; the store computes the real hash, persists the
completed meta, and returns the hash. Passing any other placeholder is a loud
error, so a stale/incorrect hash can never be trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from futures_engine.core.types import BAR_COLUMNS, Bars, DatasetMeta

# Sentinel value for ``DatasetMeta.snapshot_hash`` handed to ``save`` before the
# real content hash is known. Non-empty so it satisfies ``DatasetMeta`` validation.
PENDING_SNAPSHOT_HASH = "PENDING"

# Optional per-frame column the store recognises alongside ``BAR_COLUMNS``.
_OPEN_INTEREST = "open_interest"
_ALLOWED_COLUMNS: frozenset[str] = frozenset(BAR_COLUMNS) | {_OPEN_INTEREST}

_HASH_VERSION_TAG = b"FE-SNAPSHOT-v1\n"
_HEX_HASH_LEN = 64

# Meta fields that participate in the content hash (provenance that defines the
# dataset's identity). ``as_of`` / ``snapshot_hash`` are excluded on purpose.
_ESSENTIAL_META_FIELDS: tuple[str, ...] = (
    "symbol_root",
    "source",
    "interval",
    "start",
    "end",
    "validation_grade",
    "continuous",
)


class DataIntegrityError(Exception):
    """Raised when data violates a point-in-time / provenance invariant (G1/G3/G4)."""


class SnapshotExistsError(DataIntegrityError):
    """Raised on an attempt to overwrite an existing (write-once) snapshot."""


def require_validation_grade(meta: DatasetMeta) -> None:
    """Assert ``meta`` is safe for a research/backtest/validation path.

    Raises :class:`DataIntegrityError` when the data is dev-grade (``G1``: the
    yfinance dev fetcher is never validation-grade) or when futures data lacks
    :class:`~futures_engine.core.types.ContinuousMeta` (``G3``: continuous
    stitching must be explicit).
    """
    if not meta.validation_grade:
        raise DataIntegrityError(
            f"dataset {meta.symbol_root!r} from {meta.source!r} is NOT validation-grade "
            "(dev-only data is refused in research/backtest/validation paths, G1)"
        )
    if meta.continuous is None:
        raise DataIntegrityError(
            f"futures dataset {meta.symbol_root!r} has no ContinuousMeta; continuous "
            "stitching must be explicit before use (G3)"
        )


def _validate_frame(bars: Bars) -> list[str]:
    """Validate frame shape for hashing/storage; return recognised columns in canonical order."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataIntegrityError("bars must have a DatetimeIndex")
    if bars.index.tz is None or str(bars.index.tz) != "UTC":
        raise DataIntegrityError(f"bars index must be UTC, got tz={bars.index.tz!r}")
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise DataIntegrityError(f"bars missing required column(s): {missing}")
    unknown = [c for c in bars.columns if c not in _ALLOWED_COLUMNS]
    if unknown:
        raise DataIntegrityError(f"bars have unrecognised column(s): {unknown}")
    ordered = list(BAR_COLUMNS)
    if _OPEN_INTEREST in bars.columns:
        ordered.append(_OPEN_INTEREST)
    return ordered


def _canonical_meta_json(meta: DatasetMeta) -> bytes:
    dumped = meta.model_dump(mode="json")
    essential = {field: dumped[field] for field in _ESSENTIAL_META_FIELDS}
    return json.dumps(essential, sort_keys=True, separators=(",", ":")).encode("utf-8")


class SnapshotStore:
    """Content-addressed, write-once store of ``(bars, DatasetMeta)`` snapshots."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def content_hash(self, bars: Bars, meta: DatasetMeta) -> str:
        """Return the canonical content hash for ``(bars, meta)`` (see module docstring)."""
        columns = _validate_frame(bars)
        index = cast(pd.DatetimeIndex, bars.index)  # _validate_frame guarantees this
        hasher = hashlib.sha256()
        hasher.update(_HASH_VERSION_TAG)
        hasher.update(b"index\0")
        # Normalise to int64 nanoseconds since the UTC epoch at a *fixed* ns
        # resolution (pandas' default datetime resolution changed to us in 3.0, so
        # ``asi8``/``view`` are resolution-dependent and unsafe for a stable hash).
        # Little-endian byte order keeps the digest identical across platforms.
        nanos = index.to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False)
        hasher.update(np.ascontiguousarray(nanos).tobytes())
        hasher.update(b"\ncols\0")
        for name in columns:
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            values = bars[name].to_numpy(dtype="<f8", copy=False)
            hasher.update(np.ascontiguousarray(values).tobytes())
            hasher.update(b"\0")
        hasher.update(b"\nmeta\0")
        hasher.update(_canonical_meta_json(meta))
        return hasher.hexdigest()

    def _parquet_path(self, snapshot_hash: str) -> Path:
        return self._root / f"{snapshot_hash}.parquet"

    def _meta_path(self, snapshot_hash: str) -> Path:
        return self._root / f"{snapshot_hash}.meta.json"

    def save(self, bars: Bars, meta: DatasetMeta) -> str:
        """Persist ``(bars, meta)`` immutably and return its content hash.

        ``meta.snapshot_hash`` must be :data:`PENDING_SNAPSHOT_HASH`; the resolved
        hash is written into the persisted meta. Raises :class:`SnapshotExistsError`
        if a snapshot with the resolved hash already exists (write-once).
        """
        if meta.snapshot_hash != PENDING_SNAPSHOT_HASH:
            raise DataIntegrityError(
                "save() requires meta.snapshot_hash == PENDING_SNAPSHOT_HASH "
                f"({PENDING_SNAPSHOT_HASH!r}); the store fills in the resolved hash. "
                f"got {meta.snapshot_hash!r}"
            )
        snapshot_hash = self.content_hash(bars, meta)
        parquet_path = self._parquet_path(snapshot_hash)
        meta_path = self._meta_path(snapshot_hash)
        if parquet_path.exists() or meta_path.exists():
            raise SnapshotExistsError(
                f"snapshot {snapshot_hash} already exists; snapshots are write-once"
            )
        resolved = meta.model_copy(update={"snapshot_hash": snapshot_hash})
        bars.to_parquet(parquet_path)
        meta_path.write_text(resolved.model_dump_json(), encoding="utf-8")
        return snapshot_hash

    def load(self, snapshot_hash: str) -> tuple[Bars, DatasetMeta]:
        """Return the ``(bars, meta)`` for ``snapshot_hash`` (bars bit-identical to save)."""
        parquet_path = self._parquet_path(snapshot_hash)
        meta_path = self._meta_path(snapshot_hash)
        if not parquet_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no snapshot {snapshot_hash} under {self._root}")
        bars = pd.read_parquet(parquet_path)
        meta = DatasetMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return bars, meta
