"""Append-only trial store backed by SQLite.

Every backtest/validation trial is logged here so the honest trial count feeds
the Deflated Sharpe Ratio (Global Constraint G10). The store is *append-only*:
there is deliberately no update or delete API, and ``trial_id`` is unique. Each
record must carry full provenance (config hash, data snapshot hashes, seed, git
SHA); missing fields are rejected by ``TrialRecord`` validation.

Concurrency: a fresh connection per operation plus WAL mode and a busy timeout
make writes safe across threads and processes; an in-process lock serialises
writers to avoid ``database is locked`` under thread contention.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id              TEXT NOT NULL UNIQUE,
    run_id                TEXT NOT NULL,
    ts                    TEXT NOT NULL,
    strategy_family       TEXT NOT NULL,
    config_hash           TEXT NOT NULL,
    params                TEXT NOT NULL,
    data_snapshot_hashes  TEXT NOT NULL,
    cv_scheme             TEXT NOT NULL,
    metrics               TEXT NOT NULL,
    seed                  INTEGER NOT NULL,
    git_sha               TEXT NOT NULL
)
"""

_INSERT = """
INSERT INTO trials (
    trial_id, run_id, ts, strategy_family, config_hash, params,
    data_snapshot_hashes, cv_scheme, metrics, seed, git_sha
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_BUSY_TIMEOUT_MS = 30_000


class TrialRecord(BaseModel):
    """One logged trial. All fields are mandatory (no silent-defaulted provenance)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    ts: datetime
    strategy_family: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    params: dict[str, Any]
    data_snapshot_hashes: list[str]
    cv_scheme: str = Field(min_length=1)
    metrics: dict[str, float]
    seed: int
    git_sha: str = Field(min_length=1)


class TrialLogger:
    """Append-only SQLite-backed store of :class:`TrialRecord`."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._write_lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    def log(self, record: TrialRecord) -> None:
        """Append ``record``. Raises on a duplicate ``trial_id`` (append-only)."""
        row = (
            record.trial_id,
            record.run_id,
            record.ts.isoformat(),
            record.strategy_family,
            record.config_hash,
            json.dumps(record.params),
            json.dumps(record.data_snapshot_hashes),
            record.cv_scheme,
            json.dumps(record.metrics),
            record.seed,
            record.git_sha,
        )
        with self._write_lock, self._connect() as conn:
            conn.execute(_INSERT, row)

    def count(self, strategy_family: str | None = None) -> int:
        """Number of logged trials, optionally filtered by ``strategy_family``."""
        with self._connect() as conn:
            if strategy_family is None:
                cursor = conn.execute("SELECT COUNT(*) FROM trials")
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM trials WHERE strategy_family = ?",
                    (strategy_family,),
                )
            return int(cursor.fetchone()[0])

    def all(self, strategy_family: str | None = None) -> list[TrialRecord]:
        """All logged trials in insertion order, optionally filtered by family."""
        with self._connect() as conn:
            if strategy_family is None:
                cursor = conn.execute("SELECT * FROM trials ORDER BY id")
            else:
                cursor = conn.execute(
                    "SELECT * FROM trials WHERE strategy_family = ? ORDER BY id",
                    (strategy_family,),
                )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TrialRecord:
        return TrialRecord(
            trial_id=row["trial_id"],
            run_id=row["run_id"],
            ts=datetime.fromisoformat(row["ts"]),
            strategy_family=row["strategy_family"],
            config_hash=row["config_hash"],
            params=json.loads(row["params"]),
            data_snapshot_hashes=json.loads(row["data_snapshot_hashes"]),
            cv_scheme=row["cv_scheme"],
            metrics=json.loads(row["metrics"]),
            seed=row["seed"],
            git_sha=row["git_sha"],
        )
