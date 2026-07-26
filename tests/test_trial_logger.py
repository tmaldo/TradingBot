"""Tests for the append-only trial logger (task T0 acceptance: TrialLogger).

Covers: append-only API (no update/delete), count() across process restarts,
thread-safe concurrent writes, and rejection of records missing mandatory
provenance fields (data_snapshot_hashes, config_hash, seed, git_sha).
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from futures_engine.trials.logger import TrialLogger, TrialRecord

TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _full_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trial_id": "t-0",
        "run_id": "run-0",
        "ts": TS,
        "strategy_family": "trend_momentum",
        "config_hash": "cfg-abc",
        "params": {"lookback": 96, "vol_target": 0.15},
        "data_snapshot_hashes": ["snap-1", "snap-2"],
        "cv_scheme": "purged_kfold",
        "metrics": {"sharpe": 1.4, "dsr": 0.6},
        "seed": 7,
        "git_sha": "deadbeef1234",
    }
    kwargs.update(overrides)
    return kwargs


def _record(**overrides: Any) -> TrialRecord:
    return TrialRecord(**_full_kwargs(**overrides))


# --- basic logging + count ---------------------------------------------------


def test_log_then_count(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    assert logger.count() == 0
    logger.log(_record(trial_id="t-1"))
    logger.log(_record(trial_id="t-2"))
    assert logger.count() == 2


def test_count_filters_by_strategy_family(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    logger.log(_record(trial_id="a", strategy_family="trend_momentum"))
    logger.log(_record(trial_id="b", strategy_family="trend_momentum"))
    logger.log(_record(trial_id="c", strategy_family="mean_reversion"))
    assert logger.count() == 3
    assert logger.count("trend_momentum") == 2
    assert logger.count("mean_reversion") == 1
    assert logger.count("does_not_exist") == 0


def test_all_round_trips_records_in_insertion_order(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    logger.log(_record(trial_id="first"))
    logger.log(_record(trial_id="second"))

    records = logger.all()
    assert [r.trial_id for r in records] == ["first", "second"]

    r = records[0]
    assert r.ts == TS
    assert r.params == {"lookback": 96, "vol_target": 0.15}
    assert r.data_snapshot_hashes == ["snap-1", "snap-2"]
    assert r.metrics == {"sharpe": 1.4, "dsr": 0.6}
    assert r.seed == 7
    assert r.git_sha == "deadbeef1234"


def test_all_filters_by_strategy_family(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    logger.log(_record(trial_id="a", strategy_family="trend_momentum"))
    logger.log(_record(trial_id="c", strategy_family="mean_reversion"))
    families = {r.strategy_family for r in logger.all("trend_momentum")}
    assert families == {"trend_momentum"}


# --- persistence across restarts ---------------------------------------------


def test_count_persists_for_new_instance(tmp_path: Path) -> None:
    db = tmp_path / "trials.db"
    TrialLogger(db).log(_record(trial_id="t-1"))
    # A brand-new logger object reading the same on-disk file sees the record.
    assert TrialLogger(db).count() == 1


def test_count_persists_across_processes(tmp_path: Path) -> None:
    db = tmp_path / "trials.db"
    child = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "from futures_engine.trials.logger import TrialLogger, TrialRecord\n"
        "logger = TrialLogger(sys.argv[1])\n"
        "for i in range(int(sys.argv[2])):\n"
        "    logger.log(TrialRecord(trial_id=f'child-{i}', run_id='r', "
        "ts=datetime(2026,1,1,tzinfo=timezone.utc), strategy_family='trend', "
        "config_hash='c', params={}, data_snapshot_hashes=['h'], "
        "cv_scheme='purged_kfold', metrics={'sharpe':1.0}, seed=1, "
        "git_sha='abc'))\n"
    )
    subprocess.run([sys.executable, "-c", child, str(db), "5"], check=True)
    # Fresh process (this one) opens the same file and sees all committed rows.
    assert TrialLogger(db).count() == 5


# --- concurrency -------------------------------------------------------------


def test_concurrent_writes_are_safe(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    n_threads, per_thread = 8, 25
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(per_thread):
                logger.log(_record(trial_id=f"w{worker_id}-{i}"))
        except Exception as exc:  # surface any thread failure to the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread failures: {errors}"
    assert logger.count() == n_threads * per_thread
    assert len(logger.all()) == n_threads * per_thread


# --- append-only guarantees --------------------------------------------------


def test_no_mutation_api_exists(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    for forbidden in ("update", "delete", "remove", "clear", "drop", "pop"):
        assert not hasattr(logger, forbidden)


def test_duplicate_trial_id_is_rejected(tmp_path: Path) -> None:
    logger = TrialLogger(tmp_path / "trials.db")
    logger.log(_record(trial_id="dup"))
    with pytest.raises(sqlite3.IntegrityError):
        logger.log(_record(trial_id="dup"))


# --- mandatory provenance fields ---------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "data_snapshot_hashes",
        "config_hash",
        "seed",
        "git_sha",
        "trial_id",
        "run_id",
        "ts",
        "strategy_family",
        "params",
        "cv_scheme",
        "metrics",
    ],
)
def test_missing_required_field_is_rejected(field: str) -> None:
    kwargs = _full_kwargs()
    del kwargs[field]
    with pytest.raises(ValidationError):
        TrialRecord(**kwargs)
