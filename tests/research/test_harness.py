"""Tests for the vectorized sweep harness (G10 honest trial logging).

The sweep's contract: refuse non-validation-grade data, evaluate every grid
combination net of costs (gross + 1-bar-delay variants alongside), and log
*exactly one* TrialRecord per combination -- including combinations that error
out -- so the Deflated Sharpe Ratio's trial count can never be understated.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from futures_engine.costs.model import CostConfig
from futures_engine.data.store import DataIntegrityError
from futures_engine.research.harness import causal_positions, config_hash, sweep
from futures_engine.research.strategies.momentum import DonchianBreakout
from futures_engine.trials.logger import TrialLogger
from futures_engine.validation.splitters import PurgedKFold
from tests.research.conftest import save_snapshot, synthetic_mes_1min

FIXED_TS = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
COST_CFG = CostConfig(
    commission_per_side_usd=0.15,
    exchange_fee_per_side_usd=0.35,
    nfa_fee_per_side_usd=0.01,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=1.0,
    delay_bars=0,
)


def _run_sweep(store, mes_spec, logger, grid, *, n_bars=3000):  # type: ignore[no-untyped-def]
    bars = synthetic_mes_1min(n_bars)
    snap = save_snapshot(store, bars, validation_grade=True)
    return sweep(
        DonchianBreakout(),
        grid,
        [snap],
        COST_CFG,
        PurgedKFold(n_splits=4, embargo_frac=0.01),
        logger,
        seed=11,
        store=store,
        spec=mes_spec,
        git_sha="testsha0000",
        ts=FIXED_TS,
    )


# --- causal shift + config hash ----------------------------------------------


def test_causal_positions_shifts_raw_by_one_bar() -> None:
    raw = pd.Series([1.0, 1.0, -1.0, 0.0], index=pd.RangeIndex(4))
    held = causal_positions(raw)
    expected = pd.Series([0.0, 1.0, 1.0, -1.0], index=pd.RangeIndex(4))
    pd.testing.assert_series_equal(held, expected, check_names=False)


def test_config_hash_is_deterministic_and_param_sensitive() -> None:
    a = config_hash({"window": 10}, COST_CFG, seed=1)
    b = config_hash({"window": 10}, COST_CFG, seed=1)
    c = config_hash({"window": 20}, COST_CFG, seed=1)
    assert a == b
    assert len(a) == 64 and all(ch in "0123456789abcdef" for ch in a)
    assert a != c


# --- sweep behaviour ---------------------------------------------------------


def test_sweep_logs_exactly_one_trial_per_combo(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    result = _run_sweep(store, mes_spec, logger, {"window": [5, 10, 20]})
    assert result.n_trials_logged == 3
    assert logger.count() == 3
    assert len(result.table) == 3


def test_sweep_n_trials_logged_equals_grid_cross_product(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    # 3 x 2 = 6 combinations even though window/extra are independent axes.
    result = _run_sweep(store, mes_spec, logger, {"window": [5, 10, 20], "unused": [1, 2]})
    assert result.n_trials_logged == 6
    assert logger.count() == 6


def test_sweep_refuses_non_validation_grade(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    bars = synthetic_mes_1min(500)
    dev_snap = save_snapshot(store, bars, validation_grade=False)
    with pytest.raises(DataIntegrityError):
        sweep(
            DonchianBreakout(),
            {"window": [10]},
            [dev_snap],
            COST_CFG,
            PurgedKFold(n_splits=3, embargo_frac=0.0),
            logger,
            seed=1,
            store=store,
            spec=mes_spec,
            git_sha="testsha0000",
            ts=FIXED_TS,
        )


def test_sweep_errored_combo_still_logs_with_error_metric(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    # window=0 is an invalid parameter -> the combo errors, but must still log.
    result = _run_sweep(store, mes_spec, logger, {"window": [5, 0, 20]})
    assert result.n_trials_logged == 3
    assert logger.count() == 3
    errored = result.table[result.table["error"] == 1.0]
    assert len(errored) == 1
    assert int(errored.iloc[0]["window"]) == 0
    logged_errors = [r for r in logger.all() if r.metrics.get("error") == 1.0]
    assert len(logged_errors) == 1


def test_sweep_table_has_gross_net_and_delay1_columns(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    result = _run_sweep(store, mes_spec, logger, {"window": [10, 30]})
    expected = ("sharpe_gross", "sharpe_net", "sharpe_net_delay1", "n_trades", "win_rate", "max_dd")
    for col in expected:
        assert col in result.table.columns


def test_sweep_best_maximises_sharpe_net(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    result = _run_sweep(store, mes_spec, logger, {"window": [8, 16, 32, 64]})
    ok = result.table[result.table["error"] == 0.0]
    best_net = ok["sharpe_net"].max()
    assert result.best["sharpe_net"] == pytest.approx(best_net)


def test_sweep_trial_records_carry_full_provenance(store, mes_spec) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(store._root / "trials.db")  # type: ignore[attr-defined]
    bars = synthetic_mes_1min(3000)
    snap = save_snapshot(store, bars, validation_grade=True)
    sweep(
        DonchianBreakout(),
        {"window": [10, 20]},
        [snap],
        COST_CFG,
        PurgedKFold(n_splits=4, embargo_frac=0.01),
        logger,
        seed=11,
        store=store,
        spec=mes_spec,
        git_sha="testsha0000",
        ts=FIXED_TS,
    )
    records = logger.all()
    assert {r.config_hash for r in records}.__len__() == 2  # distinct per combo
    for rec in records:
        assert rec.strategy_family == "trend_momentum"
        assert rec.data_snapshot_hashes == [snap]
        assert rec.seed == 11
        assert rec.git_sha == "testsha0000"
        assert len(rec.config_hash) == 64
        assert "PurgedKFold" in rec.cv_scheme
