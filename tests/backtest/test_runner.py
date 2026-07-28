"""BacktestRunner acceptance: G3 loud-fail, BacktestResultLike, reproducible
manifest, one TrialRecord per run, and the Nautilus<->T2 gross reconciliation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model.identifiers import Venue

from futures_engine.backtest.engine import (
    BacktestRunner,
    StrategyConfig,
    build_trade_log,
    reconcile_gross,
)
from futures_engine.backtest.instrument import build_nautilus_instrument
from futures_engine.backtest.strategy_adapter import (
    SIGNAL_REGISTRY,
    bar_timestamps_ns,
    run_event_loop,
)
from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig
from futures_engine.data.store import DataIntegrityError, SnapshotStore
from futures_engine.research.harness import causal_positions
from futures_engine.trials.logger import TrialLogger
from futures_engine.validation.stats import BacktestResultLike, red_flags
from tests.backtest.conftest import save_snapshot, trending_mes_1min

_STRATEGY = StrategyConfig(signal="donchian_breakout", params={"window": 20}, qty=1, seed=42)
_COST = CostConfig(
    commission_per_side_usd=0.35,
    exchange_fee_per_side_usd=0.37,
    nfa_fee_per_side_usd=0.02,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=0.5,
    delay_bars=1,
)
_CREATED = pd.Timestamp("2026-07-27T00:00:00Z").to_pydatetime()

_COST_BASE = {
    "commission_per_side_usd": 0.35,
    "exchange_fee_per_side_usd": 0.37,
    "nfa_fee_per_side_usd": 0.02,
    "spread_ticks": 1.0,
    "slippage": "fixed_ticks",
    "slippage_ticks": 0.5,
}


def _total_net(result: object) -> float:
    trades = result.trades  # type: ignore[attr-defined]
    return 0.0 if trades.empty else float(trades["net_pnl_usd"].sum())


def test_runner_delay_one_is_worse_or_equal_on_trend(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec
) -> None:
    # Runner-level (full BacktestResult) closure of the mechanism-vs-strategy gap:
    # on a clean up-trend the reference strategy is long-and-hold to the data edge;
    # a 1-bar delay pushes that held-to-edge position's tail fill off the data, so
    # the trade is dropped and the aggregate result is strictly worse. Deterministic
    # on this fixture (no seed sensitivity), complementing the single-jump mechanism
    # test in test_fill_model.py.
    bars = trending_mes_1min(1500, slope=0.5)
    snap = save_snapshot(store, bars, validation_grade=True, continuous=True)
    strat = StrategyConfig(signal="donchian_breakout", params={"window": 20}, qty=1, seed=1)
    runner = BacktestRunner(store, logger)
    r0 = runner.run(
        snap, strat, CostConfig(**_COST_BASE, delay_bars=0), mes_spec, created_at=_CREATED
    )
    r1 = runner.run(
        snap, strat, CostConfig(**_COST_BASE, delay_bars=1), mes_spec, created_at=_CREATED
    )
    net0, net1 = _total_net(r0), _total_net(r1)
    assert net0 > 0.0  # a real edge exists at delay 0
    assert net1 <= net0  # worse-or-equal under the 1-bar delay
    assert net1 != net0  # and strictly different


def test_runner_refuses_dev_grade_snapshot(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec, bars: pd.DataFrame
) -> None:
    dev_hash = save_snapshot(store, bars, validation_grade=False, continuous=True)
    runner = BacktestRunner(store, logger)
    with pytest.raises(DataIntegrityError):
        runner.run(dev_hash, _STRATEGY, _COST, mes_spec, created_at=_CREATED)


def test_runner_refuses_futures_without_continuous_meta(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec, bars: pd.DataFrame
) -> None:
    no_cont = save_snapshot(store, bars, validation_grade=True, continuous=False)
    runner = BacktestRunner(store, logger)
    with pytest.raises(DataIntegrityError):
        runner.run(no_cont, _STRATEGY, _COST, mes_spec, created_at=_CREATED)


def test_result_satisfies_backtest_result_like(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec, valid_snapshot: str
) -> None:
    runner = BacktestRunner(store, logger)
    result = runner.run(valid_snapshot, _STRATEGY, _COST, mes_spec, created_at=_CREATED)

    assert isinstance(result, BacktestResultLike)
    assert {"sharpe", "win_rate", "sharpe_gross", "n_trades", "max_dd"} <= result.metrics.keys()
    assert (result.equity.to_numpy() > 0).all()  # positive-valued equity curve
    assert len(result.returns) == len(result.trades)
    # red_flags consumes it directly.
    assert isinstance(red_flags(result, None, None), list)


def test_run_logs_exactly_one_trial(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec, valid_snapshot: str
) -> None:
    runner = BacktestRunner(store, logger)
    assert logger.count() == 0
    result = runner.run(valid_snapshot, _STRATEGY, _COST, mes_spec, created_at=_CREATED)
    assert logger.count() == 1
    record = logger.all()[0]
    assert record.config_hash == result.manifest.config_hash
    assert record.seed == 42
    assert record.data_snapshot_hashes == [valid_snapshot]


def test_manifest_is_populated_and_reproducible(
    store: SnapshotStore, logger: TrialLogger, mes_spec: InstrumentSpec, valid_snapshot: str
) -> None:
    runner1 = BacktestRunner(store, logger)
    r1 = runner1.run(valid_snapshot, _STRATEGY, _COST, mes_spec, created_at=_CREATED)

    manifest = r1.manifest
    assert manifest.git_sha and manifest.config_hash
    assert manifest.data_snapshot_hashes == [valid_snapshot]
    assert manifest.seed == 42
    assert manifest.trial_ids

    # Same inputs -> identical metrics (bit-reproducible), via a fresh logger/store view.
    logger2 = TrialLogger(store._root.parent / "trials2.db")  # type: ignore[attr-defined]
    r2 = BacktestRunner(store, logger2).run(
        valid_snapshot, _STRATEGY, _COST, mes_spec, created_at=_CREATED
    )
    assert r1.manifest.config_hash == r2.manifest.config_hash
    assert r1.metrics == r2.metrics


def test_nautilus_gross_reconciles_with_t2(mes_spec: InstrumentSpec, bars: pd.DataFrame) -> None:
    # Cross-check only (never the reported number): with fee_model=0, Nautilus
    # realised PnL == T2 gross formula on the same fill prices.
    signal = SIGNAL_REGISTRY["donchian_breakout"]()
    held = causal_positions(signal.generate(bars, {"window": 20}))
    venue = Venue("GLBX")
    instrument = build_nautilus_instrument(mes_spec, venue)
    ts_ns = bar_timestamps_ns(bars)
    targets = {
        int(ns): int(t)
        for ns, t in zip(ts_ns, np.rint(held.to_numpy(dtype=float)).astype("int64"), strict=True)
    }
    run = run_event_loop(bars, targets, instrument, venue, starting_balance=100_000.0)
    assert len(run.closed) > 20
    assert reconcile_gross(run, mes_spec) < 1e-6
    # Sanity: boundaries are real trades.
    assert not build_trade_log(run, bars).empty
