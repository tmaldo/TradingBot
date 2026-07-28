"""Parity crux (G16): the vectorized T5 path and the event-driven T6 path must
agree on the same reference momentum signal + snapshot, within a stated tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nautilus_trader.model.identifiers import Venue

from futures_engine.backtest.engine import build_trade_log, price_trades
from futures_engine.backtest.instrument import build_nautilus_instrument
from futures_engine.backtest.parity import ParityTolerance, compare, vectorized_priced
from futures_engine.backtest.strategy_adapter import (
    SIGNAL_REGISTRY,
    bar_timestamps_ns,
    run_event_loop,
)
from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig
from futures_engine.research.harness import causal_positions

_PARAMS = {"window": 20}
_COST = CostConfig(
    commission_per_side_usd=0.35,
    exchange_fee_per_side_usd=0.37,
    nfa_fee_per_side_usd=0.02,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=0.5,
    delay_bars=0,
)


def _run_both(
    bars: pd.DataFrame, spec: InstrumentSpec, cost_cfg: CostConfig, qty: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    signal = SIGNAL_REGISTRY["donchian_breakout"]()
    raw = signal.generate(bars, _PARAMS)
    held = causal_positions(raw)

    venue = Venue("GLBX")
    instrument = build_nautilus_instrument(spec, venue)
    ts_ns = bar_timestamps_ns(bars)
    targets_int = np.rint(held.to_numpy(dtype=float) * qty).astype("int64")
    targets = {int(ns): int(t) for ns, t in zip(ts_ns, targets_int, strict=True)}
    run = run_event_loop(bars, targets, instrument, venue, starting_balance=100_000.0)
    event = price_trades(build_trade_log(run, bars), bars, spec, cost_cfg)
    vec = vectorized_priced(held, bars, spec, cost_cfg, float(qty))
    return vec, event, held


def test_parity_trade_count_and_pnl(bars: pd.DataFrame, mes_spec: InstrumentSpec) -> None:
    vec, event, _held = _run_both(bars, mes_spec, _COST)
    report = compare(vec, event, ParityTolerance())

    # A meaningful test needs real trades on both sides.
    assert report.n_trades_vectorized > 30
    assert report.trade_count_within_tolerance, (
        f"trade count deviated: vec={report.n_trades_vectorized} event={report.n_trades_event}"
    )
    assert report.net_pnl_within_tolerance, f"net PnL deviated by {report.net_pnl_deviation}"
    assert report.ok


def test_parity_holds_under_delay_one(bars: pd.DataFrame, mes_spec: InstrumentSpec) -> None:
    cost = _COST.model_copy(update={"delay_bars": 1})
    vec, event, _held = _run_both(bars, mes_spec, cost)
    report = compare(vec, event)
    assert report.ok
    assert report.trade_count_deviation == 0
