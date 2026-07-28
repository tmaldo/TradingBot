"""Event-driven validation backtester (T6).

Nautilus Trader is the event **sequencer** and order/position state machine;
T2 (:mod:`futures_engine.costs.model`) is the single source of truth for every
fill price and cost. The public surface:

* :class:`~futures_engine.backtest.engine.BacktestRunner` /
  :class:`~futures_engine.backtest.engine.BacktestResult` /
  :class:`~futures_engine.backtest.engine.StrategyConfig`
* :func:`~futures_engine.backtest.parity.compare` -- vectorized-vs-event parity.
"""

from __future__ import annotations

from futures_engine.backtest.engine import (
    BacktestResult,
    BacktestRunner,
    StrategyConfig,
)
from futures_engine.backtest.parity import ParityReport, ParityTolerance, compare

__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "ParityReport",
    "ParityTolerance",
    "StrategyConfig",
    "compare",
]
