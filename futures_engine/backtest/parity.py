"""Vectorized-vs-event-driven parity: the crux of T6 (G16).

The T5 vectorized triage path
(:func:`research.harness.positions_to_trades` -> price) and the T6 event-driven
path (Nautilus position events -> :func:`backtest.engine.build_trade_log` ->
price) must agree on the *same* reference signal and snapshot. Because both paths
end in the identical T2 pricing (``delayed_fill_prices`` + ``apply_costs``) applied
to round-turns segmented on the *same* target-change bars, they agree to
floating-point tolerance **by construction**.

Stated tolerances (:class:`ParityTolerance`) and asserted in CI:

* **trade count** -- within ``±1 per 100`` trades (from the acceptance criterion).
* **net PnL** -- within ``max(net_pnl_abs, net_pnl_rel * |pnl|)``; near-zero in
  practice.

Legitimate sources of residual difference (documented, all nil for the
integer-position reference signal):

1. *Fractional target rounding.* The event engine trades **whole contracts**;
   :func:`backtest.engine.BacktestRunner._held_targets` rounds a fractional target
   position to the nearest contract, whereas ``positions_to_trades`` keeps the
   fractional ``qty``. For an integer-valued signal (e.g. Donchian breakout in
   ``{-1, 0, 1}``) this is exactly zero.
2. *Final-open mark-to-open.* Both paths close a still-open final position at the
   last bar's open and drop a degenerate final run, so no residual arises there.
3. *Delayed-fill tail dropping.* Under ``delay_bars=1`` a trade whose fill runs off
   the end of the data is dropped by both paths identically.
4. *Floating-point summation order* -- sub-nanodollar, absorbed by ``net_pnl_abs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from futures_engine.backtest.engine import price_trades
from futures_engine.core.types import Bars, InstrumentSpec
from futures_engine.costs.model import CostConfig
from futures_engine.research.harness import positions_to_trades


@dataclass(frozen=True)
class ParityTolerance:
    """Tolerances the two paths must meet (asserted in :meth:`ParityReport.ok`)."""

    trade_count_per_100: float = 1.0
    net_pnl_abs: float = 1e-6
    net_pnl_rel: float = 1e-9


@dataclass(frozen=True)
class ParityReport:
    """Result of comparing the vectorized and event-driven paths."""

    n_trades_vectorized: int
    n_trades_event: int
    trade_count_deviation: int
    net_pnl_vectorized: float
    net_pnl_event: float
    net_pnl_deviation: float
    tolerance: ParityTolerance

    @property
    def trade_count_within_tolerance(self) -> bool:
        allowed = self.tolerance.trade_count_per_100 * max(self.n_trades_vectorized, 1) / 100.0
        # The acceptance criterion is "within +/-1 per 100 trades"; allow at least 1.
        return self.trade_count_deviation <= max(1.0, allowed)

    @property
    def net_pnl_within_tolerance(self) -> bool:
        allowed = max(
            self.tolerance.net_pnl_abs,
            self.tolerance.net_pnl_rel * abs(self.net_pnl_vectorized),
        )
        return self.net_pnl_deviation <= allowed

    @property
    def ok(self) -> bool:
        """True iff both trade-count and net-PnL deviations are within tolerance."""
        return self.trade_count_within_tolerance and self.net_pnl_within_tolerance


def _net_pnl(priced: pd.DataFrame) -> float:
    if priced.empty:
        return 0.0
    return float(priced["net_pnl_usd"].to_numpy(dtype=float).sum())


def vectorized_priced(
    held: pd.Series,
    bars: Bars,
    spec: InstrumentSpec,
    cost_cfg: CostConfig,
    qty: float,
) -> pd.DataFrame:
    """Price the vectorized path's trades through the same T2 fill + cost pipeline."""
    trades = positions_to_trades(held, bars, spec, qty)
    return price_trades(trades, bars, spec, cost_cfg)


def compare(
    vectorized: pd.DataFrame,
    event: pd.DataFrame,
    tolerance: ParityTolerance | None = None,
) -> ParityReport:
    """Compare two *priced* trade logs and report trade-count + net-PnL deviations."""
    tol = tolerance if tolerance is not None else ParityTolerance()
    n_vec, n_evt = len(vectorized), len(event)
    pnl_vec, pnl_evt = _net_pnl(vectorized), _net_pnl(event)
    return ParityReport(
        n_trades_vectorized=n_vec,
        n_trades_event=n_evt,
        trade_count_deviation=int(abs(n_vec - n_evt)),
        net_pnl_vectorized=pnl_vec,
        net_pnl_event=pnl_evt,
        net_pnl_deviation=float(np.abs(pnl_vec - pnl_evt)),
        tolerance=tol,
    )
