"""Regression test for the BINDING delay-fill causality convention.

Constructs a price path where the intrabar sign ``sign(close - open)`` perfectly
predicts the *next* open-to-open move. A pipeline that (wrongly) acts on same-bar
information is therefore a flawless look-ahead oracle, while the causal pipeline
-- which holds each decision only from the following bar -- earns the opposite.
The causal path must be *measurably worse*: proof our harness would expose the
legacy same-bar-fill bug rather than bank the phantom edge.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig, apply_costs, delayed_fill_prices
from futures_engine.research.harness import _evaluate, causal_positions, positions_to_trades

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)
COST_CFG = CostConfig(
    commission_per_side_usd=0.15,
    exchange_fee_per_side_usd=0.35,
    nfa_fee_per_side_usd=0.01,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=1.0,
    delay_bars=0,
)


class _IntrabarSign:
    """Look-ahead-prone signal: raw[t] = sign(close[t] - open[t])."""

    family = "trend_momentum"

    def generate(self, bars: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.Series(np.sign(bars["close"] - bars["open"]).to_numpy(), index=bars.index)


def _oracle_bars(n: int = 60) -> pd.DataFrame:
    # Alternating regime r[t] = (-1)^t; next open move = r[t] * move[t] (varying),
    # and close[t] = open[t] + r[t]*0.25 so sign(close-open) == r[t].
    regime = np.array([1.0 if t % 2 == 0 else -1.0 for t in range(n)])
    move = 4.0 + (np.arange(n) % 3) * 0.5
    opens = np.empty(n)
    opens[0] = 1000.0
    for t in range(n - 1):
        opens[t + 1] = opens[t] + regime[t] * move[t]
    close = opens + regime * 0.25
    t0 = datetime(2026, 1, 5, tzinfo=UTC)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(n)])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) + 0.5,
            "low": np.minimum(opens, close) - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _net_total(trades: pd.DataFrame) -> float:
    priced = apply_costs(delayed_fill_prices(trades, _oracle_bars(), MES, 0), MES, COST_CFG)
    return float(priced["net_pnl_usd"].sum())


def test_same_bar_fill_beats_causal_exposing_lookahead() -> None:
    bars = _oracle_bars()
    raw = _IntrabarSign().generate(bars, {})

    buggy_trades = positions_to_trades(raw, bars, MES, qty=1)  # same-bar (no shift)
    causal_trades = positions_to_trades(causal_positions(raw), bars, MES, qty=1)

    buggy = _evaluate(buggy_trades, bars, MES, COST_CFG)
    causal = _evaluate(causal_trades, bars, MES, COST_CFG)

    # The look-ahead pipeline is a near-perfect winner; the causal one is not.
    assert buggy["win_rate"] == 1.0
    assert causal["win_rate"] == 0.0
    assert buggy["sharpe_net"] > causal["sharpe_net"]
    assert buggy["sharpe_net"] > 0.0 > causal["sharpe_net"]
    # ... and the phantom edge is measurably large in USD.
    assert _net_total(buggy_trades) > 0.0 > _net_total(causal_trades)
