"""Hand-computed toy cross-check: 3 trades on synthetic bars, exact USD equality.

Pins the vectorized position->trades->costs path against arithmetic worked out by
hand (including every cost component), so a regression in the PnL engine or the
cost decomposition cannot slip through as an approximate match.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig, apply_costs, delayed_fill_prices
from futures_engine.research.harness import positions_to_trades

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)
FIXED_CFG = CostConfig(
    commission_per_side_usd=0.15,
    exchange_fee_per_side_usd=0.35,
    nfa_fee_per_side_usd=0.01,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=1.0,
    delay_bars=0,
)

# Per round-turn (qty=1) cost: commission 0.15*2 + fees (0.35+0.01)*2
#   + spread 1.0*1.25 + slippage 1.0*1.25 = 0.30 + 0.72 + 1.25 + 1.25 = 3.52
_COST_PER_TRADE = 3.52


def _bars() -> pd.DataFrame:
    opens = [100.0, 102.0, 105.0, 103.0, 101.0, 99.0, 100.0, 104.0, 108.0]
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(len(opens))])
    close = [o + 0.25 for o in opens]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 1.0 for o, c in zip(opens, close, strict=True)],
            "low": [min(o, c) - 1.0 for o, c in zip(opens, close, strict=True)],
            "close": close,
            "volume": [1000.0] * len(opens),
        },
        index=idx,
    )


def test_three_trade_toy_case_exact_usd() -> None:
    bars = _bars()
    # held: long bars1-2, flat, short bars4-5, flat, long bar7, flat.
    held = pd.Series([0, 1, 1, 0, -1, -1, 0, 1, 0], index=bars.index, dtype=float)
    trades = positions_to_trades(held, bars, MES, qty=1)
    priced = apply_costs(trades, MES, FIXED_CFG)

    assert len(priced) == 3
    # gross: long (103-102)*5=5 ; short (101-100)*5=5 ; long (108-104)*5=20
    assert priced["gross_pnl_usd"].tolist() == [5.0, 5.0, 20.0]
    # every trade pays the exact same round-turn cost stack
    for _, row in priced.iterrows():
        assert math.isclose(row["commission_usd"], 0.30)
        assert math.isclose(row["fees_usd"], 0.72)
        assert math.isclose(row["spread_cost_usd"], 1.25)
        assert math.isclose(row["slippage_usd"], 1.25)
    # net = gross - 3.52 each (float repr of 3.52 -> compare at machine precision)
    assert priced["net_pnl_usd"].tolist() == pytest.approx(
        [5.0 - _COST_PER_TRADE, 5.0 - _COST_PER_TRADE, 20.0 - _COST_PER_TRADE], abs=1e-9
    )
    assert priced["net_pnl_usd"].sum() == pytest.approx(30.0 - 3 * _COST_PER_TRADE, abs=1e-9)


def test_toy_case_gross_matches_delayed_fill_at_delay_zero() -> None:
    bars = _bars()
    held = pd.Series([0, 1, 1, 0, -1, -1, 0, 1, 0], index=bars.index, dtype=float)
    trades = positions_to_trades(held, bars, MES, qty=1)
    refilled = delayed_fill_prices(trades, bars, MES, delay_bars=0)
    assert refilled["gross_pnl_usd"].tolist() == [5.0, 5.0, 20.0]
