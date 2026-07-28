"""Fill model (G8/G16): costs come from T2 only; the 1-bar delay is honoured and
erodes a momentum edge on a trending fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futures_engine.backtest.engine import price_trades
from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import (
    COST_COLUMNS,
    CostConfig,
    apply_costs,
    delayed_fill_prices,
)

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)

_COST0 = CostConfig(
    commission_per_side_usd=0.35,
    exchange_fee_per_side_usd=0.37,
    nfa_fee_per_side_usd=0.02,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=0.5,
    delay_bars=0,
)


def _flat_bars(opens: list[float]) -> pd.DataFrame:
    n = len(opens)
    idx = pd.DatetimeIndex(
        pd.date_range("2022-01-03", periods=n, freq="1min", tz="UTC"), name="timestamp"
    )
    o = np.array(opens, dtype=float)
    return pd.DataFrame(
        {"open": o, "high": o + 1.0, "low": o - 1.0, "close": o, "volume": 100.0}, index=idx
    )


def _one_long_trade(bars: pd.DataFrame, entry_i: int, exit_i: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_ts": [bars.index[entry_i]],
            "exit_ts": [bars.index[exit_i]],
            "side": ["long"],
            "qty": [1.0],
        }
    )


def test_costs_are_the_t2_config_not_reimplemented() -> None:
    # A step-up "trend": open jumps 100 -> 110 between bar 10 and bar 11.
    bars = _flat_bars([100.0] * 11 + [110.0] * 9)
    trades = _one_long_trade(bars, entry_i=5, exit_i=15)

    priced = price_trades(trades, bars, MES, _COST0)

    # price_trades must delegate to T2 verbatim: identical to apply_costs on the
    # delayed fills. This is what proves costs are never re-implemented inline.
    expected = apply_costs(delayed_fill_prices(trades, bars, MES, 0), MES, _COST0)
    for col in COST_COLUMNS:
        assert priced[col].to_numpy() == expected[col].to_numpy()

    # And each cost equals its T2 formula for qty=1 round-turn.
    row = priced.iloc[0]
    assert row["commission_usd"] == _COST0.commission_per_side_usd * 2
    assert row["fees_usd"] == (_COST0.exchange_fee_per_side_usd + _COST0.nfa_fee_per_side_usd) * 2
    assert row["spread_cost_usd"] == _COST0.spread_ticks * MES.tick_value
    assert row["slippage_usd"] == _COST0.slippage_ticks * MES.tick_value
    assert np.isclose(
        row["net_pnl_usd"],
        row["gross_pnl_usd"]
        - row["commission_usd"]
        - row["fees_usd"]
        - row["spread_cost_usd"]
        - row["slippage_usd"],
    )


def test_delay_one_differs_from_delay_zero_and_is_worse_on_trend() -> None:
    # Favourable open-to-open jump immediately AFTER the signal bar: delay 0 captures
    # it, delay 1 forfeits it -> a strictly different and worse (edge-eroding) result.
    bars = _flat_bars([100.0] * 11 + [110.0] * 9)
    trades = _one_long_trade(bars, entry_i=10, exit_i=15)

    priced0 = price_trades(trades, bars, MES, _COST0)
    priced1 = price_trades(trades, bars, MES, _COST0.model_copy(update={"delay_bars": 1}))

    gross0 = float(priced0["gross_pnl_usd"].iloc[0])
    gross1 = float(priced1["gross_pnl_usd"].iloc[0])
    assert gross0 == (110.0 - 100.0) * MES.multiplier  # +50
    assert gross1 == 0.0  # entered after the jump, exited flat
    assert gross1 != gross0  # strictly different
    assert gross1 <= gross0  # worse-or-equal on the trend


def test_delay_one_drops_trades_that_run_off_the_end() -> None:
    # A trade whose exit is the last bar cannot be delayed -> dropped (documented,
    # mirrors the T5 harness room filter), rather than raising.
    bars = _flat_bars([100.0, 101.0, 102.0, 103.0])
    trades = _one_long_trade(bars, entry_i=1, exit_i=3)  # exit is last bar
    priced = price_trades(trades, bars, MES, _COST0.model_copy(update={"delay_bars": 1}))
    assert priced.empty
