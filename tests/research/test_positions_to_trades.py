"""Tests for positions_to_trades: float position series -> T2 TradeLog.

These pin the BINDING delay-fill causality convention's *lower half*: a held
target-position series (already causal -- ``positions[t]`` is the position held
*during* bar ``t``) is converted into round-turn trades whose ``entry_ts`` /
``exit_ts`` name the bar at whose OPEN the order acts. A position flip is two
trades (close the old, open the new) sharing the flip bar as the crossover fill.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import TRADE_COLUMNS, delayed_fill_prices
from futures_engine.research.harness import _delayed_fill_prices_vec, positions_to_trades

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)


def _bars(opens: list[float]) -> pd.DataFrame:
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(len(opens))])
    close = [o + 0.25 for o in opens]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, close, strict=True)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, close, strict=True)],
            "close": close,
            "volume": [1000.0] * len(opens),
        },
        index=idx,
    )


def _pos(values: list[float], bars: pd.DataFrame) -> pd.Series:
    return pd.Series(values, index=bars.index, dtype=float)


def test_output_has_trade_columns() -> None:
    bars = _bars([100.0, 101.0, 102.0, 103.0])
    out = positions_to_trades(_pos([0.0, 1.0, 1.0, 0.0], bars), bars, MES, qty=1)
    assert list(out.columns) == list(TRADE_COLUMNS)


def test_single_long_run_makes_one_trade() -> None:
    # held long during bars 1,2,3 -> enter at open[1], exit at open[4] (flip to flat).
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
    out = positions_to_trades(_pos([0.0, 1.0, 1.0, 1.0, 0.0], bars), bars, MES, qty=1)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["side"] == "long"
    assert row["entry_ts"] == bars.index[1]
    assert row["exit_ts"] == bars.index[4]
    assert math.isclose(row["entry_px"], 101.0)
    assert math.isclose(row["exit_px"], 104.0)
    assert math.isclose(row["qty"], 1.0)
    # gross = (104 - 101) * multiplier(5) * qty(1) = 15
    assert math.isclose(row["gross_pnl_usd"], 15.0)


def test_short_run_gross_sign() -> None:
    # declining opens so a short is profitable.
    bars = _bars([100.0, 105.0, 104.0, 101.0, 100.0])
    out = positions_to_trades(_pos([0.0, -1.0, -1.0, 0.0, 0.0], bars), bars, MES, qty=1)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["side"] == "short"
    # enter short at open[1]=105, exit at open[3]=101 -> gross = (105-101)*5 = 20
    assert math.isclose(row["gross_pnl_usd"], 20.0)


def test_flip_long_to_short_makes_two_trades() -> None:
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
    out = positions_to_trades(_pos([0.0, 1.0, 1.0, -1.0, -1.0], bars), bars, MES, qty=1)
    assert len(out) == 2
    long_row, short_row = out.iloc[0], out.iloc[1]
    assert long_row["side"] == "long"
    assert short_row["side"] == "short"
    # the flip bar is the long's exit AND the short's entry (crossover at one open).
    assert long_row["exit_ts"] == bars.index[3]
    assert short_row["entry_ts"] == bars.index[3]
    # long: (103-101)*5 = 10 ; short closes at last bar open[4]: (103-104)*5 = -5
    assert math.isclose(long_row["gross_pnl_usd"], 10.0)
    assert math.isclose(short_row["gross_pnl_usd"], -5.0)


def test_fractional_position_scales_qty() -> None:
    bars = _bars([100.0, 101.0, 102.0, 103.0])
    out = positions_to_trades(_pos([0.0, 0.5, 0.5, 0.0], bars), bars, MES, qty=4)
    assert len(out) == 1
    # contracts = |0.5| * 4 = 2
    assert math.isclose(out.iloc[0]["qty"], 2.0)


def test_flat_series_makes_no_trades() -> None:
    bars = _bars([100.0, 101.0, 102.0])
    out = positions_to_trades(_pos([0.0, 0.0, 0.0], bars), bars, MES, qty=1)
    assert len(out) == 0
    assert list(out.columns) == list(TRADE_COLUMNS)


def test_gross_matches_delayed_fill_prices_at_delay_zero() -> None:
    # positions_to_trades must be self-consistent with the T2 delay semantic at 0.
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    trades = positions_to_trades(_pos([0.0, 1.0, 1.0, -1.0, -1.0, 0.0], bars), bars, MES, qty=1)
    refilled = delayed_fill_prices(trades, bars, MES, delay_bars=0)
    pd.testing.assert_series_equal(
        trades["gross_pnl_usd"], refilled["gross_pnl_usd"], check_names=False
    )


def test_index_mismatch_raises() -> None:
    bars = _bars([100.0, 101.0, 102.0])
    bad = pd.Series([0.0, 1.0], index=bars.index[:2], dtype=float)
    with pytest.raises(ValueError, match="index"):
        positions_to_trades(bad, bars, MES, qty=1)


@pytest.mark.parametrize("delay", [0, 1])
def test_delay_semantic_matches_t2(delay: int) -> None:
    # The harness's fast vectorised fill must be byte-identical to T2's helper,
    # so the "one delay semantic everywhere" guarantee holds despite reimplementing.
    # positions close by bar 6, so bar 7 remains a valid delay-1 fill target for all.
    bars = _bars([100.0, 101.0, 103.0, 102.0, 106.0, 108.0, 107.0, 109.0])
    trades = positions_to_trades(
        _pos([0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0], bars), bars, MES, qty=2
    )
    reference = delayed_fill_prices(trades, bars, MES, delay_bars=delay)
    fast = _delayed_fill_prices_vec(trades, bars.index, bars["open"].to_numpy(), MES, delay)
    for col in ("entry_ts", "exit_ts", "entry_px", "exit_px", "gross_pnl_usd"):
        pd.testing.assert_series_equal(fast[col], reference[col], check_names=False)
