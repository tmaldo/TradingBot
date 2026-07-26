"""Tests for the transaction-cost model (task T2, Global Constraint G8/G16).

Every reported result must be net of costs, and gross-vs-net must always be
reported. These tests hand-verify the per-trade cost decomposition against
worked examples for MES and MNQ (the hardest-tested component per G16).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import (
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
MNQ = InstrumentSpec(
    symbol_root="MNQ",
    exchange="CME",
    tick_size=0.25,
    tick_value=0.50,
    multiplier=2,
    currency="USD",
)

# A cost profile whose all-in round-turn for 1 contract is
# (0.15 + 0.35 + 0.01) * 2 = $1.02 (commission + exchange + NFA per side, both sides).
FIXED_CFG = CostConfig(
    commission_per_side_usd=0.15,
    exchange_fee_per_side_usd=0.35,
    nfa_fee_per_side_usd=0.01,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=1.0,
    delay_bars=0,
)


def _one_trade(gross: float, qty: int = 1, *, side: str = "long") -> pd.DataFrame:
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    return pd.DataFrame(
        {
            "entry_ts": [t0],
            "exit_ts": [t0 + timedelta(minutes=5)],
            "side": [side],
            "qty": [qty],
            "entry_px": [5000.0],
            "exit_px": [5000.0 + gross],
            "gross_pnl_usd": [gross],
        }
    )


# --- CostConfig validation ---------------------------------------------------


def test_delay_bars_must_be_zero_or_one() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CostConfig(
            commission_per_side_usd=0.15,
            exchange_fee_per_side_usd=0.35,
            nfa_fee_per_side_usd=0.01,
            spread_ticks=1.0,
            slippage="fixed_ticks",
            slippage_ticks=1.0,
            delay_bars=2,
        )


def test_all_in_round_turn_lands_in_expected_band() -> None:
    # Documented target: all-in round-turn for 1 micro contract in ~$1.02-$1.04.
    all_in_rt = (
        FIXED_CFG.commission_per_side_usd
        + FIXED_CFG.exchange_fee_per_side_usd
        + FIXED_CFG.nfa_fee_per_side_usd
    ) * 2
    assert 1.02 <= all_in_rt <= 1.04


# --- apply_costs: worked examples (hand-verified) ----------------------------


def test_apply_costs_mes_worked_example() -> None:
    # MES, qty=2, gross=$100.00, FIXED_CFG.
    #   commission = 0.15/side * 2 sides * 2 qty          = 0.60
    #   fees       = (0.35 + 0.01)/side * 2 sides * 2 qty = 1.44
    #   spread     = 1.0 tick * $1.25/tick * 2 qty        = 2.50
    #   slippage   = 1.0 tick * $1.25/tick * 2 qty        = 2.50
    #   total costs                                       = 7.04
    #   net = 100.00 - 7.04                               = 92.96
    out = apply_costs(_one_trade(100.0, qty=2), MES, FIXED_CFG)
    row = out.iloc[0]
    assert math.isclose(row["commission_usd"], 0.60)
    assert math.isclose(row["fees_usd"], 1.44)
    assert math.isclose(row["spread_cost_usd"], 2.50)
    assert math.isclose(row["slippage_usd"], 2.50)
    assert math.isclose(row["net_pnl_usd"], 92.96)


def test_apply_costs_mnq_worked_example() -> None:
    # MNQ, qty=1, gross=$50.00, FIXED_CFG.
    #   commission = 0.15 * 2 * 1              = 0.30
    #   fees       = (0.35 + 0.01) * 2 * 1     = 0.72
    #   spread     = 1.0 * $0.50 * 1           = 0.50
    #   slippage   = 1.0 * $0.50 * 1           = 0.50
    #   total                                  = 2.02
    #   net = 50.00 - 2.02                     = 47.98
    out = apply_costs(_one_trade(50.0, qty=1), MNQ, FIXED_CFG)
    row = out.iloc[0]
    assert math.isclose(row["commission_usd"], 0.30)
    assert math.isclose(row["fees_usd"], 0.72)
    assert math.isclose(row["spread_cost_usd"], 0.50)
    assert math.isclose(row["slippage_usd"], 0.50)
    assert math.isclose(row["net_pnl_usd"], 47.98)


def test_net_equals_gross_minus_sum_of_costs_exactly() -> None:
    out = apply_costs(_one_trade(37.5, qty=3), MES, FIXED_CFG)
    row = out.iloc[0]
    total = row["commission_usd"] + row["fees_usd"] + row["spread_cost_usd"] + row["slippage_usd"]
    assert math.isclose(row["net_pnl_usd"], row["gross_pnl_usd"] - total)


def test_apply_costs_preserves_input_columns_and_is_a_copy() -> None:
    trades = _one_trade(100.0, qty=1)
    out = apply_costs(trades, MES, FIXED_CFG)
    for col in ("entry_ts", "exit_ts", "side", "qty", "entry_px", "exit_px", "gross_pnl_usd"):
        assert col in out.columns
    # input frame untouched (no cost columns leaked back into it)
    assert "net_pnl_usd" not in trades.columns


def test_apply_costs_rejects_missing_required_column() -> None:
    trades = _one_trade(100.0).drop(columns=["gross_pnl_usd"])
    with pytest.raises(ValueError, match="gross_pnl_usd"):
        apply_costs(trades, MES, FIXED_CFG)


# --- vol-scaled slippage -----------------------------------------------------


def test_vol_scaled_slippage_worked_example() -> None:
    # vol_scaled: effective slip ticks = slippage_ticks * atr_ticks_at_entry.
    # MES qty=1, slippage_ticks=0.5 (coefficient), atr_ticks_at_entry=8
    #   effective = 0.5 * 8 = 4 ticks -> 4 * $1.25 * 1 = $5.00
    cfg = FIXED_CFG.model_copy(update={"slippage": "vol_scaled", "slippage_ticks": 0.5})
    trades = _one_trade(100.0, qty=1)
    trades["atr_ticks_at_entry"] = [8.0]
    out = apply_costs(trades, MES, cfg)
    assert math.isclose(out.iloc[0]["slippage_usd"], 5.0)


def test_vol_scaled_without_atr_column_raises_loudly() -> None:
    cfg = FIXED_CFG.model_copy(update={"slippage": "vol_scaled"})
    with pytest.raises(ValueError, match="atr_ticks_at_entry"):
        apply_costs(_one_trade(100.0), MES, cfg)


# --- delay_bars semantics ----------------------------------------------------


def _bars() -> pd.DataFrame:
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(5)])
    # Deliberately non-linear opens so delay changes the P&L.
    return pd.DataFrame({"open": [100.0, 100.5, 102.0, 101.0, 104.0]}, index=idx)


def _signal_trade() -> pd.DataFrame:
    bars = _bars()
    return pd.DataFrame(
        {
            "entry_ts": [bars.index[0]],  # t0
            "exit_ts": [bars.index[2]],  # t2
            "side": ["long"],
            "qty": [1],
            "entry_px": [100.0],
            "exit_px": [102.0],
            "gross_pnl_usd": [(102.0 - 100.0) * MES.multiplier],
        }
    )


def test_delay_zero_fills_at_signal_bar_open() -> None:
    out = delayed_fill_prices(_signal_trade(), _bars(), MES, delay_bars=0)
    row = out.iloc[0]
    assert math.isclose(row["entry_px"], 100.0)  # t0 open
    assert math.isclose(row["exit_px"], 102.0)  # t2 open
    assert math.isclose(row["gross_pnl_usd"], (102.0 - 100.0) * 5)


def test_delay_one_shifts_execution_to_next_bar_open() -> None:
    # signal at t0/t2 -> execution at t1 open (100.5) and t3 open (101.0)
    #   gross = (101.0 - 100.5) * 5 * 1 = 2.5
    out = delayed_fill_prices(_signal_trade(), _bars(), MES, delay_bars=1)
    row = out.iloc[0]
    assert math.isclose(row["entry_px"], 100.5)
    assert math.isclose(row["exit_px"], 101.0)
    assert math.isclose(row["gross_pnl_usd"], (101.0 - 100.5) * 5)
    # timestamps advance to the fill bar
    assert row["entry_ts"] == _bars().index[1]
    assert row["exit_ts"] == _bars().index[3]


def test_delay_one_short_side_gross_sign() -> None:
    trades = _signal_trade()
    trades["side"] = ["short"]
    out = delayed_fill_prices(trades, _bars(), MES, delay_bars=1)
    # short: gross = (entry_px - exit_px) * mult = (100.5 - 101.0) * 5 = -2.5
    assert math.isclose(out.iloc[0]["gross_pnl_usd"], (100.5 - 101.0) * 5)


def test_delay_out_of_range_raises() -> None:
    bars = _bars()
    trades = pd.DataFrame(
        {
            "entry_ts": [bars.index[0]],
            "exit_ts": [bars.index[4]],  # last bar; +1 is out of range
            "side": ["long"],
            "qty": [1],
            "entry_px": [100.0],
            "exit_px": [104.0],
            "gross_pnl_usd": [0.0],
        }
    )
    with pytest.raises(ValueError, match="out of range"):
        delayed_fill_prices(trades, bars, MES, delay_bars=1)


def test_delay_unknown_timestamp_raises() -> None:
    bars = _bars()
    trades = _signal_trade()
    trades.loc[0, "entry_ts"] = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)  # not in bars
    with pytest.raises(ValueError, match="not found"):
        delayed_fill_prices(trades, bars, MES, delay_bars=1)
