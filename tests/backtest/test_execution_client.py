"""The backtest ExecutionClient (G13/G14): bar-open fills honouring the delay
convention, netted positions / account tracking, and Protocol conformance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futures_engine.core.types import InstrumentSpec
from futures_engine.execution import (
    AccountState,
    BacktestExecutionClient,
    ExecutionClient,
    Order,
    Position,
)

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)


def _bars(opens: list[float]) -> pd.DataFrame:
    n = len(opens)
    idx = pd.DatetimeIndex(
        pd.date_range("2022-06-01", periods=n, freq="1min", tz="UTC"), name="timestamp"
    )
    o = np.array(opens, dtype=float)
    return pd.DataFrame(
        {"open": o, "high": o + 1.0, "low": o - 1.0, "close": o + 0.5, "volume": 100.0},
        index=idx,
    )


def _order(oid: str, side: str, qty: int) -> Order:
    return Order(client_order_id=oid, instrument="MES", side=side, qty=qty, type="market")


def test_satisfies_execution_client_protocol() -> None:
    client = BacktestExecutionClient(_bars([100.0, 101.0]), MES)
    assert isinstance(client, ExecutionClient)


def test_order_is_automated_by_default() -> None:
    assert _order("x", "buy", 1).is_automated is True


def test_market_fills_at_next_bar_open_under_delay_one() -> None:
    bars = _bars([100.0, 105.0, 110.0, 108.0])
    client = BacktestExecutionClient(bars, MES, delay_bars=1)
    client.advance(bars.index[0])  # observing bar 0 -> fill at bar 1 open (105)
    ack = client.submit(_order("o1", "buy", 2))
    assert ack.accepted
    pos = client.positions()
    assert pos == [Position(instrument="MES", qty=2, avg_px=105.0)]


def test_market_fills_at_current_bar_open_under_delay_zero() -> None:
    bars = _bars([100.0, 105.0, 110.0])
    client = BacktestExecutionClient(bars, MES, delay_bars=0)
    client.advance(bars.index[1])  # fill at bar 1 open (105)
    client.submit(_order("o1", "buy", 1))
    assert client.positions()[0].avg_px == 105.0


def test_round_turn_realizes_pnl_into_balance() -> None:
    bars = _bars([100.0, 100.0, 110.0, 110.0])
    client = BacktestExecutionClient(bars, MES, starting_balance=50_000.0, delay_bars=1)
    client.advance(bars.index[0])
    client.submit(_order("buy", "buy", 1))  # fill bar1 open = 100
    client.advance(bars.index[2])
    client.submit(_order("sell", "sell", 1))  # fill bar3 open = 110
    acct = client.account()
    assert client.positions() == []  # flat
    # realised = (110 - 100) * multiplier(5) * 1 = 50
    assert acct.balance == 50_050.0
    assert acct.equity == 50_050.0


def test_flip_closes_and_reopens_on_the_other_side() -> None:
    bars = _bars([100.0, 100.0, 120.0, 120.0])
    client = BacktestExecutionClient(bars, MES, starting_balance=10_000.0, delay_bars=1)
    client.advance(bars.index[0])
    client.submit(_order("long", "buy", 1))  # long 1 @ 100
    client.advance(bars.index[2])
    client.submit(_order("flip", "sell", 2))  # close long @120 (+100), open short 1 @120
    assert client.positions() == [Position(instrument="MES", qty=-1, avg_px=120.0)]
    assert client.account().balance == 10_000.0 + (120.0 - 100.0) * 5


def test_callbacks_are_noops_in_backtest() -> None:
    client = BacktestExecutionClient(_bars([100.0, 101.0]), MES)
    fired = []
    client.on_disconnect(lambda: fired.append("disc"))
    client.on_data_stale(lambda: fired.append("stale"))
    client.advance(_bars([100.0, 101.0]).index[0])
    client.submit(_order("o", "buy", 1))
    assert fired == []  # never fired in the deterministic backtest


def test_account_returns_account_state_type() -> None:
    client = BacktestExecutionClient(_bars([100.0, 101.0]), MES)
    client.advance(_bars([100.0, 101.0]).index[0])
    assert isinstance(client.account(), AccountState)
