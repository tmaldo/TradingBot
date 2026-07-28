"""G14 seam lock: the runner's fill path (Nautilus boundaries + T2 price_trades)
and the standalone ``BacktestExecutionClient`` must produce the SAME money and the
SAME bar-open/delay fills on the reference snapshot, for both delay 0 and delay 1.

The two implement the bar-open + ``delay_bars in {0,1}`` + netting convention
independently, so this test fails CI if either drifts. (I2 option (b): the runner
keeps Nautilus as the event sequencer per the binding Path-3 design; option (a)
would have to abandon Nautilus-as-sequencer to route submissions through the
client, so it is rejected as fighting that design -- see the report.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model.identifiers import Venue

from futures_engine.backtest.engine import build_trade_log, price_trades
from futures_engine.backtest.instrument import build_nautilus_instrument
from futures_engine.backtest.strategy_adapter import (
    SIGNAL_REGISTRY,
    bar_timestamps_ns,
    run_event_loop,
)
from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig
from futures_engine.execution import BacktestExecutionClient, Order
from futures_engine.research.harness import causal_positions

_PARAMS = {"window": 20}


def _zero_cost(delay: int) -> CostConfig:
    return CostConfig(
        commission_per_side_usd=0.0,
        exchange_fee_per_side_usd=0.0,
        nfa_fee_per_side_usd=0.0,
        spread_ticks=0.0,
        slippage="fixed_ticks",
        slippage_ticks=0.0,
        delay_bars=delay,
    )


def _flat_tail_targets(bars: pd.DataFrame) -> pd.Series:
    """Causal donchian targets, forced flat over the last 10 bars so both paths
    fully close with no final-open edge (and delay-1 has tail room)."""
    signal = SIGNAL_REGISTRY["donchian_breakout"]()
    held = causal_positions(signal.generate(bars, _PARAMS))
    held.iloc[-10:] = 0.0
    return held


def _drive_client(
    bars: pd.DataFrame, spec: InstrumentSpec, targets_by_ts: dict[pd.Timestamp, int], delay: int
) -> BacktestExecutionClient:
    client = BacktestExecutionClient(bars, spec, starting_balance=100_000.0, delay_bars=delay)
    for ts in bars.index:
        client.advance(ts)
        target = targets_by_ts.get(ts, 0)
        net = client.positions()[0].qty if client.positions() else 0
        delta = target - net
        if delta != 0:
            side = "buy" if delta > 0 else "sell"
            client.submit(Order("o", spec.symbol_root, side, abs(delta), "market"))
    return client


@pytest.mark.parametrize("delay", [0, 1])
def test_runner_and_execution_client_agree(
    bars: pd.DataFrame, mes_spec: InstrumentSpec, delay: int
) -> None:
    held = _flat_tail_targets(bars)
    targets_int = np.rint(held.to_numpy(dtype=float)).astype("int64")
    ts_ns = bar_timestamps_ns(bars)
    targets_ns = {int(ns): int(t) for ns, t in zip(ts_ns, targets_int, strict=True)}
    targets_by_ts = {ts: int(t) for ts, t in zip(bars.index, targets_int, strict=True)}

    # Runner path: Nautilus sequencer -> boundaries -> T2 price_trades (zero costs
    # so net == gross), summed to a single realised-gross number.
    venue = Venue("GLBX")
    instrument = build_nautilus_instrument(mes_spec, venue)
    run = run_event_loop(bars, targets_ns, instrument, venue, starting_balance=100_000.0)
    assert run.final_open is None  # held flat at the tail -> fully closed
    priced = price_trades(build_trade_log(run, bars), bars, mes_spec, _zero_cost(delay))
    runner_gross = float(priced["gross_pnl_usd"].sum())

    # Execution-client path: independent bar-open/delay/netting arithmetic.
    client = _drive_client(bars, mes_spec, targets_by_ts, delay)
    client_realized = client.account().balance - 100_000.0

    assert client.positions() == []  # ends flat -> all PnL realised
    assert runner_gross != 0.0  # a meaningful, non-trivial comparison
    assert np.isclose(runner_gross, client_realized, atol=1e-6, rtol=0.0), (
        f"delay={delay}: runner_gross={runner_gross} client_realized={client_realized}"
    )

    # Every client fill priced at the same delayed bar open the runner's T2 path uses.
    opens = bars["open"].to_numpy(dtype=float)
    pos_of = {ts: i for i, ts in enumerate(bars.index)}
    fills = client.fills()
    assert not fills.empty
    expected = np.array([opens[pos_of[ts] + delay] for ts in fills["signal_ts"]])
    assert np.allclose(fills["fill_px"].to_numpy(dtype=float), expected, atol=0.0)
