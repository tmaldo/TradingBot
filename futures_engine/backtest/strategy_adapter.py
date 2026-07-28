"""Bridge a vectorized :class:`VectorSignal` into the Nautilus event loop.

The runner pre-computes a *held target-position* series (the exact same causal
series the T5 vectorized path uses: :func:`research.harness.causal_positions` of
the raw signal, scaled to whole contracts). :class:`PositionSeriesStrategy` then
walks that series bar-by-bar inside Nautilus, submitting market orders to move the
netted position onto each bar's target. Nautilus is thus the genuine order/position
**state machine**; the ``PositionClosed`` events it emits carry the round-turn
boundaries (``ts_opened`` / ``ts_closed`` land on the bars where the target
changed -- identical to :func:`research.harness.positions_to_trades`).

Fill *prices* from Nautilus are deliberately ignored downstream (Nautilus fills a
market order at the current bar's close under ``bar_execution``; our binding
convention is the next bar's open). They are captured here only so the runner can
run a gross-reconciliation cross-check. All reported prices come from T2's
:func:`~futures_engine.costs.model.delayed_fill_prices` (G8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FeeModel, FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig as NautilusStrategyConfig

from futures_engine.core.types import Bars
from futures_engine.research.strategies.momentum import DonchianBreakout, MACrossVolTarget

# Registry of reference signal families the runner can resolve from config (G15:
# no signal object is smuggled in; a run names its signal by a stable key).
SIGNAL_REGISTRY: dict[str, type] = {
    "donchian_breakout": DonchianBreakout,
    "ma_cross_vol_target": MACrossVolTarget,
}


class ZeroFeeModel(FeeModel):  # type: ignore[misc]
    """A ``FeeModel`` that charges nothing, so Nautilus reports pure gross PnL.

    T2's ``apply_costs`` is the single source of truth for every friction (G8);
    injecting fees here as well would double-count. Zeroing this seam makes that
    structurally impossible and lets the runner cross-check Nautilus gross against
    T2 ``gross_pnl_usd``.
    """

    def get_commission(self, order: Any, fill_qty: Any, fill_px: Any, instrument: Any) -> Money:
        return Money(0.0, USD)


@dataclass
class ClosedTrade:
    """One round-turn closed position captured from a ``PositionClosed`` event.

    Timestamps are stored as raw UTC-nanosecond ints (Nautilus event fields); the
    runner maps them back to the canonical ``bars.index`` entries by position, so
    a pandas resolution mismatch can never break the later fill-price lookup.
    """

    entry_ts_ns: int
    exit_ts_ns: int
    side: str  # "long" | "short"
    qty: float
    nautilus_pnl: float  # Nautilus realised PnL (fee=0 -> gross), for reconciliation
    entry_px_close: float  # Nautilus avg entry price (a bar CLOSE, not reported)
    exit_px_close: float  # Nautilus avg exit price (a bar CLOSE, not reported)


@dataclass
class NautilusRun:
    """The captured outcome of one Nautilus event-loop pass."""

    closed: list[ClosedTrade] = field(default_factory=list)
    final_open: tuple[int, str, float] | None = None  # (entry_ts_ns, side, qty)


class _PositionConfig(NautilusStrategyConfig, frozen=True):  # type: ignore[misc,call-arg]
    instrument_id: InstrumentId
    bar_type: str


class PositionSeriesStrategy(Strategy):  # type: ignore[misc]
    """Drive the netted position onto a precomputed per-bar target series."""

    def __init__(self, config: _PositionConfig, targets: dict[int, int], run: NautilusRun) -> None:
        super().__init__(config)
        self._targets = targets
        self._run = run
        self._open: tuple[int, str, float] | None = None

    def on_start(self) -> None:
        self.subscribe_bars(BarType.from_str(self.config.bar_type))

    def on_bar(self, bar: Bar) -> None:
        target = self._targets.get(bar.ts_event, 0)
        current = int(self.portfolio.net_position(self.config.instrument_id))
        delta = target - current
        if delta == 0:
            return
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=Quantity.from_int(abs(delta)),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def on_position_opened(self, event: Any) -> None:
        self._open = (
            int(event.ts_opened),
            "long" if event.entry == OrderSide.BUY else "short",
            float(event.peak_qty),
        )

    def on_position_closed(self, event: Any) -> None:
        self._run.closed.append(
            ClosedTrade(
                entry_ts_ns=int(event.ts_opened),
                exit_ts_ns=int(event.ts_closed),
                side="long" if event.entry == OrderSide.BUY else "short",
                qty=float(event.peak_qty),
                nautilus_pnl=float(event.realized_pnl.as_double()),
                entry_px_close=float(event.avg_px_open),
                exit_px_close=float(event.avg_px_close),
            )
        )
        self._open = None

    def on_stop(self) -> None:
        if int(self.portfolio.net_position(self.config.instrument_id)) != 0:
            self._run.final_open = self._open


def bar_timestamps_ns(bars: Bars) -> np.ndarray:
    """UTC-nanosecond ``int64`` epoch of each bar (resolution-stable on pandas 3.0).

    pandas 3.0's default datetime resolution is microseconds, so ``asi8`` /
    ``view`` are resolution-dependent; normalising to ``datetime64[ns]`` first
    gives the fixed nanosecond epoch Nautilus expects and that the runner uses to
    key the target series.
    """
    index = cast("pd.DatetimeIndex", bars.index)
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    return naive.to_numpy(dtype="datetime64[ns]").astype("int64")


def _bar_type(instrument: FuturesContract) -> BarType:
    # The declared aggregation is nominal: Nautilus sequences purely by ts_event,
    # so a fixed 1-MINUTE LAST external bar type is used for any bar interval.
    return BarType(
        instrument.id,
        BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )


def run_event_loop(
    bars: Bars,
    targets: dict[int, int],
    instrument: FuturesContract,
    venue: Venue,
    *,
    starting_balance: float,
) -> NautilusRun:
    """Run one Nautilus pass over ``bars`` driving positions onto ``targets``.

    ``targets`` maps a bar's UTC-nanosecond ``ts_event`` to the integer contract
    position to hold from that bar on. Returns the captured closed round-turns and
    any still-open final position. Nautilus logging is fully bypassed so trial
    output stays clean.
    """
    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BT-001"),
            logging=LoggingConfig(bypass_logging=True),
        )
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(starting_balance, USD)],
        fee_model=ZeroFeeModel(),
        fill_model=FillModel(prob_slippage=0.0),
        bar_execution=True,
    )
    engine.add_instrument(instrument)

    bar_type = _bar_type(instrument)
    precision = instrument.price_precision
    ts_ns = bar_timestamps_ns(bars)
    nautilus_bars = [
        Bar(
            bar_type,
            Price(row.open, precision),
            Price(row.high, precision),
            Price(row.low, precision),
            Price(row.close, precision),
            Quantity(row.volume, 0),
            int(ts),
            int(ts),
        )
        for ts, row in zip(ts_ns, bars.itertuples(index=False), strict=True)
    ]
    engine.add_data(nautilus_bars)

    run = NautilusRun()
    strategy = PositionSeriesStrategy(
        _PositionConfig(instrument_id=instrument.id, bar_type=str(bar_type)),
        targets,
        run,
    )
    engine.add_strategy(strategy)
    engine.run()
    engine.dispose()
    return run
