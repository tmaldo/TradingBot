"""The broker-agnostic execution seam (Global Constraints G13/G14).

This module defines the value objects and the :class:`ExecutionClient` Protocol
that *every* execution venue -- the backtest simulation here, and the live
Tradovate / TopstepX adapters that task T8 will add -- implements. A strategy
holds an ``ExecutionClient``, never a concrete broker client, so the exact same
strategy code drives a backtest and a live account (G14 parity).

Fill causality (backtest)
-------------------------
:class:`BacktestExecutionClient` fills market orders at a **bar open**, honouring
the project's binding delay convention (the same one T2's
:func:`~futures_engine.costs.model.delayed_fill_prices` encodes): an order
submitted while the client's clock is on bar ``t`` fills at the open of bar
``t + delay_bars`` (``delay_bars`` in ``{0, 1}``). Positions are netted per
instrument and the account's realised balance / mark-to-market equity are tracked
bar by bar. The ``on_disconnect`` / ``on_data_stale`` callbacks exist so the live
adapter shares the interface; in the deterministic backtest they never fire, so
registering one is a no-op that simply stores the callback.

Costs are **not** applied here -- gross fills only. The single source of truth for
commissions, fees, spread and slippage remains T2's ``apply_costs`` (G8), applied
once, downstream, by the backtest runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from futures_engine.core.types import Bars, InstrumentSpec

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]


@dataclass(frozen=True)
class Order:
    """A single order routed through an :class:`ExecutionClient`.

    ``is_automated`` defaults to ``True`` and is always ``True`` in this system:
    every order is machine-generated, which CME Rule 575 requires be flagged.
    """

    client_order_id: str
    instrument: str
    side: OrderSide
    qty: int
    type: OrderType
    limit_px: float | None = None
    stop_px: float | None = None
    is_automated: bool = True


@dataclass(frozen=True)
class OrderAck:
    """Acknowledgement returned by :meth:`ExecutionClient.submit`."""

    client_order_id: str
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class Position:
    """A netted position in one instrument (``qty`` signed: + long, - short)."""

    instrument: str
    qty: int
    avg_px: float


@dataclass(frozen=True)
class AccountState:
    """A snapshot of the account: realised ``balance`` and mark-to-market ``equity``."""

    balance: float
    equity: float
    positions: list[Position]


@runtime_checkable
class ExecutionClient(Protocol):
    """Broker-agnostic execution interface shared by backtest and live (G13/G14).

    Task T8's Tradovate / TopstepX adapters implement this identically so a
    strategy is portable across a backtest and a live account with no code change.
    """

    def submit(self, order: Order) -> OrderAck:
        """Submit ``order`` and return an acknowledgement."""
        ...

    def cancel(self, client_order_id: str) -> None:
        """Cancel a working order by id (no-op if already filled/unknown)."""
        ...

    def positions(self) -> list[Position]:
        """Return the current open positions."""
        ...

    def account(self) -> AccountState:
        """Return the current account snapshot."""
        ...

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when the venue connection drops."""
        ...

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when market data goes stale."""
        ...


@dataclass
class _NetPosition:
    """Mutable per-instrument netted position (internal)."""

    qty: int = 0  # signed contracts
    avg_px: float = 0.0  # volume-weighted average entry price of the open lot


class BacktestExecutionClient:
    """A deterministic, bar-open fill simulation implementing :class:`ExecutionClient`.

    Construct with the snapshot ``bars`` and the instrument ``spec``; drive the
    simulation clock with :meth:`advance` (call it once per bar as the event loop
    steps forward) and submit orders via :meth:`submit`. Market orders fill at the
    open of bar ``clock + delay_bars``; limit / stop orders fill at their trigger
    price only if the fill bar's ``[low, high]`` range trades through it, otherwise
    they are rejected (this backtest client does not keep a resting book -- the
    live adapter does). Positions net per instrument; realised PnL accrues to
    ``balance`` and ``equity`` marks the open lot to the current bar's close.
    """

    def __init__(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        *,
        starting_balance: float = 100_000.0,
        delay_bars: int = 1,
    ) -> None:
        if delay_bars not in (0, 1):
            raise ValueError(f"delay_bars must be 0 or 1, got {delay_bars}")
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError("bars must have a DatetimeIndex")
        if not bars.index.is_unique:
            raise ValueError("bars index must be unique")
        self._bars = bars
        self._spec = spec
        self._delay_bars = delay_bars
        self._balance = float(starting_balance)
        self._starting_balance = float(starting_balance)
        self._pos = _NetPosition()
        self._clock_pos = -1  # advanced onto bar 0 before the first submit
        self._pos_of: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(bars.index)}
        self._opens = bars["open"].to_numpy(dtype=float)
        self._highs = bars["high"].to_numpy(dtype=float)
        self._lows = bars["low"].to_numpy(dtype=float)
        self._closes = bars["close"].to_numpy(dtype=float)
        self._disconnect_cbs: list[Callable[[], None]] = []
        self._stale_cbs: list[Callable[[], None]] = []
        # Fill log: (signal-bar timestamp, fill price, signed filled qty) for every
        # accepted fill. The signal bar is the clock bar; the price is the bar the
        # fill actually landed on (clock + delay). Used to lock this client's
        # bar-open/delay/netting arithmetic against the runner's path (G14).
        self._fills: list[tuple[pd.Timestamp, float, int]] = []

    # --- clock ---------------------------------------------------------------

    def advance(self, ts: pd.Timestamp) -> None:
        """Move the simulation clock onto the bar at ``ts``."""
        if ts not in self._pos_of:
            raise ValueError(f"timestamp {ts!r} not in bars index")
        self._clock_pos = self._pos_of[ts]

    # --- ExecutionClient -----------------------------------------------------

    def submit(self, order: Order) -> OrderAck:
        """Fill ``order`` against the appropriate bar open (see class docstring)."""
        if self._clock_pos < 0:
            return OrderAck(order.client_order_id, accepted=False, reason="clock not started")
        if order.qty <= 0:
            return OrderAck(order.client_order_id, accepted=False, reason="qty must be positive")

        fill_pos = self._clock_pos + self._delay_bars
        if fill_pos >= len(self._opens):
            return OrderAck(order.client_order_id, accepted=False, reason="fill bar out of range")

        fill_px = self._fill_price(order, fill_pos)
        if fill_px is None:
            return OrderAck(
                order.client_order_id, accepted=False, reason="order not marketable in fill bar"
            )

        signed = order.qty if order.side == "buy" else -order.qty
        self._apply_fill(signed, fill_px)
        self._fills.append((self._bars.index[self._clock_pos], fill_px, signed))
        return OrderAck(order.client_order_id, accepted=True, reason=None)

    def cancel(self, client_order_id: str) -> None:
        """No-op: this backtest client fills or rejects synchronously (no resting book)."""
        return None

    def positions(self) -> list[Position]:
        """Return the open netted position(s); empty when flat."""
        if self._pos.qty == 0:
            return []
        return [Position(self._spec.symbol_root, self._pos.qty, self._pos.avg_px)]

    def account(self) -> AccountState:
        """Return realised ``balance`` and mark-to-market ``equity`` at the clock bar."""
        return AccountState(
            balance=self._balance, equity=self._equity(), positions=self.positions()
        )

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        """Store ``cb`` (never fired in the deterministic backtest)."""
        self._disconnect_cbs.append(cb)

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Store ``cb`` (never fired in the deterministic backtest)."""
        self._stale_cbs.append(cb)

    def fills(self) -> pd.DataFrame:
        """The fill log as a frame: ``signal_ts`` (clock bar), ``fill_px``, ``qty`` (signed)."""
        return pd.DataFrame(self._fills, columns=["signal_ts", "fill_px", "qty"])

    # --- internals -----------------------------------------------------------

    def _fill_price(self, order: Order, fill_pos: int) -> float | None:
        if order.type == "market":
            return float(self._opens[fill_pos])
        low, high = float(self._lows[fill_pos]), float(self._highs[fill_pos])
        if order.type == "limit":
            if order.limit_px is None:
                raise ValueError("limit order requires limit_px")
            crossed = order.limit_px >= low if order.side == "buy" else order.limit_px <= high
            return order.limit_px if crossed else None
        # stop
        if order.stop_px is None:
            raise ValueError("stop order requires stop_px")
        crossed = order.stop_px <= high if order.side == "buy" else order.stop_px >= low
        return order.stop_px if crossed else None

    def _apply_fill(self, signed_qty: int, fill_px: float) -> None:
        """Net ``signed_qty`` into the position, realising PnL on any reduction."""
        mult = self._spec.multiplier
        cur = self._pos.qty
        if cur == 0 or (cur > 0) == (signed_qty > 0):
            # opening or adding in the same direction: volume-weighted avg price
            new_qty = cur + signed_qty
            self._pos.avg_px = (abs(cur) * self._pos.avg_px + abs(signed_qty) * fill_px) / abs(
                new_qty
            )
            self._pos.qty = new_qty
            return
        # reducing / closing / flipping
        closing = min(abs(signed_qty), abs(cur))
        direction = 1.0 if cur > 0 else -1.0
        self._balance += direction * (fill_px - self._pos.avg_px) * mult * closing
        remaining = abs(signed_qty) - abs(cur)
        if remaining > 0:  # flip: open the residual on the new side at fill_px
            self._pos.qty = int(remaining) * (1 if signed_qty > 0 else -1)
            self._pos.avg_px = fill_px
        else:
            self._pos.qty = cur + signed_qty
            if self._pos.qty == 0:
                self._pos.avg_px = 0.0

    def _equity(self) -> float:
        if self._pos.qty == 0:
            return self._balance
        mark = float(self._closes[self._clock_pos])
        direction = 1.0 if self._pos.qty > 0 else -1.0
        unrealized = (
            direction * (mark - self._pos.avg_px) * self._spec.multiplier * abs(self._pos.qty)
        )
        return self._balance + unrealized
