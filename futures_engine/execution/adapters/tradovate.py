"""Tradovate execution adapter -- FULL implementation (MFFU route, demo endpoints).

Implements the :class:`~futures_engine.execution.client.ExecutionClient` Protocol
against Tradovate's REST + WebSocket API shapes. All I/O goes through an injected
:class:`~futures_engine.execution.adapters.RestTransport`, so the adapter is
exercised end-to-end from recorded fixtures with no network (G15). Order
serialization always sets ``isAutomated=True`` (CME Rule 575).

References: Tradovate ``order/placeorder`` and ``order/cancelorder`` REST
endpoints, ``position/list`` and ``cashBalance/getcashbalance``, and the
market-data WebSocket ``md/subscribequote`` quote frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from futures_engine.execution.adapters import AdapterError, MarketTick, RestTransport
from futures_engine.execution.client import AccountState, Order, OrderAck, OrderType, Position
from futures_engine.execution.live_config import TradovateAdapterConfig

_log = logging.getLogger("futures_engine.execution.adapters.tradovate")

# Tradovate order-type names keyed by our OrderType.
_ORDER_TYPE: dict[OrderType, str] = {"market": "Market", "limit": "Limit", "stop": "Stop"}


class TradovateExecutionClient:
    """Tradovate adapter implementing the shared ExecutionClient Protocol."""

    def __init__(self, config: TradovateAdapterConfig, transport: RestTransport) -> None:
        self._cfg = config
        self._transport = transport
        self._order_ids: dict[str, int] = {}  # client_order_id -> venue orderId
        self._disconnect_cbs: list[Callable[[], None]] = []
        self._stale_cbs: list[Callable[[], None]] = []

    # --- serialization (pure; asserted in tests) ---------------------------

    def serialize_order(self, order: Order) -> dict[str, Any]:
        """Build the Tradovate ``placeorder`` payload for ``order`` (isAutomated=True)."""
        if not order.is_automated:
            raise AdapterError(
                f"order {order.client_order_id} is not automated; CME Rule 575 "
                "requires every machine-generated order be flagged isAutomated"
            )
        body: dict[str, Any] = {
            "accountSpec": self._cfg.account_spec,
            "accountId": self._cfg.account_id,
            "clOrdId": order.client_order_id,
            "action": "Buy" if order.side == "buy" else "Sell",
            "symbol": order.instrument,
            "orderQty": order.qty,
            "orderType": _ORDER_TYPE[order.type],
            "isAutomated": True,  # CME Rule 575 -- always true in this system.
        }
        if order.limit_px is not None:
            body["price"] = order.limit_px
        if order.stop_px is not None:
            body["stopPrice"] = order.stop_px
        return body

    # --- ExecutionClient ---------------------------------------------------

    def submit(self, order: Order) -> OrderAck:
        """Place ``order`` and return an ack (parsing Tradovate's response shape)."""
        resp = self._transport.post("/order/placeorder", self.serialize_order(order))
        return self._parse_ack(order, resp)

    def cancel(self, client_order_id: str) -> None:
        """Cancel a working order by its recorded venue ``orderId`` (no-op if unknown)."""
        order_id = self._order_ids.get(client_order_id)
        if order_id is None:
            _log.info("cancel %s: no known venue orderId (no-op)", client_order_id)
            return
        self._transport.post("/order/cancelorder", {"orderId": order_id})

    def positions(self) -> list[Position]:
        """Return open positions parsed from ``position/list``."""
        resp = self._transport.get("/position/list")
        items = resp.get("positions", resp) if isinstance(resp, dict) else resp
        return [self._parse_position(p) for p in items if int(p["netPos"]) != 0]

    def account(self) -> AccountState:
        """Return the account snapshot parsed from ``cashBalance/getcashbalance``."""
        resp = self._transport.get("/cashBalance/getcashbalance")
        balance = float(resp["cashBalance"])
        equity = float(resp.get("totalPnL", 0.0)) + balance
        return AccountState(balance=balance, equity=equity, positions=self.positions())

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when the venue WebSocket drops."""
        self._disconnect_cbs.append(cb)

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when the market-data feed goes stale."""
        self._stale_cbs.append(cb)

    # --- websocket normalization -------------------------------------------

    def normalize_quote(self, frame: dict[str, Any]) -> MarketTick:
        """Normalize a Tradovate md quote frame into a :class:`MarketTick`.

        Tradovate quote entries carry an ``id`` (contract), ``timestamp`` and an
        ``entries.Trade.price`` last-trade price.
        """
        try:
            entries = frame["entries"]
            price = float(entries["Trade"]["price"])
            instrument = str(frame["contract"])
            ts = float(frame["ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"unparseable quote frame: {frame!r}") from exc
        return MarketTick(instrument=instrument, price=price, ts=ts)

    def fire_disconnect(self) -> None:
        """Invoke registered disconnect callbacks (called by the WS layer)."""
        for cb in self._disconnect_cbs:
            cb()

    # --- parsing internals -------------------------------------------------

    def _parse_ack(self, order: Order, resp: dict[str, Any]) -> OrderAck:
        if "failureReason" in resp or "failureText" in resp:
            reason = str(resp.get("failureText") or resp.get("failureReason"))
            return OrderAck(order.client_order_id, accepted=False, reason=reason)
        order_id = resp.get("orderId")
        if order_id is None:
            raise AdapterError(f"placeorder response missing orderId: {resp!r}")
        self._order_ids[order.client_order_id] = int(order_id)
        return OrderAck(order.client_order_id, accepted=True, reason=None)

    @staticmethod
    def _parse_position(raw: dict[str, Any]) -> Position:
        return Position(
            instrument=str(raw["symbol"]),
            qty=int(raw["netPos"]),
            avg_px=float(raw["netPrice"]),
        )
