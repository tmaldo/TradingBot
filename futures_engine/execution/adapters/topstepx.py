"""TopstepX (ProjectX gateway) execution adapter -- typed + fixture-tested.

Implemented to the extent the public ProjectX gateway docs allow (architect
decision #6): order placement/cancel, open positions and account search. All I/O
goes through an injected :class:`RestTransport`; CI drives it from recorded
fixtures with no network. Every outbound order carries the automated flag (CME
Rule 575); the public ProjectX schema does not expose a dedicated field, so we
attach it as ``isAutomated`` on the payload -- documented best-effort.

References: ProjectX gateway ``/api/Order/place`` and ``/api/Order/cancel``,
``/api/Position/searchOpen`` and ``/api/Account/search``. Order ``type`` enum:
1=Limit, 2=Market, 4=Stop. ``side`` enum: 0=Bid (buy), 1=Ask (sell).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from futures_engine.execution.adapters import AdapterError, RestTransport
from futures_engine.execution.client import AccountState, Order, OrderAck, OrderType, Position
from futures_engine.execution.live_config import TopstepXAdapterConfig

_log = logging.getLogger("futures_engine.execution.adapters.topstepx")

# ProjectX order-type enum keyed by our OrderType.
_ORDER_TYPE: dict[OrderType, int] = {"limit": 1, "market": 2, "stop": 4}
# ProjectX side enum: 0 = Bid (buy), 1 = Ask (sell).
_SIDE: dict[str, int] = {"buy": 0, "sell": 1}


class TopstepXExecutionClient:
    """TopstepX / ProjectX adapter implementing the shared ExecutionClient Protocol."""

    def __init__(self, config: TopstepXAdapterConfig, transport: RestTransport) -> None:
        self._cfg = config
        self._transport = transport
        self._order_ids: dict[str, int] = {}
        self._disconnect_cbs: list[Callable[[], None]] = []
        self._stale_cbs: list[Callable[[], None]] = []

    def serialize_order(self, order: Order) -> dict[str, Any]:
        """Build the ProjectX ``/api/Order/place`` payload (automated flagged)."""
        if not order.is_automated:
            raise AdapterError(
                f"order {order.client_order_id} is not automated; CME Rule 575 "
                "requires the automated flag on every machine-generated order"
            )
        body: dict[str, Any] = {
            "accountId": self._cfg.account_id,
            "contractId": order.instrument,
            "type": _ORDER_TYPE[order.type],
            "side": _SIDE[order.side],
            "size": order.qty,
            "customTag": order.client_order_id,
            # Public ProjectX docs expose no automated flag; we attach one so the
            # CME-575 marking is present and auditable (best-effort, documented).
            "isAutomated": True,
        }
        if order.limit_px is not None:
            body["limitPrice"] = order.limit_px
        if order.stop_px is not None:
            body["stopPrice"] = order.stop_px
        return body

    def submit(self, order: Order) -> OrderAck:
        """Place ``order`` via ``/api/Order/place`` and parse the ProjectX ack."""
        resp = self._transport.post("/api/Order/place", self.serialize_order(order))
        return self._parse_ack(order, resp)

    def cancel(self, client_order_id: str) -> None:
        """Cancel a working order by its recorded ProjectX ``orderId``."""
        order_id = self._order_ids.get(client_order_id)
        if order_id is None:
            _log.info("cancel %s: no known orderId (no-op)", client_order_id)
            return
        self._transport.post(
            "/api/Order/cancel", {"accountId": self._cfg.account_id, "orderId": order_id}
        )

    def positions(self) -> list[Position]:
        """Return open positions from ``/api/Position/searchOpen``."""
        resp = self._transport.post("/api/Position/searchOpen", {"accountId": self._cfg.account_id})
        self._require_success(resp)
        return [self._parse_position(p) for p in resp["positions"] if int(p["size"]) != 0]

    def account(self) -> AccountState:
        """Return the account snapshot from ``/api/Account/search``."""
        resp = self._transport.post("/api/Account/search", {"onlyActiveAccounts": True})
        self._require_success(resp)
        acct = next(a for a in resp["accounts"] if int(a["id"]) == self._cfg.account_id)
        balance = float(acct["balance"])
        return AccountState(balance=balance, equity=balance, positions=self.positions())

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when the venue connection drops."""
        self._disconnect_cbs.append(cb)

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Register a callback fired when market data goes stale."""
        self._stale_cbs.append(cb)

    # --- parsing internals -------------------------------------------------

    def _parse_ack(self, order: Order, resp: dict[str, Any]) -> OrderAck:
        if not resp.get("success", False):
            reason = str(resp.get("errorMessage") or resp.get("errorCode") or "rejected")
            return OrderAck(order.client_order_id, accepted=False, reason=reason)
        order_id = resp.get("orderId")
        if order_id is None:
            raise AdapterError(f"place response missing orderId: {resp!r}")
        self._order_ids[order.client_order_id] = int(order_id)
        return OrderAck(order.client_order_id, accepted=True, reason=None)

    @staticmethod
    def _require_success(resp: dict[str, Any]) -> None:
        if not resp.get("success", False):
            raise AdapterError(f"gateway error: {resp!r}")

    @staticmethod
    def _parse_position(raw: dict[str, Any]) -> Position:
        # ProjectX position type: 1 = Long, 2 = Short. Sign the netted size.
        size = int(raw["size"])
        signed = size if int(raw["type"]) == 1 else -size
        return Position(
            instrument=str(raw["contractId"]),
            qty=signed,
            avg_px=float(raw["averagePrice"]),
        )
