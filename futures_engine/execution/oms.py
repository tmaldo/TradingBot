"""The Order Management System: the only execution handle a strategy receives.

The OMS is the strategy-facing facade (G13). It owns the :class:`RiskManager`
(which in turn owns the :class:`~futures_engine.execution.client.ExecutionClient`)
as a private attribute; it exposes only ``submit`` / ``cancel`` / ``positions`` /
``account`` / ``recover``. No public accessor returns the RiskManager or the raw
client, so a strategy has -- by construction -- no path to bypass the risk gate.

Every :meth:`OMS.submit` routes ``RiskManager.approve`` -> (only on approval)
``RiskManager.execute`` -> ``client.submit``. Orders are deduplicated on
``client_order_id`` and persisted to a JSON outbox *before* they are sent, so a
kill-restart can replay pending orders (:meth:`OMS.recover`) without
double-sending (a re-sent id is a no-op at both the outbox and the venue).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from futures_engine.execution.client import AccountState, Order, OrderAck, OrderType, Position
from futures_engine.execution.risk import RiskManager

_log = logging.getLogger("futures_engine.execution.oms")

_PENDING = "pending"
_SENT = "sent"
_REJECTED = "rejected"


class Outbox:
    """A crash-safe, JSON-backed record of every order and its lifecycle state.

    States: ``pending`` (persisted, not yet confirmed sent), ``sent`` (ack
    recorded), ``rejected`` (risk-declined, never sent). Persisted atomically on
    every mutation so a process kill leaves a consistent, replayable file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: dict[str, dict[str, object]] = {}
        if self._path is not None and self._path.exists():
            self._records = json.loads(self._path.read_text(encoding="utf-8"))

    def status(self, client_order_id: str) -> str | None:
        """The lifecycle state of ``client_order_id``, or ``None`` if unknown."""
        rec = self._records.get(client_order_id)
        return None if rec is None else str(rec["status"])

    def cached_ack(self, client_order_id: str) -> OrderAck:
        """The stored ack for a previously terminal order (idempotent replay)."""
        rec = self._records[client_order_id]
        ack = rec["ack"]
        assert isinstance(ack, dict)
        return OrderAck(**ack)

    def record_pending(self, order: Order) -> None:
        """Persist ``order`` as pending *before* it is sent to the venue."""
        self._records[order.client_order_id] = {"status": _PENDING, "order": asdict(order)}
        self._flush()

    def mark_sent(self, client_order_id: str, ack: OrderAck) -> None:
        """Mark an order sent and store its ack."""
        rec = self._records[client_order_id]
        rec["status"] = _SENT
        rec["ack"] = asdict(ack)
        self._flush()

    def mark_rejected(self, order: Order, ack: OrderAck) -> None:
        """Record a risk-declined order (it was never sent to the venue)."""
        self._records[order.client_order_id] = {
            "status": _REJECTED,
            "order": asdict(order),
            "ack": asdict(ack),
        }
        self._flush()

    def pending_orders(self) -> list[Order]:
        """Every order still in the ``pending`` state (replay set on restart)."""
        out: list[Order] = []
        for rec in self._records.values():
            if rec["status"] == _PENDING:
                out.append(self._order_from(rec["order"]))
        return out

    @staticmethod
    def _order_from(raw: object) -> Order:
        assert isinstance(raw, dict)
        otype: OrderType = raw["type"]
        return Order(
            client_order_id=str(raw["client_order_id"]),
            instrument=str(raw["instrument"]),
            side=raw["side"],
            qty=int(raw["qty"]),
            type=otype,
            limit_px=raw["limit_px"],
            stop_px=raw["stop_px"],
            is_automated=bool(raw["is_automated"]),
        )

    def _flush(self) -> None:
        if self._path is None:
            return
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic on POSIX and Windows


class OMS:
    """Strategy-facing execution facade; owns the RiskManager privately (G13)."""

    def __init__(self, risk_manager: RiskManager, *, outbox_path: Path | None = None) -> None:
        # Private: there is no public accessor to the RiskManager or, through it,
        # the ExecutionClient. This is the by-construction non-overridability.
        self._risk = risk_manager
        self._outbox = Outbox(outbox_path)

    def submit(self, order: Order) -> OrderAck:
        """Route ``order`` through the risk gate and, only on approval, the client.

        Idempotent: a ``client_order_id`` already ``sent``/``rejected`` returns the
        cached ack without re-invoking risk or the venue.
        """
        status = self._outbox.status(order.client_order_id)
        if status in (_SENT, _REJECTED):
            _log.info("duplicate submit %s: no-op (%s)", order.client_order_id, status)
            return self._outbox.cached_ack(order.client_order_id)

        approval = self._risk.approve(order, self._risk.account())
        if not approval.ok:
            ack = OrderAck(order.client_order_id, accepted=False, reason=approval.reason)
            self._outbox.mark_rejected(order, ack)
            return ack

        # Persist BEFORE sending so a crash mid-send leaves a replayable record.
        self._outbox.record_pending(order)
        ack = self._risk.execute(order)
        self._outbox.mark_sent(order.client_order_id, ack)
        return ack

    def recover(self) -> list[OrderAck]:
        """Replay every pending outbox order after a restart (no double-send).

        Orders already ``sent`` are skipped; a venue that deduplicates on
        ``client_order_id`` makes even a re-sent pending order a no-op.
        """
        acks: list[OrderAck] = []
        for order in self._outbox.pending_orders():
            _log.warning("recover: replaying pending order %s", order.client_order_id)
            approval = self._risk.approve(order, self._risk.account())
            if not approval.ok:
                ack = OrderAck(order.client_order_id, accepted=False, reason=approval.reason)
                self._outbox.mark_rejected(order, ack)
                continue
            ack = self._risk.execute(order)
            self._outbox.mark_sent(order.client_order_id, ack)
            acks.append(ack)
        return acks

    def cancel(self, client_order_id: str) -> None:
        """Cancel a working order (pass-through; no client handle is exposed)."""
        self._risk.cancel(client_order_id)

    def positions(self) -> list[Position]:
        """Current open positions (read-only pass-through)."""
        return self._risk.positions()

    def account(self) -> AccountState:
        """Current account snapshot (read-only pass-through)."""
        return self._risk.account()
