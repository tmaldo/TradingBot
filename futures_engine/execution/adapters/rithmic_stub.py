"""Rithmic execution adapter -- typed stub (interface present, not implemented).

Architect decision #6: Rithmic is a typed stub. The class implements the
:class:`~futures_engine.execution.client.ExecutionClient` Protocol shape so it is
a drop-in *type*, but every trading method raises :class:`NotImplementedError`.
The two callback registrations (``on_disconnect`` / ``on_data_stale``) are the sole
exception -- they are no-ops so the stub can be mounted in a ``RiskManager`` (whose
constructor wires them) without exploding. Rithmic's R|Protocol API (a binary/protobuf
gateway, not REST) is out of scope for T8; this placeholder keeps the seam ready
without pretending to trade.
"""

from __future__ import annotations

from collections.abc import Callable

from futures_engine.execution.client import AccountState, Order, OrderAck, Position

_NOT_IMPLEMENTED = (
    "Rithmic adapter is a typed stub (architect decision #6); the R|Protocol "
    "gateway integration is out of scope for T8."
)


class RithmicExecutionClient:
    """A not-yet-implemented ExecutionClient placeholder for the Rithmic venue."""

    def submit(self, order: Order) -> OrderAck:
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def cancel(self, client_order_id: str) -> None:
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def positions(self) -> list[Position]:
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def account(self) -> AccountState:
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        """Register (and ignore) a disconnect callback -- a deliberate no-op.

        Unlike the other stub methods, this must not raise: ``RiskManager.__init__``
        wires ``client.on_disconnect(...)`` on construction, so raising here would
        make the typed stub un-mountable in a RiskManager. The callback is simply
        dropped (the stub never trades, so it can never fire one)."""

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Register (and ignore) a data-stale callback -- a deliberate no-op.

        No-op for the same reason as :meth:`on_disconnect`: ``RiskManager.__init__``
        calls it on construction, so it must not raise."""
