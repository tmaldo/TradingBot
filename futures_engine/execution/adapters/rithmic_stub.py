"""Rithmic execution adapter -- typed stub (interface present, not implemented).

Architect decision #6: Rithmic is a typed stub. The class implements the
:class:`~futures_engine.execution.client.ExecutionClient` Protocol shape so it is
a drop-in *type*, but every method raises :class:`NotImplementedError`. Rithmic's
R|Protocol API (a binary/protobuf gateway, not REST) is out of scope for T8; this
placeholder keeps the seam ready without pretending to trade.
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
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        """Not implemented -- raises :class:`NotImplementedError`."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
