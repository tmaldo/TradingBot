"""Execution layer: the broker-agnostic ``ExecutionClient`` seam (G13/G14).

Strategies never hold a broker client directly; they act through the
:class:`~futures_engine.execution.client.ExecutionClient` Protocol. The backtest
implementation (:class:`~futures_engine.execution.client.BacktestExecutionClient`)
and the live adapters that task T8 will add (Tradovate / TopstepX) implement the
*same* Protocol, so the backtest and live code paths are shared (G14 parity).
"""

from __future__ import annotations

from futures_engine.execution.client import (
    AccountState,
    BacktestExecutionClient,
    ExecutionClient,
    Order,
    OrderAck,
    Position,
)

__all__ = [
    "AccountState",
    "BacktestExecutionClient",
    "ExecutionClient",
    "Order",
    "OrderAck",
    "Position",
]
