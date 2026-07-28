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
from futures_engine.execution.live_config import LiveConfig
from futures_engine.execution.monitor import LiveMonitor, ShutdownDecision, StalenessClock
from futures_engine.execution.oms import OMS, Outbox
from futures_engine.execution.reconcile import (
    BrokerSnapshot,
    Discrepancy,
    Fill,
    LocalState,
    ReconcileReport,
    reconcile,
)
from futures_engine.execution.risk import Approval, RiskManager

__all__ = [
    "OMS",
    "AccountState",
    "Approval",
    "BacktestExecutionClient",
    "BrokerSnapshot",
    "Discrepancy",
    "ExecutionClient",
    "Fill",
    "LiveConfig",
    "LiveMonitor",
    "LocalState",
    "Order",
    "OrderAck",
    "Outbox",
    "Position",
    "ReconcileReport",
    "RiskManager",
    "ShutdownDecision",
    "StalenessClock",
    "reconcile",
]
