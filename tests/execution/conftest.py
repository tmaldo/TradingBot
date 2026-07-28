"""Shared fixtures for the execution-layer tests: a controllable fake client,
a deterministic clock, and loaders for the live config / prop rules.

All offline: no network, no real broker. The :class:`FakeExecutionClient`
implements the :class:`~futures_engine.execution.client.ExecutionClient` Protocol
so it is a drop-in for both the OMS/RiskManager tests and reconciliation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from futures_engine.core.types import InstrumentSpec
from futures_engine.execution.client import (
    AccountState,
    ExecutionClient,
    Order,
    OrderAck,
    Position,
)
from futures_engine.execution.live_config import LiveConfig
from futures_engine.prop.rules import PropRuleSet

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)


class FakeExecutionClient:
    """An in-memory :class:`ExecutionClient` that records submits and is fully
    controllable (positions/account set by the test). Dedupes on
    ``client_order_id`` the way a real venue does with a client-supplied id."""

    def __init__(
        self,
        *,
        balance: float = 50_000.0,
        equity: float = 50_000.0,
        positions: list[Position] | None = None,
        reject_ids: set[str] | None = None,
    ) -> None:
        self._balance = balance
        self._equity = equity
        self._positions = list(positions or [])
        self._reject_ids = reject_ids or set()
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []
        self._seen: set[str] = set()
        self._disconnect_cbs: list[Callable[[], None]] = []
        self._stale_cbs: list[Callable[[], None]] = []

    # --- ExecutionClient ---------------------------------------------------
    def submit(self, order: Order) -> OrderAck:
        if order.client_order_id in self._seen:
            # venue-side idempotency: a re-sent id is a no-op ack.
            return OrderAck(order.client_order_id, accepted=True, reason="duplicate")
        self._seen.add(order.client_order_id)
        self.submitted.append(order)
        if order.client_order_id in self._reject_ids:
            return OrderAck(order.client_order_id, accepted=False, reason="venue reject")
        return OrderAck(order.client_order_id, accepted=True, reason=None)

    def cancel(self, client_order_id: str) -> None:
        self.cancelled.append(client_order_id)

    def positions(self) -> list[Position]:
        return list(self._positions)

    def account(self) -> AccountState:
        return AccountState(
            balance=self._balance, equity=self._equity, positions=list(self._positions)
        )

    def on_disconnect(self, cb: Callable[[], None]) -> None:
        self._disconnect_cbs.append(cb)

    def on_data_stale(self, cb: Callable[[], None]) -> None:
        self._stale_cbs.append(cb)

    # --- test controls -----------------------------------------------------
    def set_account(
        self,
        *,
        balance: float | None = None,
        equity: float | None = None,
        positions: list[Position] | None = None,
    ) -> None:
        if balance is not None:
            self._balance = balance
        if equity is not None:
            self._equity = equity
        if positions is not None:
            self._positions = list(positions)

    def fire_disconnect(self) -> None:
        for cb in self._disconnect_cbs:
            cb()


class ManualClock:
    """A monotonic clock whose value the test advances explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def spec() -> InstrumentSpec:
    return MES


@pytest.fixture
def live_config() -> LiveConfig:
    return LiveConfig.load(CONFIGS_DIR / "live.yaml")


@pytest.fixture
def prop_rules() -> PropRuleSet:
    # Topstep 50K: $1,000 daily loss limit, $2,000 EOD trailing DD freezing at start.
    return PropRuleSet(
        name="topstep_50k",
        start_balance=50_000.0,
        trailing_dd=2_000.0,
        trailing_mode="eod",
        trailing_freezes_at_start_balance=True,
        daily_loss_limit=1_000.0,
        consistency_max_day_pct=0.50,
        profit_target=3_000.0,
        min_trading_days=0,
    )


@pytest.fixture
def fake_client() -> FakeExecutionClient:
    return FakeExecutionClient()


def market_order(oid: str, side: str, qty: int, instrument: str = "MES") -> Order:
    """Helper: a market order (is_automated defaults True)."""
    from typing import cast

    from futures_engine.execution.client import OrderSide

    return Order(
        client_order_id=oid,
        instrument=instrument,
        side=cast(OrderSide, side),
        qty=qty,
        type="market",
    )


def _protocol_check() -> None:
    # Static/runtime assurance the fake honours the Protocol.
    assert isinstance(FakeExecutionClient(), ExecutionClient)
