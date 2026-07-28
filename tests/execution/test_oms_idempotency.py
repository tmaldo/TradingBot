"""OMS idempotency, non-overridability (G13) and crash-restart replay (G16).

Covers: a strategy has no path to the raw ExecutionClient; every submit routes
through RiskManager.approve; duplicate client_order_ids are no-ops; and a
kill-restart against a persisted outbox replays pending orders without
double-sending.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from futures_engine.execution.client import ExecutionClient, Position
from futures_engine.execution.monitor import StalenessClock
from futures_engine.execution.oms import OMS
from futures_engine.execution.risk import RiskManager

from .conftest import FakeExecutionClient, ManualClock, market_order


def _oms(client: FakeExecutionClient, cfg, rules, spec, path: Path) -> OMS:
    clock = ManualClock()
    rm = RiskManager(client, cfg, rules, spec, clock=clock, staleness=StalenessClock(now=clock))
    rm.feed_tick()
    return OMS(rm, outbox_path=path)


# --- G13 non-overridability by construction ---------------------------------


def test_strategy_cannot_reach_the_execution_client(
    fake_client, live_config, prop_rules, spec, tmp_path
) -> None:
    oms = _oms(fake_client, live_config, prop_rules, spec, tmp_path / "ob.json")
    # No PUBLIC attribute or zero-arg method of the OMS returns an ExecutionClient
    # (or the RiskManager, which owns it). This is the by-construction G13 proof.
    for name in dir(oms):
        if name.startswith("_"):
            continue
        attr = getattr(oms, name)
        assert not isinstance(attr, ExecutionClient), f"OMS.{name} leaks the client"
        assert not isinstance(attr, RiskManager), f"OMS.{name} leaks the RiskManager"
        if callable(attr):
            try:
                result = attr()
            except Exception:
                continue
            assert not isinstance(result, ExecutionClient), f"OMS.{name}() leaks the client"
    # The known handle a strategy holds exposes only submit/cancel/positions/account.
    assert not hasattr(oms, "client")
    assert not hasattr(oms, "risk")


def test_every_submit_routes_through_approve(
    fake_client, live_config, prop_rules, spec, tmp_path, monkeypatch
) -> None:
    oms = _oms(fake_client, live_config, prop_rules, spec, tmp_path / "ob.json")
    calls: list[str] = []
    rm = oms._risk  # test-only introspection; not part of the public surface

    real_approve = rm.approve

    def spy(order, state):  # type: ignore[no-untyped-def]
        calls.append(order.client_order_id)
        return real_approve(order, state)

    monkeypatch.setattr(rm, "approve", spy)
    oms.submit(market_order("a1", "buy", 1))
    assert calls == ["a1"]


def test_rejected_order_never_reaches_the_client(
    fake_client, live_config, prop_rules, spec, tmp_path
) -> None:
    oms = _oms(fake_client, live_config, prop_rules, spec, tmp_path / "ob.json")
    # qty over the sizing cap (3) is rejected -> the client is never called.
    ack = oms.submit(market_order("big", "buy", 99))
    assert not ack.accepted
    assert fake_client.submitted == []


# --- idempotency: duplicate client_order_id is a no-op ----------------------


def test_duplicate_submit_is_a_no_op(fake_client, live_config, prop_rules, spec, tmp_path) -> None:
    oms = _oms(fake_client, live_config, prop_rules, spec, tmp_path / "ob.json")
    ack1 = oms.submit(market_order("dup", "buy", 1))
    ack2 = oms.submit(market_order("dup", "buy", 1))
    assert ack1.accepted and ack2.accepted
    # the client saw the order exactly once.
    assert [o.client_order_id for o in fake_client.submitted] == ["dup"]


# --- crash-restart: replay pending outbox without double-sending ------------


def test_crash_restart_replays_pending_without_double_send(
    live_config, prop_rules, spec, tmp_path
) -> None:
    path = tmp_path / "outbox.json"

    # --- session 1: A sends cleanly; B crashes mid-send (persisted pending) ---
    client1 = FakeExecutionClient()
    oms1 = _oms(client1, live_config, prop_rules, spec, path)
    oms1.submit(market_order("A", "buy", 1))
    assert [o.client_order_id for o in client1.submitted] == ["A"]

    # Simulate a crash *after* the outbox persists B as pending but *before* the
    # ack is recorded: the client raises on B, leaving B pending on disk.
    oms1._risk._client = _RaiseOnB(client1)  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        oms1.submit(market_order("B", "buy", 1))

    # --- session 2 (restart): fresh client, same persisted outbox ---
    client2 = FakeExecutionClient()
    oms2 = _oms(client2, live_config, prop_rules, spec, path)
    replayed = oms2.recover()

    # B (pending) is replayed exactly once; A (already sent) is NOT replayed.
    assert [o.client_order_id for o in client2.submitted] == ["B"]
    assert [a.client_order_id for a in replayed] == ["B"]

    # Re-submitting A after restart is a no-op (idempotent, still marked sent).
    oms2.submit(market_order("A", "buy", 1))
    assert [o.client_order_id for o in client2.submitted] == ["B"]


class _RaiseOnB:
    """Wraps a client so submit() of order 'B' raises (crash simulation)."""

    def __init__(self, inner: FakeExecutionClient) -> None:
        self._inner = inner

    def submit(self, order):  # type: ignore[no-untyped-def]
        if order.client_order_id == "B":
            raise RuntimeError("process killed mid-send")
        return self._inner.submit(order)

    def cancel(self, cid):  # type: ignore[no-untyped-def]
        return self._inner.cancel(cid)

    def positions(self) -> list[Position]:
        return self._inner.positions()

    def account(self):  # type: ignore[no-untyped-def]
        return self._inner.account()

    def on_disconnect(self, cb):  # type: ignore[no-untyped-def]
        return self._inner.on_disconnect(cb)

    def on_data_stale(self, cb):  # type: ignore[no-untyped-def]
        return self._inner.on_data_stale(cb)
