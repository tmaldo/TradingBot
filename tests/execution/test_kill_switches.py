"""Adversarial, boundary-exact tests for the five kill switches (G13/G16).

Each switch is exercised independently and config-driven: thresholds come from
``configs/live.yaml`` and the Topstep prop preset, never literals in the risk
code. Every switch: below-threshold passes, at/over-threshold rejects (and
flattens/halts where specified), and the state transition is logged.
"""

from __future__ import annotations

import logging

from futures_engine.core.types import InstrumentSpec
from futures_engine.execution.client import AccountState, Position
from futures_engine.execution.live_config import LiveConfig
from futures_engine.execution.monitor import StalenessClock
from futures_engine.execution.risk import RiskManager
from futures_engine.prop.rules import PropRuleSet

from .conftest import FakeExecutionClient, ManualClock, market_order


def _rm(
    client: FakeExecutionClient,
    cfg: LiveConfig,
    rules: PropRuleSet,
    spec: InstrumentSpec,
    clock: ManualClock,
) -> RiskManager:
    return RiskManager(
        client,
        cfg,
        rules,
        spec,
        clock=clock,
        staleness=StalenessClock(now=clock),
    )


def _state(equity: float, positions: list[Position] | None = None) -> AccountState:
    return AccountState(balance=equity, equity=equity, positions=positions or [])


# --- 1. daily loss limit (buffer) -------------------------------------------


def test_daily_loss_below_buffer_is_approved(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    # limit 1000, buffer 150 -> breach threshold at loss >= 850. Loss of 800 is fine.
    approval = rm.approve(market_order("o1", "buy", 1), _state(50_000.0 - 800.0))
    assert approval.ok


def test_daily_loss_at_buffer_boundary_rejects_and_flattens(
    fake_client, live_config, prop_rules, spec, caplog
) -> None:
    clock = ManualClock()
    fake_client.set_account(positions=[Position("MES", 2, 5000.0)])
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    # loss of exactly 850 == limit(1000) - buffer(150): breach.
    with caplog.at_level(logging.WARNING):
        approval = rm.approve(market_order("o2", "buy", 1), _state(50_000.0 - 850.0))
    assert not approval.ok
    assert "daily loss" in (approval.reason or "").lower()
    # flatten fired: an offsetting order for the open long was sent.
    assert any(o.side == "sell" and o.qty == 2 for o in fake_client.submitted)
    assert rm.halted
    assert any("daily loss" in r.message.lower() for r in caplog.records)


# --- 2. trailing-DD guard (margin) ------------------------------------------


def test_trailing_dd_far_from_floor_is_approved(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    # Ratchet the floor to its frozen level (50000) so we can probe the trailing
    # guard at equities that do NOT also trip the daily-loss switch.
    rm.observe(_state(55_000.0))  # topstep freezes floor at start balance -> 50000
    # margin 250 -> reject at equity <= 50250; 51000 is far above and approved.
    assert rm.approve(market_order("o3", "buy", 1), _state(51_000.0)).ok


def test_trailing_dd_within_margin_rejects(
    fake_client, live_config, prop_rules, spec, caplog
) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    rm.observe(_state(55_000.0))  # floor freezes at 50000
    # equity 50200 is within margin (<=50250) but the day's loss (-200) is well
    # short of the daily-loss switch, so the trailing-DD guard is the binding one.
    with caplog.at_level(logging.WARNING):
        approval = rm.approve(market_order("o4", "buy", 1), _state(50_200.0))
    assert not approval.ok
    assert "trailing" in (approval.reason or "").lower()


def test_trailing_dd_floor_ratchets_with_equity(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    # push equity up so the HWM (and floor) ratchet; 55000 -> floor 53000 (freezes? no,
    # freeze only at start balance, floor 53000 > 50000 so it locks at 50000 start).
    rm.observe(_state(55_000.0))
    # topstep freezes floor at start balance once reached -> floor = 50000; margin 250
    # -> reject at equity <= 50250.
    assert not rm.approve(market_order("o5", "buy", 1), _state(50_200.0)).ok
    assert rm.approve(market_order("o6", "buy", 1), _state(50_400.0)).ok


# --- 3. stale-data halt ------------------------------------------------------


def test_fresh_data_is_approved(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    clock.advance(4.0)  # max_age_s = 5.0
    assert rm.approve(market_order("o7", "buy", 1), _state(50_000.0)).ok


def test_stale_data_halts_and_flattens(fake_client, live_config, prop_rules, spec, caplog) -> None:
    clock = ManualClock()
    fake_client.set_account(positions=[Position("MES", -3, 5000.0)])
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    clock.advance(5.01)  # strictly older than max_age_s
    with caplog.at_level(logging.WARNING):
        approval = rm.approve(market_order("o8", "buy", 1), _state(50_000.0))
    assert not approval.ok
    assert "stale" in (approval.reason or "").lower()
    # flatten covered the short (a buy of 3).
    assert any(o.side == "buy" and o.qty == 3 for o in fake_client.submitted)
    assert rm.halted


def test_stale_check_boundary_exact_age_is_fresh(
    fake_client, live_config, prop_rules, spec
) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    clock.advance(5.0)  # exactly max_age is NOT stale (strictly greater trips)
    assert rm.check_stale_data() is None


# --- 4. flatten on disconnect ------------------------------------------------


def test_disconnect_queues_flatten_and_alarms(
    fake_client, live_config, prop_rules, spec, caplog
) -> None:
    clock = ManualClock()
    fake_client.set_account(positions=[Position("MES", 2, 5000.0)])
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    with caplog.at_level(logging.WARNING):
        fake_client.fire_disconnect()  # simulated WS drop
    assert not rm.connected
    # while disconnected, no new orders are approved.
    assert not rm.approve(market_order("o9", "buy", 1), _state(50_000.0)).ok
    # flatten is queued, not sent (we cannot reach the broker while disconnected).
    assert fake_client.submitted == []
    assert any("disconnect" in r.message.lower() for r in caplog.records)
    # on reconnect the queued flatten is sent.
    rm.handle_reconnect()
    assert any(o.side == "sell" and o.qty == 2 for o in fake_client.submitted)
    assert rm.connected


# --- 5. max order rate -------------------------------------------------------


def test_order_rate_burst_rejected(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    # max_per_minute = 10; submit 10 within the window, the 11th is rejected.
    for i in range(10):
        ack = rm.approve(market_order(f"r{i}", "buy", 1), _state(50_000.0))
        assert ack.ok
        rm.note_order()  # OMS records each accepted submission
    assert not rm.approve(market_order("r10", "buy", 1), _state(50_000.0)).ok


def test_order_rate_window_slides(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    for _ in range(10):
        rm.note_order()
    assert rm.check_order_rate() is not None
    clock.advance(61.0)  # all prior orders age out of the 60s window
    assert rm.check_order_rate() is None


# --- 6. sizing cap (qty <= position_size) -----------------------------------


def test_qty_within_size_cap_is_approved(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    assert rm.approve(market_order("s1", "buy", rm.max_qty), _state(50_000.0)).ok


def test_qty_over_size_cap_rejected(fake_client, live_config, prop_rules, spec) -> None:
    clock = ManualClock()
    rm = _rm(fake_client, live_config, prop_rules, spec, clock)
    rm.feed_tick()
    assert not rm.approve(market_order("s2", "buy", rm.max_qty + 1), _state(50_000.0)).ok
