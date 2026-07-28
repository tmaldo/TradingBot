"""Staleness clock (synthetic tick streams with gaps) and shutdown-criteria hooks.

All shutdown thresholds come from ``configs/live.yaml`` -- the monitor holds no
magic numbers (G15). The staleness clock is driven by a synthetic tick stream with
a controlled gap to prove the stale-data detection.
"""

from __future__ import annotations

from futures_engine.execution.live_config import ShutdownConfig
from futures_engine.execution.monitor import LiveMonitor, StalenessClock

from .conftest import ManualClock

# --- staleness clock ---------------------------------------------------------


def test_no_tick_yet_is_stale() -> None:
    clock = ManualClock()
    sc = StalenessClock(now=clock)
    assert sc.age() is None
    assert sc.is_stale(5.0)


def test_fresh_then_stale_across_a_gap() -> None:
    clock = ManualClock()
    sc = StalenessClock(now=clock)
    # a stream of ticks arriving every second keeps the feed fresh.
    for _ in range(5):
        sc.tick()
        clock.advance(1.0)
        assert not sc.is_stale(5.0)
    # then a 6-second gap with no ticks trips staleness (max_age 5.0).
    clock.advance(6.0)
    assert sc.is_stale(5.0)
    assert sc.age() == 7.0  # 1.0 left over from the loop + 6.0 gap


def test_exact_age_boundary_is_not_stale() -> None:
    clock = ManualClock()
    sc = StalenessClock(now=clock)
    sc.tick()
    clock.advance(5.0)
    assert not sc.is_stale(5.0)  # strictly greater than max_age trips
    clock.advance(0.001)
    assert sc.is_stale(5.0)


# --- shutdown criteria (config-driven) --------------------------------------


def _shutdown_cfg(**overrides: object) -> ShutdownConfig:
    base: dict[str, object] = {
        "max_slippage_divergence_usd": 40.0,
        "rolling_sharpe_floor": 0.25,
        "rolling_sharpe_min_samples": 5,
        "rolling_window": 50,
        "drift_threshold_z": 3.0,
    }
    base.update(overrides)
    return ShutdownConfig(**base)  # type: ignore[arg-type]


def test_slippage_divergence_trips_shutdown() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    for _ in range(10):
        monitor.record_fill(live_px=5000.0, backtest_px=4950.0)  # 50 > 40 threshold
    decision = monitor.evaluate()
    assert decision.halt
    assert any("slippage" in r for r in decision.reasons)


def test_slippage_within_threshold_does_not_trip() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    for _ in range(10):
        monitor.record_fill(live_px=5000.0, backtest_px=4980.0)  # 20 < 40
    assert not monitor.evaluate().halt


def test_rolling_sharpe_below_floor_trips_but_only_after_min_samples() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    # a losing, choppy series -> negative Sharpe. Below min samples: no verdict yet.
    monitor.record_trade_return(-10.0)
    monitor.record_trade_return(5.0)
    assert monitor.rolling_sharpe() is None
    assert not monitor.evaluate().halt
    for r in (-10.0, 5.0, -8.0):
        monitor.record_trade_return(r)  # now >= 5 samples
    assert monitor.rolling_sharpe() is not None
    decision = monitor.evaluate()
    assert decision.halt
    assert any("Sharpe" in r for r in decision.reasons)


def test_healthy_sharpe_does_not_trip() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    for _ in range(10):
        monitor.record_trade_return(100.0)
        monitor.record_trade_return(90.0)  # consistently positive, tiny variance
    assert not monitor.evaluate().halt


def test_drift_z_beyond_threshold_trips() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    monitor.record_drift(3.5)  # > 3.0
    decision = monitor.evaluate()
    assert decision.halt
    assert any("drift" in r for r in decision.reasons)


def test_clean_state_is_no_shutdown() -> None:
    monitor = LiveMonitor(_shutdown_cfg())
    assert monitor.evaluate() == monitor.evaluate()
    assert not monitor.evaluate().halt
