"""Live market-data staleness clock and the live-vs-backtest monitoring hooks.

Two responsibilities live here:

* :class:`StalenessClock` -- a monotonic "when did the last tick arrive?" clock.
  The market-data handler calls :meth:`StalenessClock.tick` on every normalized
  tick; the RiskManager's stale-data kill switch reads :meth:`age`. Feeding a
  synthetic tick stream with a controlled gap drives the switch in tests.

* :class:`LiveMonitor` -- rolling live-vs-backtest slippage, realised Sharpe and
  drift statistics, compared against the shutdown criteria in ``configs/live.yaml``
  (:class:`~futures_engine.execution.live_config.ShutdownConfig`). It never sizes
  or trades; it only reports whether an orderly shutdown is warranted. Every
  threshold is config-driven (G15).
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from futures_engine.execution.live_config import ShutdownConfig

_log = logging.getLogger("futures_engine.execution.monitor")


class StalenessClock:
    """Tracks the age of the most recent market-data tick.

    ``now`` defaults to :func:`time.monotonic`; tests inject a deterministic clock.
    Timestamps are seconds on the same monotonic scale as ``now``.
    """

    def __init__(self, *, now: Callable[[], float] = monotonic) -> None:
        self._now = now
        self._last_tick: float | None = None

    def tick(self, ts: float | None = None) -> None:
        """Record a fresh tick at ``ts`` (defaults to the current clock reading)."""
        self._last_tick = self._now() if ts is None else ts

    @property
    def last_tick(self) -> float | None:
        """The timestamp of the last recorded tick, or ``None`` if none yet."""
        return self._last_tick

    def age(self, now: float | None = None) -> float | None:
        """Seconds since the last tick, or ``None`` if no tick has arrived yet."""
        if self._last_tick is None:
            return None
        return (self._now() if now is None else now) - self._last_tick

    def is_stale(self, max_age_s: float, now: float | None = None) -> bool:
        """Whether more than ``max_age_s`` seconds have elapsed since the last tick.

        With no tick yet the feed is treated as stale (we have never had data).
        """
        age = self.age(now)
        if age is None:
            return True
        return age > max_age_s


@dataclass(frozen=True)
class ShutdownDecision:
    """The monitor's verdict: whether to shut down and the breached criteria."""

    halt: bool
    reasons: tuple[str, ...] = ()


@dataclass
class LiveMonitor:
    """Rolling live-vs-backtest diagnostics and shutdown-criteria evaluation.

    Feed it fills (:meth:`record_fill`), realised trade returns
    (:meth:`record_trade_return`) and per-observation drift z-scores
    (:meth:`record_drift`); call :meth:`evaluate` for a :class:`ShutdownDecision`
    against the config's shutdown criteria.
    """

    config: ShutdownConfig
    _slippage: deque[float] = field(init=False)
    _returns: deque[float] = field(init=False)
    _last_drift_z: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        window = self.config.rolling_window
        self._slippage = deque(maxlen=window)
        self._returns = deque(maxlen=window)

    def record_fill(self, live_px: float, backtest_px: float) -> float:
        """Record a fill's live-vs-backtest slippage (USD); returns the abs divergence."""
        divergence = abs(live_px - backtest_px)
        self._slippage.append(divergence)
        return divergence

    def record_trade_return(self, pnl: float) -> None:
        """Record one realised trade P&L for the rolling Sharpe estimate."""
        self._returns.append(pnl)

    def record_drift(self, z_score: float) -> None:
        """Record the latest live-vs-backtest drift z-score."""
        self._last_drift_z = z_score

    def mean_slippage(self) -> float | None:
        """Rolling mean absolute slippage divergence, or ``None`` if no fills yet."""
        if not self._slippage:
            return None
        return statistics.fmean(self._slippage)

    def rolling_sharpe(self) -> float | None:
        """Rolling realised Sharpe, or ``None`` before enough samples / zero variance."""
        n = len(self._returns)
        if n < self.config.rolling_sharpe_min_samples:
            return None
        mean = statistics.fmean(self._returns)
        stdev = statistics.pstdev(self._returns)
        if stdev == 0.0:
            return None
        return mean / stdev

    def evaluate(self) -> ShutdownDecision:
        """Compare rolling statistics to the config's shutdown criteria."""
        reasons: list[str] = []

        mean_slip = self.mean_slippage()
        if mean_slip is not None and mean_slip > self.config.max_slippage_divergence_usd:
            reasons.append(
                f"slippage divergence {mean_slip:.2f} > "
                f"{self.config.max_slippage_divergence_usd:.2f}"
            )

        sharpe = self.rolling_sharpe()
        if sharpe is not None and sharpe < self.config.rolling_sharpe_floor:
            reasons.append(
                f"rolling Sharpe {sharpe:.3f} < floor {self.config.rolling_sharpe_floor:.3f}"
            )

        if abs(self._last_drift_z) > self.config.drift_threshold_z:
            reasons.append(
                f"drift z {self._last_drift_z:.2f} exceeds {self.config.drift_threshold_z:.2f}"
            )

        if reasons:
            _log.warning("shutdown criteria breached: %s", "; ".join(reasons))
        return ShutdownDecision(halt=bool(reasons), reasons=tuple(reasons))
