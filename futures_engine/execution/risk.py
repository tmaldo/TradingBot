"""The non-overridable RiskManager and its five kill switches (G13/G15/G16).

The RiskManager **owns the** :class:`~futures_engine.execution.client.ExecutionClient`
(as a private ``_client``; there is no public accessor -- see :mod:`.oms` for how a
strategy is denied any path to it) and is the sole gate every order crosses before
it can leave the system. :meth:`RiskManager.approve` composes five independently
testable, config-driven kill switches; :meth:`RiskManager.execute` is the *only*
forwarder to ``client.submit`` and the OMS calls it strictly after an ``approve``.

Kill switches (thresholds from ``configs/live.yaml`` + the firm's prop preset --
zero magic numbers here, G15):

* ``daily_loss_limit(buffer)`` -- reject + flatten once the day's loss comes within
  ``buffer_usd`` of the firm's daily loss limit.
* ``trailing_dd_guard(margin)`` -- reject once equity comes within ``margin_usd`` of
  the ratcheting trailing-drawdown floor.
* ``stale_data_halt(max_age_s)`` -- no fresh tick within ``max_age_s`` -> latch a halt
  and flatten.
* ``flatten_on_disconnect()`` -- a websocket drop queues a flatten for reconnect and
  raises an alarm; nothing is approved while disconnected.
* ``max_order_rate(n_per_minute)`` -- reject order bursts above the rolling-60s cap.

Plus a sizing cap: order qty may not exceed the current T7
:func:`~futures_engine.sizing.position.position_size` output. Every state
transition (halt, disconnect, reconnect, rejection) is logged.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from futures_engine.core.types import InstrumentSpec
from futures_engine.execution.client import (
    AccountState,
    ExecutionClient,
    Order,
    OrderAck,
    OrderSide,
    Position,
)
from futures_engine.execution.live_config import LiveConfig
from futures_engine.execution.monitor import StalenessClock
from futures_engine.prop.rules import PropRuleSet
from futures_engine.sizing.position import position_size

_log = logging.getLogger("futures_engine.execution.risk")

_RATE_WINDOW_S = 60.0  # the "per minute" in max_order_rate is a rolling 60s window.


@dataclass(frozen=True)
class Approval:
    """The RiskManager's verdict on a single order."""

    ok: bool
    reason: str | None = None


class _TrailingFloor:
    """Ratcheting trailing-drawdown floor mirroring the prop-rule mechanics (T2).

    The HWM only rises; the floor sits ``trailing_dd`` below it and, for firms that
    freeze at the start balance, locks there once reached (Topstep-style).
    """

    def __init__(self, rules: PropRuleSet) -> None:
        self._start = rules.start_balance
        self._dd = rules.trailing_dd
        self._freeze = rules.trailing_freezes_at_start_balance
        self.hwm = rules.start_balance
        self.floor = rules.start_balance - rules.trailing_dd
        self._frozen = False

    def update(self, equity: float) -> None:
        """Ratchet the HWM/floor up to ``equity`` (a no-op once frozen)."""
        if self._frozen or equity <= self.hwm:
            return
        self.hwm = equity
        self.floor = self.hwm - self._dd
        if self._freeze and self.floor >= self._start:
            self.floor = self._start
            self._frozen = True


class RiskManager:
    """Owns the :class:`ExecutionClient`; every order passes :meth:`approve` first.

    Construct with the raw broker/backtest ``client`` (which this manager then owns
    privately), the validated :class:`LiveConfig`, the firm's :class:`PropRuleSet`,
    and the traded :class:`InstrumentSpec`. ``clock`` (monotonic seconds) and an
    optional shared :class:`StalenessClock` are injectable for deterministic tests.
    """

    def __init__(
        self,
        client: ExecutionClient,
        config: LiveConfig,
        prop_rules: PropRuleSet,
        spec: InstrumentSpec,
        *,
        clock: Callable[[], float] = monotonic,
        staleness: StalenessClock | None = None,
    ) -> None:
        # Owned privately. There is deliberately NO public accessor returning this
        # object anywhere in the RiskManager or the OMS (G13, by construction).
        self._client = client
        self._cfg = config
        self._rules = prop_rules
        self._spec = spec
        self._clock = clock
        self._staleness = staleness if staleness is not None else StalenessClock(now=clock)

        self._trail = _TrailingFloor(prop_rules)
        self._day_anchor_equity = config.account.starting_balance
        self._order_times: deque[float] = deque()
        self._connected = True
        self._halted = False
        self._halt_reason: str | None = None
        self._flatten_on_reconnect = False
        self._flatten_seq = itertools.count()

        # Wire the venue callbacks to the kill switches (G14 interface parity).
        client.on_disconnect(self._handle_disconnect)
        client.on_data_stale(self._handle_stale_callback)

    # --- public state ------------------------------------------------------

    @property
    def connected(self) -> bool:
        """Whether the venue connection is currently up."""
        return self._connected

    @property
    def halted(self) -> bool:
        """Whether a kill switch has latched a trading halt."""
        return self._halted

    @property
    def max_qty(self) -> int:
        """The current T7 sizing cap: the maximum approvable order qty."""
        s = self._cfg.risk.sizing
        return position_size(
            s.vol_estimate_usd,
            s.edge.to_edge_stats(),
            self._spec,
            s.to_sizing_config(),
            s.survival_max_contracts,
        )

    # --- market-data / equity feeds ---------------------------------------

    def feed_tick(self, ts: float | None = None) -> None:
        """Record a fresh market-data tick (feeds the stale-data switch)."""
        self._staleness.tick(ts)

    def observe(self, state: AccountState) -> None:
        """Update the trailing-DD floor from a fresh account snapshot."""
        self._trail.update(state.equity)

    def start_new_day(self, anchor_equity: float) -> None:
        """Reset the daily-loss anchor at a session open (end-of-day roll)."""
        self._day_anchor_equity = anchor_equity

    # --- individual kill switches (each independently testable) ------------

    def check_daily_loss(self, state: AccountState) -> str | None:
        """Daily-loss-limit switch: reject within ``buffer_usd`` of the firm limit."""
        limit = self._rules.daily_loss_limit
        if limit is None:
            return None
        buffer = self._cfg.risk.kill_switches.daily_loss_limit.buffer_usd
        loss = self._day_anchor_equity - state.equity
        if loss >= limit - buffer:
            return f"daily loss {loss:.2f} within buffer {buffer:.2f} of limit {limit:.2f}"
        return None

    def check_trailing_dd(self, state: AccountState) -> str | None:
        """Trailing-DD-guard switch: reject within ``margin_usd`` of the floor."""
        self._trail.update(state.equity)
        margin = self._cfg.risk.kill_switches.trailing_dd_guard.margin_usd
        if state.equity <= self._trail.floor + margin:
            return (
                f"trailing-DD guard: equity {state.equity:.2f} within margin "
                f"{margin:.2f} of floor {self._trail.floor:.2f}"
            )
        return None

    def check_stale_data(self, now: float | None = None) -> str | None:
        """Stale-data switch: reject if the last tick is older than ``max_age_s``."""
        max_age = self._cfg.risk.kill_switches.stale_data_halt.max_age_s
        if self._staleness.is_stale(max_age, now):
            age = self._staleness.age(now)
            age_str = "never" if age is None else f"{age:.2f}s"
            return f"stale data: last tick {age_str} old (max {max_age:.2f}s)"
        return None

    def check_order_rate(self, now: float | None = None) -> str | None:
        """Max-order-rate switch: reject once the rolling-60s window is full."""
        t = self._clock() if now is None else now
        self._prune_order_times(t)
        cap = self._cfg.risk.kill_switches.max_order_rate.max_per_minute
        if len(self._order_times) >= cap:
            return f"order rate {len(self._order_times)} >= {cap}/min"
        return None

    def check_disconnect(self) -> str | None:
        """Disconnect switch: reject every order while the venue is disconnected."""
        if not self._connected:
            return "venue disconnected: flatten queued for reconnect"
        return None

    def check_sizing(self, order: Order) -> str | None:
        """Sizing cap: reject qty above the current T7 ``position_size`` output."""
        cap = self.max_qty
        if order.qty > cap:
            return f"order qty {order.qty} exceeds sizing cap {cap}"
        return None

    # --- composed gate -----------------------------------------------------

    def approve(self, order: Order, state: AccountState) -> Approval:
        """Run every kill switch; the first failure rejects (with side effects).

        Order of evaluation: disconnect, an already-latched halt, stale data
        (halt + flatten), order rate, daily loss (halt + flatten), trailing-DD
        guard, then the sizing cap. Data-integrity and account-protection
        switches precede the per-order checks so a compromised session can never
        slip an order through on a technicality.
        """
        reason = self.check_disconnect()
        if reason is not None:
            return self._reject(order, reason)

        if self._halted:
            return self._reject(order, f"halted: {self._halt_reason}")

        reason = self.check_stale_data()
        if reason is not None:
            self._latch_halt(reason, flatten=True)
            return self._reject(order, reason)

        reason = self.check_order_rate()
        if reason is not None:
            return self._reject(order, reason)

        reason = self.check_daily_loss(state)
        if reason is not None:
            self._latch_halt(reason, flatten=True)
            return self._reject(order, reason)

        reason = self.check_trailing_dd(state)
        if reason is not None:
            return self._reject(order, reason)

        reason = self.check_sizing(order)
        if reason is not None:
            return self._reject(order, reason)

        return Approval(ok=True, reason=None)

    # --- the sole forwarder to the client ---------------------------------

    def execute(self, order: Order) -> OrderAck:
        """Forward an already-approved order to the owned client (records rate).

        This is the ONLY method that calls ``client.submit``; the OMS calls it
        strictly after a successful :meth:`approve`.
        """
        self.note_order()
        return self._client.submit(order)

    def note_order(self, now: float | None = None) -> None:
        """Record an accepted submission against the rolling order-rate window."""
        t = self._clock() if now is None else now
        self._order_times.append(t)
        self._prune_order_times(t)

    def cancel(self, client_order_id: str) -> None:
        """Cancel a working order (pass-through to the owned client)."""
        self._client.cancel(client_order_id)

    def positions(self) -> list[Position]:
        """Current open positions (pass-through to the owned client)."""
        return self._client.positions()

    def account(self) -> AccountState:
        """Current account snapshot (pass-through to the owned client)."""
        return self._client.account()

    # --- safety actions ----------------------------------------------------

    def flatten(self) -> list[OrderAck]:
        """Submit offsetting market orders to close every open position.

        This safety path deliberately bypasses :meth:`approve` -- a kill switch
        must always be able to reduce risk. If disconnected, the flatten is
        queued for reconnect instead of sent.
        """
        if not self._connected:
            self._flatten_on_reconnect = True
            _log.warning("flatten requested while disconnected: queued for reconnect")
            return []
        acks: list[OrderAck] = []
        for pos in self._client.positions():
            if pos.qty == 0:
                continue
            side: OrderSide = "sell" if pos.qty > 0 else "buy"
            oid = f"flatten-{pos.instrument}-{next(self._flatten_seq)}"
            order = Order(
                client_order_id=oid,
                instrument=pos.instrument,
                side=side,
                qty=abs(pos.qty),
                type="market",
            )
            _log.warning("flatten: %s %d %s", side, abs(pos.qty), pos.instrument)
            acks.append(self._client.submit(order))
        return acks

    def handle_reconnect(self) -> None:
        """Restore the connection and run any flatten queued during the outage."""
        was_disconnected = not self._connected
        self._connected = True
        if was_disconnected:
            _log.warning("venue reconnected")
        if self._flatten_on_reconnect:
            self._flatten_on_reconnect = False
            self.flatten()

    # --- internals ---------------------------------------------------------

    def _handle_disconnect(self) -> None:
        """WS-drop callback: mark disconnected, queue a flatten, raise the alarm."""
        if self._connected:
            self._connected = False
            self._flatten_on_reconnect = True
            alarm = self._cfg.risk.kill_switches.flatten_on_disconnect.alarm
            _log.warning(
                "venue disconnect: flatten queued for reconnect%s",
                " (ALARM)" if alarm else "",
            )

    def _handle_stale_callback(self) -> None:
        """Venue stale-data callback: latch a halt and flatten."""
        self._latch_halt("stale data (venue callback)", flatten=True)

    def _latch_halt(self, reason: str, *, flatten: bool) -> None:
        """Latch a trading halt (idempotent) and optionally flatten."""
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            _log.warning("HALT latched: %s", reason)
            if flatten:
                self.flatten()

    def _reject(self, order: Order, reason: str) -> Approval:
        _log.info("order %s rejected: %s", order.client_order_id, reason)
        return Approval(ok=False, reason=reason)

    def _prune_order_times(self, now: float) -> None:
        cutoff = now - _RATE_WINDOW_S
        while self._order_times and self._order_times[0] <= cutoff:
            self._order_times.popleft()
