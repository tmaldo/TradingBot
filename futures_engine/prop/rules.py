"""Prop-firm evaluation survival rules (Global Constraint G11).

The survival simulator (T7) Monte-Carlos account equity paths against a
:class:`PropRuleSet` and needs a deterministic, well-defined answer to "did this path
survive, and did it pass?". :func:`simulate_account` is that oracle. All rule *numbers*
live in ``configs/prop_rules.yaml`` (G15); this module holds only the mechanics.

Day model
---------
A path is a sequence of :class:`DayPnL`. Each day carries the day's realized P&L and,
optionally, the intraday extremes of the *within-day* cumulative P&L relative to the
day's opening balance (so a day always starts at 0): ``intraday_min`` (<= 0) and
``intraday_max`` (>= 0), with ``intraday_min <= realized <= intraday_max``. T7 supplies
these from the intraday equity path it simulates; overnight-flat is assumed (the day's
closing equity is ``open + realized``).

Trailing drawdown
-----------------
A high-water mark (HWM) ratchets upward and the loss floor sits ``trailing_dd`` below it.
Two modes differ in what the HWM tracks and what is checked against the floor:

* ``eod`` -- the HWM trails **end-of-day** balances and only the end-of-day balance is
  checked against the floor; intraday dips that recover by the close do not bust.
* ``intraday_unrealized`` -- the HWM trails the **intraday peak equity** (incl. unrealized)
  and the **intraday low equity** is checked against the floor. Because only (min, max)
  are known and not their order, the peak is assumed to occur first (it lifts the floor
  before the trough is tested) -- the worst case, giving a conservative survival estimate.

If ``trailing_freezes_at_start_balance`` is set, once the floor would reach the starting
balance it locks there and stops trailing (Topstep-style).

Daily loss limit
----------------
Hitting ``daily_loss_limit`` ends the day flat: the day's loss is capped at the limit and
the intraday excursion cannot fall further. This is a soft pause under current firm rules,
never a bust by itself.

Consistency rule
----------------
``consistency_max_day_pct`` caps the largest single day's profit as a fraction of total
profit, evaluated when the profit target is reached (a payout gate). If violated the
account has not yet passed and must keep trading to dilute the outsized day.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TRAILING_DRAWDOWN = "trailing_drawdown"


class DayPnL(BaseModel):
    """One trading day's P&L summary (see module docstring for the intraday convention)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    realized: float
    intraday_min: float | None = None
    intraday_max: float | None = None

    @model_validator(mode="after")
    def _check(self) -> DayPnL:
        lo, hi = self.intraday_min, self.intraday_max
        if (lo is None) != (hi is None):
            raise ValueError("intraday_min and intraday_max must both be set or both be None")
        if lo is not None and hi is not None:
            if lo > hi:
                raise ValueError(f"intraday_min ({lo}) must be <= intraday_max ({hi})")
            if lo > 0.0 or hi < 0.0:
                raise ValueError(
                    "intraday extremes are relative to the day open, so require "
                    f"intraday_min <= 0 <= intraday_max (got {lo}, {hi})"
                )
            if not (lo <= self.realized <= hi):
                raise ValueError(
                    f"realized ({self.realized}) must lie within [intraday_min, intraday_max] "
                    f"([{lo}, {hi}])"
                )
        return self


class PropRuleSet(BaseModel):
    """A prop-firm evaluation rule set. Presets (values) live in ``prop_rules.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    start_balance: float = Field(gt=0)
    trailing_dd: float = Field(gt=0)
    trailing_mode: Literal["intraday_unrealized", "eod"]
    trailing_freezes_at_start_balance: bool
    daily_loss_limit: float | None = Field(default=None, gt=0)
    consistency_max_day_pct: float | None = Field(default=None, gt=0, le=1)
    profit_target: float = Field(gt=0)
    min_trading_days: int = Field(ge=0)


class AccountOutcome(BaseModel):
    """The result of running a path through :func:`simulate_account`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    survived: bool
    hit_target: bool
    bust_reason: str | None
    days_elapsed: int


class _Trail:
    """Mutable trailing-drawdown floor tracker for a single account run."""

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


def simulate_account(days: Sequence[DayPnL], rules: PropRuleSet) -> AccountOutcome:
    """Run an account equity path (``days``) against ``rules`` and report the outcome.

    Processing stops at the first of: a trailing-drawdown bust, or a clean pass (profit
    target reached with ``min_trading_days`` elapsed and the consistency rule satisfied).
    If neither occurs the path is exhausted (``survived=True, hit_target=False``).
    """
    trail = _Trail(rules)
    balance = rules.start_balance
    day_profits: list[float] = []

    for day_index, day in enumerate(days, start=1):
        day_open = balance

        if rules.trailing_mode == "intraday_unrealized":
            if day.intraday_min is None or day.intraday_max is None:
                raise ValueError(
                    "trailing_mode='intraday_unrealized' requires intraday_min and "
                    f"intraday_max on every day (day {day_index} is missing them)"
                )
            # Worst case: the intraday peak lifts the floor before the trough is tested.
            trail.update(day_open + day.intraday_max)
            trough_pnl = day.intraday_min
            if rules.daily_loss_limit is not None:
                trough_pnl = max(trough_pnl, -rules.daily_loss_limit)
            if day_open + trough_pnl <= trail.floor:
                return AccountOutcome(
                    survived=False,
                    hit_target=False,
                    bust_reason=TRAILING_DRAWDOWN,
                    days_elapsed=day_index,
                )
            realized = _capped_realized(day, rules)
            balance = day_open + realized
            trail.update(balance)
        else:  # eod
            realized = _capped_realized(day, rules)
            balance = day_open + realized
            if balance <= trail.floor:
                return AccountOutcome(
                    survived=False,
                    hit_target=False,
                    bust_reason=TRAILING_DRAWDOWN,
                    days_elapsed=day_index,
                )
            trail.update(balance)

        day_profits.append(realized)
        if _passes(balance, day_profits, day_index, rules):
            return AccountOutcome(
                survived=True,
                hit_target=True,
                bust_reason=None,
                days_elapsed=day_index,
            )

    return AccountOutcome(
        survived=True,
        hit_target=False,
        bust_reason=None,
        days_elapsed=len(days),
    )


def _capped_realized(day: DayPnL, rules: PropRuleSet) -> float:
    """The day's realized P&L, capped at ``-daily_loss_limit`` if the limit was hit."""
    if rules.daily_loss_limit is None:
        return day.realized
    day_low = day.intraday_min if day.intraday_min is not None else min(0.0, day.realized)
    if day_low <= -rules.daily_loss_limit:
        return -rules.daily_loss_limit
    return day.realized


def _passes(
    balance: float, day_profits: list[float], days_elapsed: int, rules: PropRuleSet
) -> bool:
    """Whether the account has cleanly passed the evaluation as of this day."""
    total_profit = balance - rules.start_balance
    if total_profit < rules.profit_target:
        return False
    if days_elapsed < rules.min_trading_days:
        return False
    if rules.consistency_max_day_pct is not None:
        best_day = max(day_profits)
        if best_day > rules.consistency_max_day_pct * total_profit:
            return False
    return True
