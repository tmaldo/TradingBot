"""Tests for the prop-firm survival rule engine (task T2, G11/G16).

The survival simulator (T7) Monte-Carlos account paths against these rules, so the
mechanics are hand-verified here: trailing drawdown in both EOD and intraday-unrealized
modes, the start-balance freeze, the daily-loss-limit cap, the consistency rule, exact
boundary touches, and a hypothesis monotonicity property.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from futures_engine.prop.rules import (
    AccountOutcome,
    DayPnL,
    PropRuleSet,
    simulate_account,
)


def _rules(**overrides: object) -> PropRuleSet:
    base: dict[str, object] = {
        "name": "test",
        "start_balance": 50_000.0,
        "trailing_dd": 2_000.0,
        "trailing_mode": "eod",
        "trailing_freezes_at_start_balance": False,
        "daily_loss_limit": None,
        "consistency_max_day_pct": None,
        "profit_target": 3_000.0,
        "min_trading_days": 0,
    }
    base.update(overrides)
    return PropRuleSet(**base)  # type: ignore[arg-type]


def _day(realized: float, lo: float | None = None, hi: float | None = None) -> DayPnL:
    return DayPnL(realized=realized, intraday_min=lo, intraday_max=hi)


# --- basic outcomes ----------------------------------------------------------


def test_flat_days_survive_without_hitting_target() -> None:
    out = simulate_account([_day(0.0, 0.0, 0.0)] * 3, _rules())
    assert out == AccountOutcome(survived=True, hit_target=False, bust_reason=None, days_elapsed=3)


def test_reaching_profit_target_passes() -> None:
    # +1000/day for 3 days -> cumulative 3000 == target, passes on day 3.
    days = [_day(1_000.0, 0.0, 1_000.0)] * 3
    out = simulate_account(days, _rules())
    assert out.survived and out.hit_target
    assert out.days_elapsed == 3


# --- trailing drawdown: EOD vs intraday-unrealized ---------------------------


def test_survives_eod_trailing_but_busts_intraday_trailing() -> None:
    # start 50,000; trailing_dd 2,000; no freeze, no DLL. Initial floor = 48,000.
    # One day: dips to -2,500 intraday (equity 47,500) but closes at +100 (equity 50,100).
    day = _day(realized=100.0, lo=-2_500.0, hi=100.0)

    # EOD mode: trailing checks only the END-OF-DAY balance.
    #   EOD equity 50,100 > floor 48,000 -> SURVIVES (intraday dip ignored).
    eod = simulate_account([day], _rules(trailing_mode="eod"))
    assert eod.survived is True
    assert eod.bust_reason is None

    # Intraday mode: peak 50,100 lifts floor to 48,100; trough equity 47,500 <= 48,100
    #   -> BUSTS on the intraday excursion.
    intr = simulate_account([day], _rules(trailing_mode="intraday_unrealized"))
    assert intr.survived is False
    assert intr.bust_reason == "trailing_drawdown"
    assert intr.days_elapsed == 1


def test_trailing_freeze_locks_floor_at_start_balance() -> None:
    # dd 2,000. Day 1 closes +2,500 -> EOD 52,500; unfrozen floor would be 50,500,
    # but with freeze the floor locks at the start balance 50,000.
    # Day 2 closes at 50,100 (realized -2,400):
    #   frozen  -> floor 50,000; 50,100 > 50,000 -> SURVIVES
    #   unfrozen-> floor 50,500; 50,100 <= 50,500 -> BUSTS
    days = [_day(2_500.0, 0.0, 2_500.0), _day(-2_400.0, -2_400.0, 0.0)]

    frozen = simulate_account(days, _rules(trailing_freezes_at_start_balance=True))
    assert frozen.survived is True

    unfrozen = simulate_account(days, _rules(trailing_freezes_at_start_balance=False))
    assert unfrozen.survived is False
    assert unfrozen.bust_reason == "trailing_drawdown"


def test_exact_touch_of_trailing_floor_busts() -> None:
    # floor 48,000; EOD equity exactly 48,000 -> breach (<=).
    busts = simulate_account([_day(-2_000.0, -2_000.0, 0.0)], _rules())
    assert busts.survived is False
    assert busts.bust_reason == "trailing_drawdown"

    # one cent above the floor survives.
    survives = simulate_account([_day(-1_999.99, -1_999.99, 0.0)], _rules())
    assert survives.survived is True


# --- daily loss limit --------------------------------------------------------


def test_daily_loss_limit_caps_the_day_and_can_prevent_a_bust() -> None:
    # start 50,000; dd 1,500 -> floor 48,500; day dips/realizes -1,500.
    #   DLL 1,000  -> loss capped at -1,000 -> EOD 49,000 > 48,500 -> SURVIVES
    #   no DLL     -> loss -1,500 -> EOD 48,500 <= 48,500 -> BUSTS (exact touch)
    day = _day(-1_500.0, -1_500.0, 0.0)

    with_dll = simulate_account([day], _rules(trailing_dd=1_500.0, daily_loss_limit=1_000.0))
    assert with_dll.survived is True

    no_dll = simulate_account([day], _rules(trailing_dd=1_500.0, daily_loss_limit=None))
    assert no_dll.survived is False
    assert no_dll.bust_reason == "trailing_drawdown"


def test_daily_loss_limit_is_not_a_bust_reason() -> None:
    # Hitting the DLL ends the day flat; it never busts the account by itself.
    out = simulate_account(
        [_day(-5_000.0, -5_000.0, 0.0)],
        _rules(trailing_dd=50_000.0, daily_loss_limit=1_000.0),
    )
    assert out.survived is True
    assert out.bust_reason is None


# --- consistency rule --------------------------------------------------------


def test_consistency_rule_delays_pass_until_best_day_diluted() -> None:
    # target 3,000; best single day must be <= 50% of total profit at pass time.
    # Day1 +2,000, Day2 +1,000 -> total 3,000, best 2,000 > 1,500 -> cannot pass yet.
    # Day3 +2,000 -> total 5,000, best 2,000 <= 2,500 -> passes on day 3.
    days = [
        _day(2_000.0, 0.0, 2_000.0),
        _day(1_000.0, 0.0, 1_000.0),
        _day(2_000.0, 0.0, 2_000.0),
    ]
    wide = {"trailing_dd": 50_000.0}  # keep trailing out of the way

    without = simulate_account(days, _rules(consistency_max_day_pct=None, **wide))
    assert without.hit_target and without.days_elapsed == 2  # passes as soon as target hit

    withc = simulate_account(days, _rules(consistency_max_day_pct=0.5, **wide))
    assert withc.hit_target and withc.days_elapsed == 3


def test_consistency_boundary_exact_ratio_passes() -> None:
    # best day exactly 50% of total -> allowed (<=).
    days = [_day(1_500.0, 0.0, 1_500.0), _day(1_500.0, 0.0, 1_500.0)]
    out = simulate_account(days, _rules(trailing_dd=50_000.0, consistency_max_day_pct=0.5))
    assert out.hit_target and out.days_elapsed == 2


# --- minimum trading days ----------------------------------------------------


def test_min_trading_days_delays_pass() -> None:
    days = [_day(3_000.0, 0.0, 3_000.0), _day(0.0, 0.0, 0.0)]
    wide = {"trailing_dd": 50_000.0}

    one = simulate_account(days, _rules(min_trading_days=1, **wide))
    assert one.hit_target and one.days_elapsed == 1

    two = simulate_account(days, _rules(min_trading_days=2, **wide))
    assert two.hit_target and two.days_elapsed == 2


# --- input validation --------------------------------------------------------


def test_intraday_mode_requires_intraday_extremes() -> None:
    with pytest.raises(ValueError, match="intraday"):
        simulate_account([_day(0.0)], _rules(trailing_mode="intraday_unrealized"))


def test_daypnl_rejects_inconsistent_extremes() -> None:
    with pytest.raises(ValidationError):
        DayPnL(realized=0.0, intraday_min=5.0, intraday_max=-5.0)  # min > max
    with pytest.raises(ValidationError):
        DayPnL(realized=100.0, intraday_min=-10.0, intraday_max=10.0)  # realized outside
    with pytest.raises(ValidationError):
        DayPnL(realized=0.0, intraday_min=-1.0, intraday_max=None)  # only one set


def test_propruleset_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        _rules(consistency_max_day_pct=1.5)  # > 1
    with pytest.raises(ValidationError):
        _rules(trailing_dd=-100.0)  # must be > 0


# --- property: busting is monotone in trailing_dd ----------------------------


@st.composite
def _day_st(draw: st.DrawFn) -> DayPnL:
    realized = draw(st.floats(min_value=-3_000, max_value=3_000, allow_nan=False))
    drop = draw(st.floats(min_value=0, max_value=3_000, allow_nan=False))
    rise = draw(st.floats(min_value=0, max_value=3_000, allow_nan=False))
    lo = min(0.0, realized) - drop
    hi = max(0.0, realized) + rise
    return DayPnL(realized=realized, intraday_min=lo, intraday_max=hi)


@given(
    days=st.lists(_day_st(), min_size=1, max_size=15),
    mode=st.sampled_from(["eod", "intraday_unrealized"]),
    freeze=st.booleans(),
)
def test_busting_is_monotone_in_trailing_dd(days: list[DayPnL], mode: str, freeze: bool) -> None:
    # A tighter (smaller) trailing_dd must never survive a path a looser one busted on.
    # Profit target unreachable so the run always exhausts the path or busts (isolates
    # bust behaviour from passing).
    dds = [500.0, 1_000.0, 2_000.0, 3_500.0, 6_000.0]
    survived = [
        simulate_account(
            days,
            _rules(
                trailing_dd=dd,
                trailing_mode=mode,
                trailing_freezes_at_start_balance=freeze,
                profit_target=1e12,
            ),
        ).survived
        for dd in dds
    ]
    # survival is non-decreasing in trailing_dd: once True it stays True.
    for tighter, looser in pairwise(survived):
        assert (not tighter) or looser
