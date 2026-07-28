"""Tests for the prop-account survival Monte-Carlo gate (task T7, G11/G16).

The survival simulator resamples a strategy's TradeLog with an ``arch`` block
bootstrap into account-equity paths and reports the probability the account
survives (does not breach the trailing-drawdown rule) over a horizon. These
tests pin the three analytically known cases required by the acceptance
criteria -- (a) all-winning -> p_survival == 1, (b) deterministic bust ->
p_survival == 0, (c) a symmetric random walk vs the closed-form gambler's-ruin
survival probability -- plus reproducibility, the 90% CI, the contract
monotonicity property, and the ``passes_gate`` helper.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from futures_engine.prop.rules import PropRuleSet
from futures_engine.prop.survival import (
    SurvivalReport,
    max_contracts_surviving,
    monte_carlo_survival,
    passes_gate,
)

# --- helpers -----------------------------------------------------------------


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


def _trades(net_pnls: list[float]) -> pd.DataFrame:
    """Minimal conforming TradeLog: one trade per calendar day of ``net_pnls``."""
    ts = pd.date_range("2024-01-01", periods=len(net_pnls), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "entry_ts": ts,
            "exit_ts": ts,
            "side": "long",
            "qty": 1,
            "entry_px": 0.0,
            "exit_px": 0.0,
            "gross_pnl_usd": net_pnls,
            "net_pnl_usd": net_pnls,
        }
    )


def _trades_mae_mfe(
    net_pnls: list[float], mae_usd: list[float], mfe_usd: list[float]
) -> pd.DataFrame:
    """A TradeLog carrying per-trade excursion columns (``mae_usd`` <= 0, ``mfe_usd`` >= 0).

    These widen each resampled day's intraday_min/max in the survival simulator, so
    they matter only for the ``intraday_unrealized`` trailing mode.
    """
    frame = _trades(net_pnls)
    frame["mae_usd"] = mae_usd
    frame["mfe_usd"] = mfe_usd
    return frame


# --- analytic case (a): all-winning trades -> p_survival == 1.0 --------------


def test_all_winning_trades_survive_with_probability_one() -> None:
    report = monte_carlo_survival(
        _trades([100.0, 150.0, 200.0, 120.0, 180.0]),
        _rules(profit_target=1e12),  # target unreachable: isolate survival
        contracts=1,
        n_paths=200,
        horizon_days=20,
        seed=7,
    )
    assert report.p_survival == 1.0
    assert report.bust_reasons == {}
    assert report.ci_90[1] == 1.0


# --- analytic case (b): deterministic monotone loss -> p_survival == 0.0 -----


def test_deterministic_loss_sequence_busts_with_probability_zero() -> None:
    # Every resampled trade is -500; trailing_dd 2,000 -> busts within ~4 days
    # on every path. Horizon far exceeds that, so no path can survive.
    report = monte_carlo_survival(
        _trades([-500.0] * 6),
        _rules(trailing_dd=2_000.0),
        contracts=1,
        n_paths=200,
        horizon_days=40,
        seed=3,
    )
    assert report.p_survival == 0.0
    assert report.p_target_before_bust == 0.0
    assert report.median_days_to_target is None
    assert report.bust_reasons == {"trailing_drawdown": 1.0}


# --- analytic case (c): symmetric random walk vs gambler's ruin --------------


def test_symmetric_random_walk_matches_gamblers_ruin() -> None:
    # Driftless +/-1 (in $) steps. Fixed-barrier gambler's ruin from a start with
    # distance D down to the loss barrier and G up to the target gives
    # P(hit target first) = D / (D + G). The trailing floor ratchets up on new
    # highs, so the simulated survival sits at or below the fixed-barrier value
    # (a conservative approximation); we assert agreement within tolerance and on
    # the correct (conservative) side. Large start balance keeps barriers ~fixed.
    step = 100.0
    dd = 1_500.0  # 15 steps of drawdown room to the loss barrier
    target = 500.0  # 5 steps up to the profit target
    rules = _rules(
        start_balance=1_000_000.0,
        trailing_dd=dd,
        trailing_mode="eod",
        profit_target=target,
        min_trading_days=0,
    )
    report = monte_carlo_survival(
        _trades([step, -step] * 25),
        rules,
        contracts=1,
        n_paths=4000,
        horizon_days=400,
        block_size=1,  # i.i.d. +/-1 walk: no autocorrelation to preserve
        seed=11,
    )
    gamblers_ruin = dd / (dd + target)  # 0.75
    # A modest target keeps ratcheting small, so the trailing floor is close to a
    # fixed barrier: the simulated value tracks the closed form to within ~0.02,
    # sitting on the conservative (<=) side because trailing can only add busts.
    assert report.p_target_before_bust <= gamblers_ruin + 0.01
    assert abs(report.p_target_before_bust - gamblers_ruin) < 0.05


# --- reproducibility ---------------------------------------------------------


def test_seeded_runs_are_reproducible() -> None:
    trades = _trades([120.0, -80.0, 60.0, -140.0, 200.0, -90.0])
    kw = dict(rules=_rules(), contracts=2, n_paths=300, horizon_days=25)
    a = monte_carlo_survival(trades, seed=42, **kw)  # type: ignore[arg-type]
    b = monte_carlo_survival(trades, seed=42, **kw)  # type: ignore[arg-type]
    c = monte_carlo_survival(trades, seed=43, **kw)  # type: ignore[arg-type]
    assert a == b
    assert a != c  # a different seed gives a (statistically) different report


# --- 90% CI and bust-reason histogram ----------------------------------------


def test_ci_90_brackets_p_survival_and_bust_reasons_normalized() -> None:
    trades = _trades([120.0, -80.0, 60.0, -140.0, 200.0, -300.0])
    report = monte_carlo_survival(
        trades, _rules(), contracts=3, n_paths=1000, horizon_days=30, seed=5
    )
    lo, hi = report.ci_90
    assert 0.0 <= lo <= report.p_survival <= hi <= 1.0
    # bust fractions + survival fractions cover every path exactly once.
    assert report.bust_reasons  # some paths bust at this size
    total_bust = sum(report.bust_reasons.values())
    assert total_bust == pytest.approx(1.0 - report.p_survival, abs=1e-12)


# --- monotonicity: p_survival non-increasing in contract count ---------------


def test_p_survival_is_monotone_non_increasing_in_contracts() -> None:
    # Fixed rules/trades/seed; a high profit target isolates bust behaviour so
    # survival is governed purely by the trailing-drawdown rule. Bigger size ->
    # same-or-more busts, hence non-increasing survival.
    trades = _trades([250.0, -200.0, 180.0, -260.0, 300.0, -220.0, 150.0])
    rules = _rules(trailing_dd=3_000.0, profit_target=1e12)
    ps = [
        monte_carlo_survival(
            trades, rules, contracts=k, n_paths=800, horizon_days=30, seed=9
        ).p_survival
        for k in range(1, 9)
    ]
    for smaller, bigger in pairwise(ps):
        assert bigger <= smaller + 1e-12


# --- passes_gate -------------------------------------------------------------


def test_passes_gate_uses_default_and_custom_threshold() -> None:
    high = SurvivalReport(
        p_survival=0.95,
        p_target_before_bust=0.9,
        median_days_to_target=10.0,
        bust_reasons={"trailing_drawdown": 0.05},
        ci_90=(0.93, 0.97),
    )
    low = high.model_copy(update={"p_survival": 0.80})
    assert passes_gate(high) is True  # default 0.90
    assert passes_gate(low) is False
    assert passes_gate(low, min_survival=0.75) is True
    # exact boundary passes (>=).
    boundary = high.model_copy(update={"p_survival": 0.90})
    assert passes_gate(boundary) is True


# --- max_contracts_surviving helper ------------------------------------------


def test_max_contracts_surviving_is_largest_passing_size() -> None:
    trades = _trades([250.0, -200.0, 180.0, -260.0, 300.0, -220.0, 150.0])
    rules = _rules(trailing_dd=3_000.0, profit_target=1e12)
    n = max_contracts_surviving(trades, rules, n_paths=800, horizon_days=30, seed=9, max_search=12)
    # k contracts passes, k+1 does not (boundary is exact).
    at_n = monte_carlo_survival(trades, rules, contracts=n, n_paths=800, horizon_days=30, seed=9)
    assert passes_gate(at_n)
    if n < 12:
        above = monte_carlo_survival(
            trades, rules, contracts=n + 1, n_paths=800, horizon_days=30, seed=9
        )
        assert not passes_gate(above)


def test_max_contracts_surviving_returns_zero_when_one_lot_fails() -> None:
    # Deterministic bust even at a single contract -> no surviving size.
    n = max_contracts_surviving(
        _trades([-500.0] * 6),
        _rules(trailing_dd=2_000.0),
        n_paths=200,
        horizon_days=40,
        seed=3,
        max_search=8,
    )
    assert n == 0


# --- intraday_unrealized trailing mode WITH MAE/MFE excursions (G11) ----------


def test_intraday_unrealized_mae_widening_busts_where_eod_survives() -> None:
    # Every trade closes at +50 (a *win*), so in ``eod`` mode the account only ever
    # gains and never busts. But each trade digs an intraday trough of -2500 (MAE)
    # and a peak of +200 (MFE); against a 2,000 trailing drawdown the intraday LOW
    # equity breaches the floor on day 1 of every path. This exercises the MAE/MFE
    # widening in _build_days and the aligned mae/mfe resampling, and proves the
    # intraday machinery -- not the realized close -- is what triggers the bust.
    trades = _trades_mae_mfe([50.0] * 6, mae_usd=[-2_500.0] * 6, mfe_usd=[200.0] * 6)

    eod = monte_carlo_survival(
        trades,
        _rules(trailing_mode="eod", trailing_dd=2_000.0, profit_target=1e12),
        contracts=1,
        n_paths=200,
        horizon_days=20,
        seed=13,
    )
    assert eod.p_survival == 1.0  # closes-only view: all wins, never busts
    assert eod.bust_reasons == {}

    intraday = monte_carlo_survival(
        trades,
        _rules(trailing_mode="intraday_unrealized", trailing_dd=2_000.0, profit_target=1e12),
        contracts=1,
        n_paths=200,
        horizon_days=20,
        seed=13,
    )
    assert intraday.p_survival == 0.0  # intraday trough breaches the floor day 1
    assert intraday.bust_reasons == {"trailing_drawdown": 1.0}


# --- intraday_unrealized trailing mode WITHOUT MAE/MFE (realized-only fallback) ---


def test_intraday_unrealized_realized_only_fallback_survives_all_wins() -> None:
    # No mae_usd/mfe_usd columns: _build_days falls back to realized-only intraday
    # extremes (intraday_min = min(0, cum), intraday_max = max(0, cum)). For a stream
    # of winning trades that means no intraday downside, so intraday_unrealized cannot
    # breach the floor -- p_survival == 1.0. This pins the realized-only branch and
    # contrasts directly with the MAE-widening test above (same winning closes -> 0.0).
    report = monte_carlo_survival(
        _trades([100.0, 150.0, 200.0, 120.0, 180.0]),
        _rules(trailing_mode="intraday_unrealized", trailing_dd=2_000.0, profit_target=1e12),
        contracts=1,
        n_paths=200,
        horizon_days=20,
        seed=17,
    )
    assert report.p_survival == 1.0
    assert report.bust_reasons == {}


def test_intraday_unrealized_realized_only_fallback_busts_on_losses() -> None:
    # Realized-only fallback again, but a deterministic losing stream: every resampled
    # trade is -600, so in intraday_unrealized mode the intraday low equals the close
    # and the balance grinds through the 2,000 floor on every path. Known answer: no
    # path survives, and the only bust reason is the trailing drawdown.
    report = monte_carlo_survival(
        _trades([-600.0] * 6),
        _rules(trailing_mode="intraday_unrealized", trailing_dd=2_000.0),
        contracts=1,
        n_paths=200,
        horizon_days=40,
        seed=19,
    )
    assert report.p_survival == 0.0
    assert report.bust_reasons == {"trailing_drawdown": 1.0}


# --- daily_loss_limit (G11) --------------------------------------------------
# NOTE: per T2 (futures_engine/prop/rules.py) the daily loss limit is a *soft* rule
# -- it caps a day's realized loss and is "never a bust by itself", so it produces
# no bust_reasons key. We therefore assert its real, observable effect on survival
# (a bust_reasons-key assertion would test behaviour that T2 deliberately does not
# implement).


def test_daily_loss_limit_caps_losses_and_prevents_a_bust() -> None:
    # Every trade is -1,500 against a 2,000 trailing drawdown over a 2-day horizon.
    # Without a daily loss limit the account is 50,000 -> 48,500 -> 47,000, breaching
    # the 48,000 floor on day 2 (p_survival == 0). With a 500 daily loss limit each
    # day's loss is capped at -500 (50,000 -> 49,500 -> 49,000), so both days clear
    # the floor and every path survives the horizon (p_survival == 1). This pins the
    # _capped_realized path driven by daily_loss_limit.
    trades = _trades([-1_500.0] * 6)

    uncapped = monte_carlo_survival(
        trades,
        _rules(trailing_dd=2_000.0, daily_loss_limit=None),
        contracts=1,
        n_paths=200,
        horizon_days=2,
        seed=23,
    )
    assert uncapped.p_survival == 0.0
    assert uncapped.bust_reasons == {"trailing_drawdown": 1.0}

    capped = monte_carlo_survival(
        trades,
        _rules(trailing_dd=2_000.0, daily_loss_limit=500.0),
        contracts=1,
        n_paths=200,
        horizon_days=2,
        seed=23,
    )
    assert capped.p_survival == 1.0
    assert capped.bust_reasons == {}


# --- consistency_max_day_pct (G11) -------------------------------------------
# NOTE: per T2 the consistency rule is a *payout gate*, not a bust condition -- when
# violated the account simply has not passed yet and keeps trading. So it too emits
# no bust_reasons key; we assert its real effect on whether the target is booked.


def test_consistency_rule_blocks_the_target_until_diluted() -> None:
    # Identical +1,000 winning trades; profit target 1,000 is reached on day 1. With a
    # 50% consistency cap the day-1 profit is 100% of total profit (> 50%), so the pass
    # is withheld: over a 1-day horizon no path books the target even though all survive
    # (p_target_before_bust == 0, p_survival == 1). Without the consistency cap the same
    # day-1 profit passes immediately (p_target_before_bust == 1). This exercises the
    # consistency branch of _passes.
    trades = _trades([1_000.0] * 6)
    common = dict(contracts=1, n_paths=200, horizon_days=1, seed=29)

    blocked = monte_carlo_survival(
        trades,
        _rules(profit_target=1_000.0, min_trading_days=1, consistency_max_day_pct=0.5),
        **common,
    )
    assert blocked.p_survival == 1.0
    assert blocked.p_target_before_bust == 0.0
    assert blocked.median_days_to_target is None

    allowed = monte_carlo_survival(
        trades,
        _rules(profit_target=1_000.0, min_trading_days=1, consistency_max_day_pct=None),
        **common,
    )
    assert allowed.p_target_before_bust == 1.0
    assert allowed.median_days_to_target == 1.0
