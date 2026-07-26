"""Tests for the red-flag reporter (G10).

Each documented red flag has its own test: implausibly high Sharpe; a high win
rate paired with implausibly smooth equity; and an edge that vanishes under a
1-bar execution delay or under realistic costs (compared against caller-provided
``delayed`` / ``gross`` variants). Thresholds live in :class:`RedFlagConfig`
(pydantic, documented defaults) — no literals in the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from futures_engine.validation.stats import RedFlag, RedFlagConfig, red_flags


@dataclass
class _Result:
    """Structural stand-in for the T6 BacktestResult (satisfies the Protocol)."""

    returns: pd.Series
    equity: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)


def _smooth_equity(n: int = 250, vol: float = 0.01, drift: float = 0.001) -> _Result:
    """Gently rising equity with a tiny drawdown: max_dd << k * vol."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(drift, vol, n))
    equity = (1.0 + r).cumprod() * 100.0
    return _Result(returns=r, equity=equity, metrics={"sharpe": 1.0, "win_rate": 0.55})


def _codes(flags: list[RedFlag]) -> set[str]:
    return {f.code for f in flags}


# --- Sharpe too high ---------------------------------------------------------


def test_flags_implausibly_high_sharpe() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 3.5
    flags = red_flags(res, None, None)
    hits = [f for f in flags if f.code == "SHARPE_IMPLAUSIBLE"]
    assert len(hits) == 1
    assert hits[0].severity == "warn"


def test_does_not_flag_reasonable_sharpe() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 2.0
    assert "SHARPE_IMPLAUSIBLE" not in _codes(red_flags(res, None, None))


# --- high win rate + smooth equity -------------------------------------------


def test_flags_high_winrate_with_smooth_equity() -> None:
    res = _smooth_equity()  # tiny drawdown -> smooth
    res.metrics["win_rate"] = 0.85
    flags = red_flags(res, None, None)
    hits = [f for f in flags if f.code == "SMOOTH_HIGH_WINRATE"]
    assert len(hits) == 1
    assert hits[0].severity == "warn"


def test_high_winrate_but_rough_equity_is_not_flagged() -> None:
    res = _smooth_equity()
    res.metrics["win_rate"] = 0.85
    # Inject a deep drawdown so max_dd > k * vol (not smooth).
    eq = res.equity.to_numpy().copy()
    eq[120:130] *= 0.4  # ~60% drawdown
    res.equity = pd.Series(eq)
    assert "SMOOTH_HIGH_WINRATE" not in _codes(red_flags(res, None, None))


def test_smooth_equity_but_modest_winrate_is_not_flagged() -> None:
    res = _smooth_equity()
    res.metrics["win_rate"] = 0.55
    assert "SMOOTH_HIGH_WINRATE" not in _codes(red_flags(res, None, None))


# --- edge vanishes under 1-bar delay -----------------------------------------


def test_flags_edge_that_dies_under_delay() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 1.5
    delayed = _smooth_equity()
    delayed.metrics["sharpe"] = 0.2  # collapses under a 1-bar delay
    flags = red_flags(res, delayed, None)
    hits = [f for f in flags if f.code == "EDGE_FAILS_DELAY"]
    assert len(hits) == 1
    assert hits[0].severity == "fail"


def test_no_delay_flag_when_edge_survives() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 1.5
    delayed = _smooth_equity()
    delayed.metrics["sharpe"] = 1.4  # essentially retained
    assert "EDGE_FAILS_DELAY" not in _codes(red_flags(res, delayed, None))


# --- edge vanishes under realistic costs -------------------------------------


def test_flags_edge_that_dies_under_costs() -> None:
    res = _smooth_equity()  # net (post-cost) result
    res.metrics["sharpe"] = 0.3
    gross = _smooth_equity()  # pre-cost result
    gross.metrics["sharpe"] = 2.0
    flags = red_flags(res, None, gross)
    hits = [f for f in flags if f.code == "EDGE_FAILS_COSTS"]
    assert len(hits) == 1
    assert hits[0].severity == "fail"


def test_no_cost_flag_when_net_edge_survives() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 1.7
    gross = _smooth_equity()
    gross.metrics["sharpe"] = 2.0
    assert "EDGE_FAILS_COSTS" not in _codes(red_flags(res, None, gross))


# --- clean result and optional variants --------------------------------------


def test_clean_result_produces_no_flags() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 1.2
    res.metrics["win_rate"] = 0.55
    delayed = _smooth_equity()
    delayed.metrics["sharpe"] = 1.1
    gross = _smooth_equity()
    gross.metrics["sharpe"] = 1.4
    assert red_flags(res, delayed, gross) == []


def test_missing_variants_do_not_crash() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 1.2
    assert red_flags(res, None, None) == []


# --- configurability ---------------------------------------------------------


def test_custom_threshold_changes_outcome() -> None:
    res = _smooth_equity()
    res.metrics["sharpe"] = 2.5
    strict = RedFlagConfig(sharpe_max=2.0)
    assert "SHARPE_IMPLAUSIBLE" in _codes(red_flags(res, None, None, config=strict))
    assert "SHARPE_IMPLAUSIBLE" not in _codes(red_flags(res, None, None))


def test_redflag_config_defaults() -> None:
    cfg = RedFlagConfig()
    assert cfg.sharpe_max == 3.0
    assert cfg.win_rate_max == 0.70
    assert cfg.smoothness_k > 0
    assert 0.0 <= cfg.min_edge_retention <= 1.0


def test_redflag_config_is_frozen_and_strict() -> None:
    cfg = RedFlagConfig()
    with pytest.raises(ValidationError):
        cfg.sharpe_max = 5.0  # frozen
    with pytest.raises(ValidationError):
        RedFlagConfig(unknown_field=1)  # extra forbidden


def test_redflag_is_frozen_with_valid_severity() -> None:
    flag = RedFlag(code="X", message="m", severity="warn")
    assert flag.severity == "warn"
    with pytest.raises(ValidationError):
        RedFlag(code="X", message="m", severity="explode")  # not in Literal
