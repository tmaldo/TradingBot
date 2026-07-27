"""Tests for the volatility-targeted, survival-dominated position sizer (T7, G12).

``position_size`` returns the minimum of four legs -- the volatility-target
size, the fractional-Kelly size (capped at quarter-Kelly), the
``survival_max_contracts`` handed in by the caller, and the hard
``max_contracts`` ceiling -- floored at zero. Each leg is unit-tested as the
binding minimum, and the survival constraint is shown to dominate Kelly when
they disagree (the drawdown rule wins, G12).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from futures_engine.core.types import InstrumentSpec
from futures_engine.sizing.position import (
    EdgeStats,
    SizingConfig,
    position_size,
)

# MES-like spec: $5 per point, 0.25 tick -> $1.25 per tick.
MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5.0,
    currency="USD",
)


def _cfg(**overrides: object) -> SizingConfig:
    base: dict[str, object] = {
        "target_daily_vol_usd": 500.0,
        "kelly_fraction_cap": 0.25,
        "max_contracts": 100,
    }
    base.update(overrides)
    return SizingConfig(**base)  # type: ignore[arg-type]


# --- SizingConfig validation: quarter-Kelly ceiling (G12) --------------------


def test_kelly_fraction_cap_above_quarter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(kelly_fraction_cap=0.26)
    with pytest.raises(ValidationError):
        _cfg(kelly_fraction_cap=0.5)
    # exactly quarter-Kelly is allowed.
    assert _cfg(kelly_fraction_cap=0.25).kelly_fraction_cap == 0.25
    # non-positive is also rejected.
    with pytest.raises(ValidationError):
        _cfg(kelly_fraction_cap=0.0)


def test_edgestats_validates_probability_and_magnitudes() -> None:
    with pytest.raises(ValidationError):
        EdgeStats(p_win=1.5, avg_win=100.0, avg_loss=80.0)
    with pytest.raises(ValidationError):
        EdgeStats(p_win=0.5, avg_win=-1.0, avg_loss=80.0)  # avg_win must be > 0
    with pytest.raises(ValidationError):
        EdgeStats(p_win=0.5, avg_win=100.0, avg_loss=-1.0)  # avg_loss magnitude > 0


# --- each leg is the binding minimum -----------------------------------------


def _edge(**overrides: object) -> EdgeStats:
    base: dict[str, object] = {"p_win": 0.6, "avg_win": 300.0, "avg_loss": 200.0}
    base.update(overrides)
    return EdgeStats(**base)  # type: ignore[arg-type]


def test_vol_target_leg_binds() -> None:
    # vol-target = floor(target_daily_vol_usd / vol_estimate) = floor(500/200)=2.
    # Small avg_loss makes the Kelly leg large (floor(0.25*0.899*500/10)=11), and
    # survival/max are generous, so the vol-target leg (2) is the binding minimum.
    size = position_size(
        vol_estimate=200.0,
        edge=_edge(p_win=0.9, avg_win=1_000.0, avg_loss=10.0),  # large Kelly leg
        spec=MES,
        cfg=_cfg(target_daily_vol_usd=500.0, max_contracts=100),
        survival_max_contracts=100,
    )
    assert size == 2


def test_kelly_leg_binds() -> None:
    # Generous vol budget and survival/max; Kelly is the tightest.
    # f* = p - (1-p)*avg_loss/avg_win = 0.6 - 0.4*200/300 = 0.3333...
    # kelly contracts = floor(cap * f* * target_daily_vol_usd / avg_loss)
    #                 = floor(0.25 * 0.3333 * 5000 / 200) = floor(2.083) = 2
    size = position_size(
        vol_estimate=1.0,  # vol-target leg huge
        edge=_edge(),
        spec=MES,
        cfg=_cfg(target_daily_vol_usd=5_000.0, kelly_fraction_cap=0.25, max_contracts=100),
        survival_max_contracts=100,
    )
    f_star = 0.6 - 0.4 * 200.0 / 300.0
    expected = math.floor(0.25 * f_star * 5_000.0 / 200.0)
    assert size == expected == 2


def test_max_contracts_leg_binds() -> None:
    size = position_size(
        vol_estimate=1.0,
        edge=_edge(p_win=0.95, avg_win=1_000.0, avg_loss=50.0),
        spec=MES,
        cfg=_cfg(target_daily_vol_usd=1e9, max_contracts=4),
        survival_max_contracts=100,
    )
    assert size == 4


def test_survival_leg_binds_and_dominates_kelly() -> None:
    # Kelly and vol-target would both allow far more, but survival permits only 1.
    # The drawdown rule must win (G12).
    kelly_generous = _edge(p_win=0.95, avg_win=2_000.0, avg_loss=50.0)
    size = position_size(
        vol_estimate=1.0,  # vol leg huge
        edge=kelly_generous,
        spec=MES,
        cfg=_cfg(target_daily_vol_usd=1e9, kelly_fraction_cap=0.25, max_contracts=100),
        survival_max_contracts=1,
    )
    assert size == 1


# --- edge cases --------------------------------------------------------------


def test_negative_edge_gives_zero_kelly_and_zero_size() -> None:
    # f* <= 0 -> Kelly leg is 0 -> final size floored at 0.
    size = position_size(
        vol_estimate=1.0,
        edge=EdgeStats(p_win=0.4, avg_win=100.0, avg_loss=200.0),  # negative edge
        spec=MES,
        cfg=_cfg(target_daily_vol_usd=1e9),
        survival_max_contracts=100,
    )
    assert size == 0


def test_zero_survival_contracts_forces_zero_size() -> None:
    size = position_size(
        vol_estimate=200.0,
        edge=_edge(),
        spec=MES,
        cfg=_cfg(),
        survival_max_contracts=0,
    )
    assert size == 0


def test_non_positive_vol_estimate_is_rejected() -> None:
    with pytest.raises(ValueError, match="vol_estimate"):
        position_size(
            vol_estimate=0.0,
            edge=_edge(),
            spec=MES,
            cfg=_cfg(),
            survival_max_contracts=10,
        )


def test_result_is_a_plain_int() -> None:
    size = position_size(
        vol_estimate=200.0,
        edge=_edge(),
        spec=MES,
        cfg=_cfg(),
        survival_max_contracts=10,
    )
    assert isinstance(size, int)
