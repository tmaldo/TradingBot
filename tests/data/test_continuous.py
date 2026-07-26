"""Tests for the continuous-contract builder (Global Constraint G3).

Two layers:

* integration tests over the recorded five-contract fixture (``continuous_fixture``)
  -- volume / open-interest / calendar roll detection, recorded roll dates and
  underlying contracts, UTC timestamps, and a DST-spanning roll;
* hypothesis **property tests** (G16, highest-stakes): panama-difference
  adjustment removes the roll gap (adjusted step == the front contract's own
  step) while a raw splice shows it; ratio adjustment preserves returns.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from futures_engine.core.types import BAR_COLUMNS, ContinuousMeta
from futures_engine.data.continuous import (
    DEFAULT_CALENDAR_OFFSET_DAYS,
    build_continuous,
)
from futures_engine.data.provider import ContractInfo
from futures_engine.data.store import DataIntegrityError

# Expected roll dates baked by tests/fixtures/_generate.py (crossover + 1
# confirming session; volume crosses at 50% of each overlap, OI at 30%).
EXPECTED_VOLUME_ROLLS = [
    date(2024, 3, 8),
    date(2024, 6, 11),
    date(2024, 9, 11),
    date(2024, 12, 11),
]
EXPECTED_OI_ROLLS = [
    date(2024, 3, 5),
    date(2024, 6, 6),
    date(2024, 9, 5),
    date(2024, 12, 5),
]
ORDERED_SYMBOLS = ["MESH24", "MESM24", "MESU24", "MESZ24", "MESH25"]


# --- fixture integration -----------------------------------------------------


def test_volume_roll_dates_and_underlying(continuous_fixture) -> None:
    bars, meta = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "volume", "panama_diff"
    )
    assert isinstance(meta, ContinuousMeta)
    assert meta.roll_rule == "volume"
    assert meta.adjustment == "panama_diff"
    assert meta.roll_dates == EXPECTED_VOLUME_ROLLS
    assert meta.underlying_contracts == ORDERED_SYMBOLS
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique


def test_open_interest_rolls_differ_from_volume(continuous_fixture) -> None:
    _, oi_meta = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "open_interest", "none"
    )
    assert oi_meta.roll_dates == EXPECTED_OI_ROLLS
    # The two rules must genuinely disagree (OI leads volume here).
    assert oi_meta.roll_dates != EXPECTED_VOLUME_ROLLS
    for oi_roll, vol_roll in zip(EXPECTED_OI_ROLLS, EXPECTED_VOLUME_ROLLS, strict=True):
        assert oi_roll < vol_roll


def test_calendar_rolls_before_expiry(continuous_fixture) -> None:
    _, meta = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "calendar", "none"
    )
    assert len(meta.roll_dates) == len(ORDERED_SYMBOLS) - 1
    expiries = [c.expiry for c in continuous_fixture.specs[:-1]]
    for roll, expiry in zip(meta.roll_dates, expiries, strict=True):
        assert roll < expiry
        # roll is offset business days before expiry (>= offset calendar days back)
        assert (expiry - roll).days >= DEFAULT_CALENDAR_OFFSET_DAYS


def test_calendar_roll_snaps_past_holiday() -> None:
    """When expiry - offset business days lands on a non-trading day, the roll
    snaps back to the last session both contracts trade."""
    # Sessions Mon 2024-03-04 .. Fri 2024-03-15 with 2024-03-11 (the nominal
    # 4-business-day-before-expiry target) removed as a holiday.
    all_days = pd.bdate_range("2024-03-04", "2024-03-15", tz="UTC")
    sessions = pd.DatetimeIndex([d for d in all_days if d.date() != date(2024, 3, 11)])
    close = pd.Series(range(len(sessions)), index=sessions, dtype="float64") + 100.0

    def frame(offset: float) -> pd.DataFrame:
        c = close + offset
        return pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0},
            index=sessions,
        )

    per_contract = {"FRONT": frame(0.0), "BACK": frame(10.0)}
    specs = [
        ContractInfo(symbol="FRONT", expiry=date(2024, 3, 15), first_trade=date(2024, 3, 4)),
        ContractInfo(symbol="BACK", expiry=date(2024, 6, 21), first_trade=date(2024, 3, 4)),
    ]
    _, meta = build_continuous(per_contract, specs, "calendar", "none", calendar_offset_days=4)
    # target 2024-03-11 is a holiday -> snap back to Fri 2024-03-08
    assert meta.roll_dates == [date(2024, 3, 8)]


def test_first_roll_spans_dst_change(continuous_fixture) -> None:
    """The MESH24->MESM24 overlap straddles the 2024-03-10 US DST change."""
    bars, meta = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "volume", "panama_diff"
    )
    first_roll = meta.roll_dates[0]
    assert date(2024, 3, 1) <= first_roll <= date(2024, 3, 15)
    # every timestamp is a tz-aware UTC instant either side of the DST boundary
    assert bars.index.tz is not None
    around_dst = bars.loc["2024-03-08":"2024-03-12"]  # type: ignore[misc]
    assert len(around_dst) >= 2
    assert all(ts.tzname() == "UTC" for ts in bars.index)


def test_panama_removes_contango_jumps(continuous_fixture) -> None:
    """Raw splice shows ~PRICE_STEP jumps at rolls; panama flattens them."""
    raw, meta = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "volume", "none"
    )
    adj, _ = build_continuous(
        continuous_fixture.per_contract, continuous_fixture.specs, "volume", "panama_diff"
    )
    for roll in meta.roll_dates:
        ts = pd.Timestamp(datetime.combine(roll, datetime.min.time()), tz="UTC").replace(hour=21)
        pos = raw.index.get_loc(ts)
        raw_step = abs(raw["close"].iloc[pos] - raw["close"].iloc[pos - 1])
        adj_step = abs(adj["close"].iloc[pos] - adj["close"].iloc[pos - 1])
        assert raw_step > 20.0  # contango gap ~= 25 dominates
        assert adj_step < 6.0  # only the native daily move remains


# --- error handling ----------------------------------------------------------


def test_empty_input_raises() -> None:
    with pytest.raises(DataIntegrityError, match="at least one"):
        build_continuous({}, [], "volume", "none")


def test_missing_spec_raises(continuous_fixture) -> None:
    specs = continuous_fixture.specs[:-1]  # drop a spec still present in per_contract
    with pytest.raises(DataIntegrityError, match="spec"):
        build_continuous(continuous_fixture.per_contract, specs, "volume", "none")


def test_open_interest_rule_requires_column(continuous_fixture) -> None:
    stripped = {
        sym: frame.drop(columns=["open_interest"])
        for sym, frame in continuous_fixture.per_contract.items()
    }
    with pytest.raises(DataIntegrityError, match="open_interest"):
        build_continuous(stripped, continuous_fixture.specs, "open_interest", "none")


def test_single_contract_has_no_rolls(continuous_fixture) -> None:
    only = {"MESH24": continuous_fixture.per_contract["MESH24"]}
    specs = [continuous_fixture.specs[0]]
    bars, meta = build_continuous(only, specs, "volume", "panama_diff")
    assert meta.roll_dates == []
    assert meta.underlying_contracts == ["MESH24"]
    assert len(bars) == len(only["MESH24"])


# --- property tests (hypothesis) --------------------------------------------

_CAL_START = date(2024, 1, 1)


def _bday_index(start: date, end: date) -> pd.DatetimeIndex:
    idx = pd.bdate_range(start, end, tz="UTC")
    return pd.DatetimeIndex(idx.tolist())


def _make_contracts(
    base_increments: list[float], gap: float, *, multiplicative: bool
) -> tuple[dict[str, pd.DataFrame], list[ContractInfo]]:
    """Three overlapping quarterly contracts sharing one base path.

    Each contract sits ``gap`` above the previous (additive) or is scaled by
    ``gap`` (multiplicative) -- a controlled, constant roll gap.
    """
    union = _bday_index(
        _CAL_START, _CAL_START + timedelta(days=len(base_increments) - 1) + timedelta(days=400)
    )
    union = union[: len(base_increments)]
    base = 500.0 + np.cumsum(np.asarray(base_increments, dtype=float))

    expiries = [union[60], union[120], union[180]]
    starts = [union[0], union[40], union[100]]
    symbols = ["C0", "C1", "C2"]

    per_contract: dict[str, pd.DataFrame] = {}
    specs: list[ContractInfo] = []
    for k, sym in enumerate(symbols):
        mask = (union >= starts[k]) & (union <= expiries[k])
        idx = union[mask]
        vals = base[mask]
        close = vals * (gap**k) if multiplicative else vals + gap * k
        per_contract[sym] = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )
        specs.append(
            ContractInfo(symbol=sym, expiry=expiries[k].date(), first_trade=starts[k].date())
        )
    return per_contract, specs


@settings(max_examples=40, deadline=None)
@given(
    increments=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=200,
        max_size=200,
    ),
    gap=st.floats(min_value=5.0, max_value=60.0),
)
def test_panama_zero_jump_property(increments: list[float], gap: float) -> None:
    per_contract, specs = _make_contracts(increments, gap, multiplicative=False)
    adj, meta = build_continuous(per_contract, specs, "calendar", "panama_diff")
    raw, _ = build_continuous(per_contract, specs, "calendar", "none")
    ordered = sorted(specs, key=lambda c: c.expiry)
    for i, roll in enumerate(meta.roll_dates):
        ts = pd.Timestamp(roll, tz="UTC")
        pos = adj.index.get_loc(ts)
        front = per_contract[ordered[i].symbol]
        prev_ts = adj.index[pos - 1]
        native_step = front["close"].loc[ts] - front["close"].loc[prev_ts]
        adj_step = adj["close"].iloc[pos] - adj["close"].iloc[pos - 1]
        raw_step = raw["close"].iloc[pos] - raw["close"].iloc[pos - 1]
        # panama: adjusted step equals the front contract's own move (no gap)
        assert math.isclose(adj_step, native_step, abs_tol=1e-6)
        # raw splice injects exactly the roll gap on top of that move
        assert math.isclose(raw_step - adj_step, gap, abs_tol=1e-6)


@settings(max_examples=40, deadline=None)
@given(
    increments=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=200,
        max_size=200,
    ),
    factor=st.floats(min_value=1.01, max_value=1.5),
)
def test_ratio_preserves_returns_property(increments: list[float], factor: float) -> None:
    per_contract, specs = _make_contracts(increments, factor, multiplicative=True)
    adj, meta = build_continuous(per_contract, specs, "calendar", "ratio")
    ordered = sorted(specs, key=lambda c: c.expiry)
    for i, roll in enumerate(meta.roll_dates):
        ts = pd.Timestamp(roll, tz="UTC")
        pos = adj.index.get_loc(ts)
        front = per_contract[ordered[i].symbol]
        prev_ts = adj.index[pos - 1]
        native_return = front["close"].loc[ts] / front["close"].loc[prev_ts]
        adj_return = adj["close"].iloc[pos] / adj["close"].iloc[pos - 1]
        assert math.isclose(adj_return, native_return, rel_tol=1e-9)
