"""Tests for the reference trend/momentum signal families (G7).

Two families make up the triage battery: a Donchian-channel breakout and an
MA-cross with a volatility-target filter. Both declare ``family ==
"trend_momentum"``, read *every* parameter from the passed ``params`` dict (no
magic constants, G15), and return a float target-position series in ``[-1, 1]``
indexed exactly like ``bars`` (the RAW, pre-causal-shift decision as of each
bar's close).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from futures_engine.research.strategies.momentum import DonchianBreakout, MACrossVolTarget


def _bars_from_close(closes: list[float]) -> pd.DataFrame:
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i) for i in range(len(closes))])
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(len(closes), 1000.0),
        },
        index=idx,
    )


# --- Donchian breakout -------------------------------------------------------


def test_donchian_family_is_trend_momentum() -> None:
    assert DonchianBreakout().family == "trend_momentum"


def test_donchian_breakout_signal_sequence() -> None:
    # window=3. Upside break at t4 (close jumps to 20 over a 10-channel), hold
    # through t5/t6, downside break at t7 (close drops to 5), hold to the end.
    closes = [10, 10, 10, 10, 20, 20, 20, 5, 5, 5]
    bars = _bars_from_close([float(c) for c in closes])
    signal = DonchianBreakout().generate(bars, {"window": 3})
    expected = pd.Series(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], index=bars.index
    )
    pd.testing.assert_series_equal(signal, expected, check_names=False)


def test_donchian_position_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    bars = _bars_from_close(list(100.0 + rng.standard_normal(200).cumsum()))
    signal = DonchianBreakout().generate(bars, {"window": 20})
    assert signal.between(-1.0, 1.0).all()
    assert signal.index.equals(bars.index)


def test_donchian_requires_window_param() -> None:
    bars = _bars_from_close([1.0, 2.0, 3.0])
    with pytest.raises((KeyError, ValueError)):
        DonchianBreakout().generate(bars, {})


# --- MA-cross with vol-target filter -----------------------------------------


def test_ma_cross_family_is_trend_momentum() -> None:
    assert MACrossVolTarget().family == "trend_momentum"


def _uptrend_bars() -> pd.DataFrame:
    return _bars_from_close([100.0 + i for i in range(40)])


def _downtrend_bars() -> pd.DataFrame:
    return _bars_from_close([140.0 - i for i in range(40)])


def test_ma_cross_long_when_fast_above_slow() -> None:
    bars = _uptrend_bars()
    # huge target_vol -> vol scale clips to 1, so position is the pure direction.
    params = {"fast": 3, "slow": 10, "vol_window": 5, "target_vol": 1e9}
    signal = MACrossVolTarget().generate(bars, params)
    assert (signal.iloc[-5:] == 1.0).all()


def test_ma_cross_short_when_fast_below_slow() -> None:
    bars = _downtrend_bars()
    params = {"fast": 3, "slow": 10, "vol_window": 5, "target_vol": 1e9}
    signal = MACrossVolTarget().generate(bars, params)
    assert (signal.iloc[-5:] == -1.0).all()


def test_ma_cross_vol_target_scales_position_down() -> None:
    bars = _uptrend_bars()
    base = {"fast": 3, "slow": 10, "vol_window": 5}
    full = MACrossVolTarget().generate(bars, {**base, "target_vol": 1e9})
    tiny = MACrossVolTarget().generate(bars, {**base, "target_vol": 1e-9})
    active = full != 0.0
    # a tiny vol target must shrink every active position below its full-size value.
    assert (tiny.abs()[active] < full.abs()[active]).all()
    assert tiny.between(-1.0, 1.0).all()


def test_ma_cross_position_in_unit_range() -> None:
    rng = np.random.default_rng(1)
    bars = _bars_from_close(list(100.0 + rng.standard_normal(300).cumsum()))
    params = {"fast": 5, "slow": 20, "vol_window": 10, "target_vol": 0.005}
    signal = MACrossVolTarget().generate(bars, params)
    assert signal.between(-1.0, 1.0).all()
    assert signal.index.equals(bars.index)


def test_ma_cross_requires_all_params() -> None:
    bars = _uptrend_bars()
    with pytest.raises((KeyError, ValueError)):
        MACrossVolTarget().generate(bars, {"fast": 3, "slow": 10})
