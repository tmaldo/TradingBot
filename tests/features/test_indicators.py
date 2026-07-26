"""Port-guard tests for the salvaged legacy indicator library.

Each test recomputes the *legacy* formula inline (copied verbatim from
``stock_researcher/indicators.py``) and asserts the port reproduces it on the
shared fixture -- so a future refactor that silently changes a formula fails
here. Behavioural assertions from the legacy test-suite (bounds, direction,
column names) are kept too. The one deliberate divergence is
``rolling_volatility``, which is now intraday-aware: the annualization factor is
an argument instead of a hardcoded ``sqrt(252)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_engine.features import indicators as ind


def test_sma_matches_manual() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_matches_legacy_formula(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    legacy = close.rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(ind.sma(close, 20), legacy, check_names=False)


def test_ema_matches_legacy_formula(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    legacy = close.ewm(span=12, adjust=False).mean()
    pd.testing.assert_series_equal(ind.ema(close, 12), legacy, check_names=False)


def test_rsi_matches_legacy_formula(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    window = 14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    legacy = 100 - 100 / (1 + rs)
    legacy = legacy.where(avg_loss != 0.0, 100.0)
    legacy[avg_gain.isna() | avg_loss.isna()] = np.nan
    pd.testing.assert_series_equal(ind.rsi(close, window), legacy, check_names=False)


def test_rsi_bounds_and_direction(bars_fixture: pd.DataFrame) -> None:
    r = ind.rsi(bars_fixture["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    rising = pd.Series(np.linspace(1, 100, 60))
    assert ind.rsi(rising).iloc[-1] == pytest.approx(100.0)
    falling = pd.Series(np.linspace(100, 1, 60))
    assert ind.rsi(falling).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_macd_columns_and_legacy_formula(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    m = ind.macd(close)
    assert list(m.columns) == ["macd", "signal", "histogram"]
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    pd.testing.assert_series_equal(m["macd"], macd_line, check_names=False)
    pd.testing.assert_series_equal(m["signal"], signal_line, check_names=False)
    pd.testing.assert_series_equal(m["histogram"], macd_line - signal_line, check_names=False)


def test_bollinger_legacy_formula_and_bounds(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    b = ind.bollinger(close)
    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    upper = middle + 2.0 * std
    lower = middle - 2.0 * std
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    pd.testing.assert_series_equal(b["middle"], middle, check_names=False)
    pd.testing.assert_series_equal(b["upper"], upper, check_names=False)
    pd.testing.assert_series_equal(b["lower"], lower, check_names=False)
    pd.testing.assert_series_equal(b["percent_b"], percent_b, check_names=False)
    valid = b.dropna()
    assert (valid["upper"] >= valid["lower"]).all()


def test_atr_matches_legacy_and_positive(bars_fixture: pd.DataFrame) -> None:
    high, low, close = bars_fixture["high"], bars_fixture["low"], bars_fixture["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    legacy = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    got = ind.atr(high, low, close, 14)
    pd.testing.assert_series_equal(got, legacy, check_names=False)
    assert (got.dropna() > 0).all()


def test_stochastic_k_legacy_and_bounds(bars_fixture: pd.DataFrame) -> None:
    high, low, close = bars_fixture["high"], bars_fixture["low"], bars_fixture["close"]
    lowest = low.rolling(14, min_periods=14).min()
    highest = high.rolling(14, min_periods=14).max()
    legacy = 100 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    got = ind.stochastic_k(high, low, close, 14)
    pd.testing.assert_series_equal(got, legacy, check_names=False)
    k = got.dropna()
    assert ((k >= 0) & (k <= 100)).all()


def test_obv_accumulates() -> None:
    close = pd.Series([10.0, 11.0, 10.5, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    out = ind.obv(close, volume)
    assert out.iloc[-1] == pytest.approx(200 - 300 + 400)


def test_rolling_volatility_matches_legacy_when_daily(bars_fixture: pd.DataFrame) -> None:
    close = bars_fixture["close"]
    log_ret = np.log(close / close.shift(1))
    legacy = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    got = ind.rolling_volatility(close, 20, annualization=252.0)
    pd.testing.assert_series_equal(got, legacy, check_names=False)


def test_rolling_volatility_annualization_scales_with_factor(bars_fixture: pd.DataFrame) -> None:
    """Intraday-awareness: the factor is an argument, not a hardcoded sqrt(252)."""
    close = bars_fixture["close"]
    daily = ind.rolling_volatility(close, 20, annualization=252.0).dropna()
    hourly = ind.rolling_volatility(close, 20, annualization=252.0 * 23.0).dropna()
    ratio = (hourly / daily).dropna()
    assert np.allclose(ratio, np.sqrt(23.0))
