"""Technical indicators computed with plain pandas (ported from the legacy
``stock_researcher/indicators.py``).

All functions take pandas Series indexed by (UTC) timestamp and return a Series
aligned to the same index; ``macd`` and ``bollinger`` return a DataFrame. Every
function is **causal** (backward-looking only): a value at time ``t`` uses bars
at or before ``t``. This is what lets :mod:`futures_engine.features.builder`
register them with the point-in-time audit (G4).

The single deliberate divergence from the legacy library is
:func:`rolling_volatility`: its annualization factor is now an explicit argument
rather than a hardcoded ``sqrt(252)``, so intraday intervals annualize correctly
(the factor is derived from the bar interval by the feature builder).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` bars."""
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the given ``span`` (recursive, no warm-up bias)."""
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in ``[0, 100]``."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # All-gain windows (avg_loss == 0) mean RSI = 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram (columns ``macd, signal, histogram``)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands and %B (columns ``middle, upper, lower, percent_b``)."""
    middle = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_width = (upper - lower).replace(0.0, np.nan)
    percent_b = (close - lower) / band_width
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower, "percent_b": percent_b})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume (running signed-volume accumulation)."""
    diff = close.diff()
    direction = pd.Series(np.sign(diff.to_numpy(dtype="float64")), index=close.index).fillna(0.0)
    out: pd.Series = (direction * volume).cumsum()
    return out


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """%K of the stochastic oscillator, in ``[0, 100]``."""
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    span = (highest - lowest).replace(0.0, np.nan)
    k: pd.Series = 100.0 * (close - lowest) / span
    return k


def rolling_volatility(close: pd.Series, window: int, annualization: float) -> pd.Series:
    """Annualized volatility of log returns over ``window`` bars.

    ``annualization`` is the number of bars per year for the series' sampling
    interval (e.g. 252 for daily bars); the standard deviation of per-bar log
    returns is scaled by ``sqrt(annualization)``. Passing the interval-derived
    factor (instead of a hardcoded ``sqrt(252)``) keeps intraday vol correct.
    """
    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
    std = log_ret.rolling(window, min_periods=window).std()
    return std * math.sqrt(annualization)
