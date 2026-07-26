"""Feature matrix builder and point-in-time (look-ahead) registry (task T4, G4/G7).

``build_features`` turns an OHLCV :class:`~futures_engine.core.types.Bars` frame
into the model-ready feature matrix whose columns are exactly
:data:`FEATURE_COLUMNS`: the legacy 15 (returns, RSI, MACD-histogram norm,
Bollinger %B, ATR ratio, realized vol, volume z-score, SMA distances, rolling
extreme distances, stochastic %K) plus a fractionally-differenced close. Every
column is a **trend/momentum-oriented, causal** transform (no microstructure/HF
features, G7); each is registered with the look-ahead audit by :func:`register_all`
so CI shift-tests the whole set (G4).

All parameters -- windows, spans, the fractional-differencing order, and the
trading calendar used to annualize volatility -- live on the frozen pydantic
:class:`FeatureConfig`; there are no magic constants in the feature code (G15).
The volatility annualization factor is *derived from the bar interval* via
:func:`periods_per_year`, so intraday intervals annualize correctly rather than
assuming 252 daily bars.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.core.types import BAR_COLUMNS, BarInterval, Bars, InstrumentSpec
from futures_engine.data.audit import register_pit_check, registered_checks
from futures_engine.features import indicators as ind
from futures_engine.features.fracdiff import frac_diff

# A single feature: a causal map from a bar history to one aligned Series.
FeatureFn = Callable[[Bars], pd.Series]

# Explicit, reviewable list of every registered feature (order = matrix column
# order). A test asserts this equals what the default FeatureConfig builds.
FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "rsi",
    "macd_hist_norm",
    "bb_percent_b",
    "atr_ratio",
    "realized_vol",
    "volume_z",
    "dist_sma_fast",
    "dist_sma_slow",
    "dist_high_extreme",
    "dist_low_extreme",
    "stoch_k",
    "fracdiff_close",
)

# Prefix under which feature checks are registered in the shared audit registry.
_CHECK_PREFIX = "feature."

# Minutes represented by each intraday bar interval (daily is handled separately).
_INTERVAL_MINUTES: dict[BarInterval, float] = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "1h": 60.0}


class FeatureConfig(BaseModel):
    """Windows, spans and calendar for the feature set (all values, no literals)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    interval: BarInterval = "1d"
    return_horizons: tuple[int, ...] = (1, 5, 10, 21)
    rsi_window: int = Field(default=14, gt=0)
    macd_fast: int = Field(default=12, gt=0)
    macd_slow: int = Field(default=26, gt=0)
    macd_signal: int = Field(default=9, gt=0)
    bollinger_window: int = Field(default=20, gt=0)
    bollinger_num_std: float = Field(default=2.0, gt=0)
    atr_window: int = Field(default=14, gt=0)
    vol_window: int = Field(default=20, gt=0)
    volume_z_window: int = Field(default=20, gt=0)
    sma_fast_window: int = Field(default=50, gt=0)
    sma_slow_window: int = Field(default=200, gt=0)
    extreme_window: int = Field(default=252, gt=0)
    extreme_min_periods: int = Field(default=60, gt=0)
    stoch_window: int = Field(default=14, gt=0)
    fracdiff_d: float = Field(default=0.4, ge=0.0)
    fracdiff_threshold: float = Field(default=1e-4, gt=0.0)
    trading_days_per_year: float = Field(default=252.0, gt=0)
    session_hours: float = Field(default=23.0, gt=0)


def periods_per_year(
    interval: BarInterval, trading_days_per_year: float, session_hours: float
) -> float:
    """Number of bars of ``interval`` in one trading year (the annualization base).

    Daily bars give ``trading_days_per_year``; intraday intervals scale that by
    the number of bars per session (``session_hours`` of trading per day).
    """
    if interval == "1d":
        return trading_days_per_year
    bars_per_session = session_hours * 60.0 / _INTERVAL_MINUTES[interval]
    return bars_per_session * trading_days_per_year


# --- individual feature functions (each causal: value at t uses bars <= t) ----


def _ret(bars: Bars, *, horizon: int) -> pd.Series:
    return bars["close"].pct_change(horizon, fill_method=None)


def _rsi(bars: Bars, *, window: int) -> pd.Series:
    return ind.rsi(bars["close"], window) / 100.0


def _macd_hist_norm(bars: Bars, *, fast: int, slow: int, signal: int) -> pd.Series:
    close = bars["close"]
    hist = ind.macd(close, fast=fast, slow=slow, signal=signal)["histogram"]
    out: pd.Series = hist / close
    return out


def _bb_percent_b(bars: Bars, *, window: int, num_std: float) -> pd.Series:
    return ind.bollinger(bars["close"], window=window, num_std=num_std)["percent_b"]


def _atr_ratio(bars: Bars, *, window: int) -> pd.Series:
    atr = ind.atr(bars["high"], bars["low"], bars["close"], window)
    out: pd.Series = atr / bars["close"]
    return out


def _realized_vol(bars: Bars, *, window: int, annualization: float) -> pd.Series:
    return ind.rolling_volatility(bars["close"], window, annualization)


def _volume_z(bars: Bars, *, window: int) -> pd.Series:
    volume = bars["volume"]
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    out: pd.Series = (volume - mean) / std
    return out


def _dist_sma(bars: Bars, *, window: int) -> pd.Series:
    out: pd.Series = bars["close"] / ind.sma(bars["close"], window) - 1.0
    return out


def _dist_high_extreme(bars: Bars, *, window: int, min_periods: int) -> pd.Series:
    high = bars["close"].rolling(window, min_periods=min_periods).max()
    out: pd.Series = bars["close"] / high - 1.0
    return out


def _dist_low_extreme(bars: Bars, *, window: int, min_periods: int) -> pd.Series:
    low = bars["close"].rolling(window, min_periods=min_periods).min()
    out: pd.Series = bars["close"] / low - 1.0
    return out


def _stoch_k(bars: Bars, *, window: int) -> pd.Series:
    return ind.stochastic_k(bars["high"], bars["low"], bars["close"], window) / 100.0


def _fracdiff_close(bars: Bars, *, d: float, threshold: float) -> pd.Series:
    return frac_diff(bars["close"], d, threshold)


def feature_functions(config: FeatureConfig) -> dict[str, FeatureFn]:
    """Return the ordered ``name -> causal feature function`` mapping for ``config``.

    Each value is a single-argument callable ``fn(bars) -> Series`` with the
    config parameters bound, suitable both for building the matrix and for
    registering with the point-in-time audit.
    """
    ann = periods_per_year(config.interval, config.trading_days_per_year, config.session_hours)
    fns: dict[str, FeatureFn] = {}
    for horizon in config.return_horizons:
        fns[f"ret_{horizon}"] = partial(_ret, horizon=horizon)
    fns["rsi"] = partial(_rsi, window=config.rsi_window)
    fns["macd_hist_norm"] = partial(
        _macd_hist_norm, fast=config.macd_fast, slow=config.macd_slow, signal=config.macd_signal
    )
    fns["bb_percent_b"] = partial(
        _bb_percent_b, window=config.bollinger_window, num_std=config.bollinger_num_std
    )
    fns["atr_ratio"] = partial(_atr_ratio, window=config.atr_window)
    fns["realized_vol"] = partial(_realized_vol, window=config.vol_window, annualization=ann)
    fns["volume_z"] = partial(_volume_z, window=config.volume_z_window)
    fns["dist_sma_fast"] = partial(_dist_sma, window=config.sma_fast_window)
    fns["dist_sma_slow"] = partial(_dist_sma, window=config.sma_slow_window)
    fns["dist_high_extreme"] = partial(
        _dist_high_extreme, window=config.extreme_window, min_periods=config.extreme_min_periods
    )
    fns["dist_low_extreme"] = partial(
        _dist_low_extreme, window=config.extreme_window, min_periods=config.extreme_min_periods
    )
    fns["stoch_k"] = partial(_stoch_k, window=config.stoch_window)
    fns["fracdiff_close"] = partial(
        _fracdiff_close, d=config.fracdiff_d, threshold=config.fracdiff_threshold
    )
    return fns


def _require_bar_columns(bars: Bars, spec: InstrumentSpec) -> None:
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(
            f"bars for {spec.symbol_root!r} missing required column(s): {', '.join(missing)}"
        )


def build_features(bars: Bars, spec: InstrumentSpec, config: FeatureConfig) -> pd.DataFrame:
    """Build the feature matrix for ``bars``.

    Columns are the features produced by ``config`` (``FEATURE_COLUMNS`` under the
    default config), aligned to ``bars.index``; warm-up rows without a full
    look-back window contain ``NaN`` (callers decide how to drop them).
    """
    _require_bar_columns(bars, spec)
    fns = feature_functions(config)
    data = {name: fn(bars) for name, fn in fns.items()}
    return pd.DataFrame(data, index=bars.index)[list(fns)]


def register_all(config: FeatureConfig | None = None) -> None:
    """Register every feature's point-in-time check with the look-ahead audit.

    Idempotent: checks already present are skipped, so importing the features
    package (which calls this) and an explicit call from tests/CI compose safely.
    Checks are namespaced ``feature.<column>``.
    """
    cfg = config or FeatureConfig()
    existing = registered_checks()
    for name, fn in feature_functions(cfg).items():
        check_name = f"{_CHECK_PREFIX}{name}"
        if check_name not in existing:
            register_pit_check(check_name, fn)
