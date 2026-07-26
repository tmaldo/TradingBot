"""Reference trend/momentum signal families (G7) for the triage battery.

Two families, both ``family == "trend_momentum"``:

* :class:`DonchianBreakout` -- long on a close above the prior ``window``-bar
  high channel, short on a close below the prior ``window``-bar low channel,
  holding the last state between breakouts (a classic stop-and-reverse trend
  follower).
* :class:`MACrossVolTarget` -- a fast/slow moving-average crossover whose signed
  direction is scaled by a volatility-target filter: the position size is
  ``clip(target_vol / realized_vol, 0, 1)``, so the same edge is down-weighted in
  turbulent regimes (a float position in ``[-1, 1]``).

Every numeric parameter is read from the ``params`` dict handed to
:meth:`generate`; missing keys raise (no magic constants baked into signal code,
G15). ``generate`` returns the RAW target position as of each bar's close -- the
harness applies the single causal shift (:func:`research.harness.causal_positions`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from futures_engine.core.types import Bars

_FAMILY = "trend_momentum"


def _require_int(params: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    if key not in params:
        raise KeyError(f"{key!r} is a required strategy parameter (no default; G15)")
    value = params[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key!r} must be an int >= {minimum}, got {value!r}")
    return value


def _require_float(params: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    if key not in params:
        raise KeyError(f"{key!r} is a required strategy parameter (no default; G15)")
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"{key!r} must be a number >= {minimum}, got {value!r}")
    return float(value)


class DonchianBreakout:
    """Donchian-channel breakout (window param); holds state between breakouts."""

    family = _FAMILY

    def generate(self, bars: Bars, params: dict[str, Any]) -> pd.Series:
        window = _require_int(params, "window", minimum=1)
        close = bars["close"]
        upper = bars["high"].rolling(window).max().shift(1)
        lower = bars["low"].rolling(window).min().shift(1)
        signal = pd.Series(np.nan, index=bars.index, dtype=float)
        signal[close > upper] = 1.0
        signal[close < lower] = -1.0
        # Hold the last breakout state between signals; flat before the first.
        return signal.ffill().fillna(0.0)


class MACrossVolTarget:
    """Fast/slow MA crossover, size-scaled by a volatility-target filter."""

    family = _FAMILY

    def generate(self, bars: Bars, params: dict[str, Any]) -> pd.Series:
        fast = _require_int(params, "fast", minimum=1)
        slow = _require_int(params, "slow", minimum=2)
        vol_window = _require_int(params, "vol_window", minimum=2)
        target_vol = _require_float(params, "target_vol", minimum=0.0)
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")

        close = bars["close"]
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        direction = pd.Series(np.sign(fast_ma - slow_ma), index=bars.index)

        realized_vol = close.pct_change().rolling(vol_window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (target_vol / realized_vol).clip(lower=0.0, upper=1.0)
        position = direction * scale
        return position.fillna(0.0)
