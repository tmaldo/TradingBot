"""Triple-barrier labeling, meta-labels and uniqueness weights (López de Prado,
AFML ch.3 & ch.4).

Triple-barrier (:func:`triple_barrier`)
---------------------------------------
For each event ``t0`` the entry price is ``close[t0]``. Three barriers bound the
trade, all expressed as *returns* off that entry:

* profit-take at ``+pt_mult * vol[t0]``,
* stop-loss at ``-sl_mult * vol[t0]``,
* a vertical (time) barrier at ``t0 + max_holding_bars`` bars.

``vol`` is a **trailing** volatility estimate supplied by the caller (e.g. an
exp-weighted return std); only the value *at entry* ``vol[t0]`` is used, so the
barriers never depend on future information (G4). Touches are detected *intrabar*
with the bar high (for the profit-take) and low (for the stop) rather than close
only. The first bar to touch wins; when a single bar's range spans **both**
barriers the outcome is ambiguous and resolves **conservatively to the stop**
(AFML's worst-case convention). The label is ``+1`` on a profit-take, ``-1`` on a
stop, and ``0`` on the vertical/time barrier. ``ret`` is the barrier return on a
profit-take/stop (a fill at the barrier) and the realized close-to-entry return
on a time exit. ``touch`` records ``"pt"`` / ``"sl"`` / ``"time"``.

Meta-labels (:func:`meta_labels`)
---------------------------------
Given a primary model's side (``+1`` long / ``-1`` short / ``0`` stand-aside),
the meta-label is ``1`` iff acting on that side would have been profitable
(``primary_side * ret > 0``) and ``0`` otherwise -- the binary "trade / skip"
target of AFML meta-labeling.

Uniqueness weights (:func:`uniqueness_weights`)
-----------------------------------------------
Overlapping label spans share information; a label's weight is its *average
uniqueness* -- the mean of ``1 / concurrency`` over the bars its ``[t0, t1]``
span covers -- so overlapping labels are down-weighted (weights ``< 1``) while
disjoint labels keep weight ``1``. The float Series is directly usable as an
sklearn / LightGBM ``sample_weight``.

Fixed-horizon (:func:`fixed_horizon_labels`)
--------------------------------------------
**Baseline only (G6).** The sign of the ``horizon_bars``-ahead return with an
optional flat dead-band ``tau``. Documented for comparison; the triple-barrier +
meta-labeling path above is the primary labeling scheme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futures_engine.core.types import Bars

Labels = pd.DataFrame
LABEL_COLUMNS: tuple[str, ...] = ("t1", "label", "ret", "touch")

_VALID_SIDES = frozenset({-1, 0, 1})


def _positions(bars: Bars) -> dict[pd.Timestamp, int]:
    return {ts: i for i, ts in enumerate(bars.index)}


def triple_barrier(
    bars: Bars,
    events: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    max_holding_bars: int,
    vol: pd.Series,
) -> Labels:
    """Label each event in ``events`` by first-barrier touch (see module docstring)."""
    if max_holding_bars < 1:
        raise ValueError(f"max_holding_bars must be >= 1, got {max_holding_bars}")
    if pt_mult < 0 or sl_mult < 0:
        raise ValueError("pt_mult and sl_mult must be non-negative")

    pos_of = _positions(bars)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    n = len(bars.index)

    t0_out: list[pd.Timestamp] = []
    t1_out: list[pd.Timestamp] = []
    label_out: list[int] = []
    ret_out: list[float] = []
    touch_out: list[str] = []

    for t0 in events:
        if t0 not in pos_of:
            raise ValueError(f"event timestamp not in bars index: {t0!r}")
        v = vol.get(t0, np.nan)
        if pd.isna(v):
            raise ValueError(f"vol is missing/NaN at event {t0!r}; supply a trailing estimate")

        i0 = pos_of[t0]
        entry = close[i0]
        pt_level = pt_mult * float(v)
        sl_level = -sl_mult * float(v)
        i_vert = min(i0 + max_holding_bars, n - 1)

        touch: str | None = None
        i_touch = i_vert
        ret = 0.0
        for b in range(i0 + 1, i_vert + 1):
            lo_ret = low[b] / entry - 1.0
            hi_ret = high[b] / entry - 1.0
            # Stop checked first so an ambiguous both-touch bar resolves to the stop.
            if lo_ret <= sl_level:
                touch, i_touch, ret = "sl", b, sl_level
                break
            if hi_ret >= pt_level:
                touch, i_touch, ret = "pt", b, pt_level
                break

        if touch is None:
            touch = "time"
            ret = close[i_vert] / entry - 1.0
            label = 0
        else:
            label = 1 if touch == "pt" else -1

        t0_out.append(t0)
        t1_out.append(bars.index[i_touch])
        label_out.append(label)
        ret_out.append(ret)
        touch_out.append(touch)

    out = pd.DataFrame(
        {
            "t1": pd.DatetimeIndex(t1_out),
            "label": np.asarray(label_out, dtype=np.int64),
            "ret": np.asarray(ret_out, dtype=np.float64),
            "touch": touch_out,
        },
        index=pd.DatetimeIndex(t0_out, name=bars.index.name),
    )
    return out[list(LABEL_COLUMNS)]


def meta_labels(primary_side: pd.Series, labels: Labels) -> pd.Series:
    """Binary ``{0, 1}`` meta-label: 1 iff ``primary_side`` would have profited."""
    side = primary_side.reindex(labels.index)
    if side.isna().any():
        raise ValueError("primary_side is missing a value for at least one label event")
    if not side.isin(list(_VALID_SIDES)).all():
        raise ValueError("primary_side values must be in {-1, 0, 1}")
    pnl = side.astype(np.float64) * labels["ret"].astype(np.float64)
    return (pnl > 0.0).astype(np.int64)


def uniqueness_weights(labels: Labels, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Average-uniqueness sample weights in ``(0, 1]`` aligned to ``labels.index``."""
    if not bar_index.is_monotonic_increasing:
        raise ValueError("bar_index must be sorted ascending")
    m = len(bar_index)
    concurrency = np.zeros(m, dtype=np.float64)

    starts = np.empty(len(labels), dtype=np.int64)
    ends = np.empty(len(labels), dtype=np.int64)
    for i, (t0, t1) in enumerate(zip(labels.index, labels["t1"], strict=True)):
        start = int(bar_index.searchsorted(t0, side="left"))
        end = int(bar_index.searchsorted(t1, side="right")) - 1
        if start > end or start < 0 or end >= m:
            raise ValueError(f"label span [{t0}, {t1}] is not covered by bar_index")
        starts[i] = start
        ends[i] = end
        concurrency[start : end + 1] += 1.0

    weights = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        weights[i] = float(np.mean(1.0 / concurrency[starts[i] : ends[i] + 1]))
    return pd.Series(weights, index=labels.index)


def fixed_horizon_labels(
    bars: Bars,
    events: pd.DatetimeIndex,
    horizon_bars: int,
    tau: float = 0.0,
) -> Labels:
    """Baseline-only (G6): sign of the ``horizon_bars``-ahead return, dead-band ``tau``.

    Returns a :data:`Labels`-shaped frame with ``touch == "time"`` for every row.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    pos_of = _positions(bars)
    close = bars["close"].to_numpy(dtype=np.float64)
    n = len(bars.index)

    t0_out: list[pd.Timestamp] = []
    t1_out: list[pd.Timestamp] = []
    label_out: list[int] = []
    ret_out: list[float] = []
    for t0 in events:
        if t0 not in pos_of:
            raise ValueError(f"event timestamp not in bars index: {t0!r}")
        i0 = pos_of[t0]
        i1 = min(i0 + horizon_bars, n - 1)
        ret = close[i1] / close[i0] - 1.0
        label = 1 if ret > tau else (-1 if ret < -tau else 0)
        t0_out.append(t0)
        t1_out.append(bars.index[i1])
        label_out.append(label)
        ret_out.append(ret)

    out = pd.DataFrame(
        {
            "t1": pd.DatetimeIndex(t1_out),
            "label": np.asarray(label_out, dtype=np.int64),
            "ret": np.asarray(ret_out, dtype=np.float64),
            "touch": ["time"] * len(t0_out),
        },
        index=pd.DatetimeIndex(t0_out, name=bars.index.name),
    )
    return out[list(LABEL_COLUMNS)]
