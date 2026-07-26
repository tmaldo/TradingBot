"""Hand-constructed path tests for triple-barrier labeling (AFML ch.3, G6/G16).

Entry is the close of ``t0``; the profit-taking barrier sits at ``+pt_mult*vol``,
the stop-loss at ``-sl_mult*vol`` (returns), and the vertical barrier at
``t0 + max_holding_bars``. Touches are detected intrabar with high/low. These
tests exercise all six first-touch orderings on explicit OHLC paths, including
the ambiguous same-bar case that must resolve conservatively to the stop, and
pin the recorded ``ret`` and ``t1`` for each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_engine.labels.triple_barrier import fixed_horizon_labels, meta_labels, triple_barrier


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame from ``(open, high, low, close)`` rows."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC"))
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    frame["volume"] = 1000.0
    return frame


# Entry close 100, vol 0.01, pt_mult=sl_mult=2 -> PT price 102, SL price 98.
_ENTRY = 100.0
_VOL = 0.01
_PT_PRICE = 102.0
_SL_PRICE = 98.0


def _vol_series(bars: pd.DataFrame, value: float = _VOL) -> pd.Series:
    return pd.Series(value, index=bars.index)


def _label_one(bars: pd.DataFrame, max_holding: int = 3) -> pd.Series:
    out = triple_barrier(
        bars,
        events=bars.index[[0]],
        pt_mult=2.0,
        sl_mult=2.0,
        max_holding_bars=max_holding,
        vol=_vol_series(bars),
    )
    return out.iloc[0]


def test_profit_take_only() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 103, 101, 102),  # high 103 >= 102 PT; low 101 > 98
            (102, 103, 101, 102),
            (102, 103, 101, 102),
        ]
    )
    row = _label_one(bars)
    assert row["label"] == 1
    assert row["touch"] == "pt"
    assert row["t1"] == bars.index[1]
    assert row["ret"] == pytest.approx(0.02)


def test_stop_loss_only() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (99, 99, 97, 98),  # low 97 <= 98 SL; high 99 < 102
            (98, 99, 97, 98),
            (98, 99, 97, 98),
        ]
    )
    row = _label_one(bars)
    assert row["label"] == -1
    assert row["touch"] == "sl"
    assert row["t1"] == bars.index[1]
    assert row["ret"] == pytest.approx(-0.02)


def test_vertical_time_barrier() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (100, 101, 99, 100),
            (100, 101.5, 98.5, 101),
            (101, 101.8, 99, 100.5),  # never touches 102 or 98
        ]
    )
    row = _label_one(bars)
    assert row["label"] == 0
    assert row["touch"] == "time"
    assert row["t1"] == bars.index[3]
    assert row["ret"] == pytest.approx(100.5 / 100.0 - 1.0)


def test_profit_take_before_stop() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 102.5, 99, 101),  # PT at bar 1
            (101, 101, 97, 98),  # SL only at bar 2 -> too late
            (98, 99, 97, 98),
        ]
    )
    row = _label_one(bars)
    assert row["label"] == 1
    assert row["touch"] == "pt"
    assert row["t1"] == bars.index[1]


def test_stop_before_profit_take() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (99, 101, 97.5, 99),  # SL at bar 1
            (99, 103, 100, 102),  # PT only at bar 2 -> too late
            (102, 103, 101, 102),
        ]
    )
    row = _label_one(bars)
    assert row["label"] == -1
    assert row["touch"] == "sl"
    assert row["t1"] == bars.index[1]


def test_ambiguous_same_bar_resolves_to_stop() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (100, 103, 97, 100),  # both PT (103) and SL (97) in the same bar
            (100, 103, 97, 100),
            (100, 103, 97, 100),
        ]
    )
    row = _label_one(bars)
    assert row["label"] == -1  # conservative: stop first
    assert row["touch"] == "sl"
    assert row["t1"] == bars.index[1]


def test_barriers_scale_with_vol_and_use_only_entry_vol() -> None:
    """A wider vol at entry moves the barriers out; vol after t0 is irrelevant."""
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 103, 101, 102),  # touches PT only if PT price <= 103
            (102, 103, 101, 102),
            (102, 103, 101, 102),
        ]
    )
    # vol at t0 = 0.05 -> PT price = 110 -> no touch -> time barrier.
    wide_vol = pd.Series(0.01, index=bars.index)
    wide_vol.iloc[0] = 0.05
    out = triple_barrier(
        bars, events=bars.index[[0]], pt_mult=2.0, sl_mult=2.0, max_holding_bars=3, vol=wide_vol
    )
    assert out.iloc[0]["label"] == 0
    assert out.iloc[0]["touch"] == "time"


def test_ret_uses_entry_vol_not_later_vol() -> None:
    """Changing vol at bars after t0 must not change the label (trailing-only)."""
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 103, 101, 102),
            (102, 103, 101, 102),
            (102, 103, 101, 102),
        ]
    )
    v1 = pd.Series(0.01, index=bars.index)
    v2 = v1.copy()
    v2.iloc[1:] = 999.0  # absurd future vol
    a = triple_barrier(bars, bars.index[[0]], 2.0, 2.0, 3, v1).iloc[0]
    b = triple_barrier(bars, bars.index[[0]], 2.0, 2.0, 3, v2).iloc[0]
    assert a["label"] == b["label"] == 1
    assert a["ret"] == pytest.approx(b["ret"])


def test_multiple_events_and_frame_shape() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 103, 101, 102),
            (102, 103, 101, 102),
            (100, 100, 97, 98),  # entry for 2nd event: close 98 -> SL price 96.04
            (98, 99, 95, 96),  # low 95 <= 96.04 -> stop
            (96, 97, 95, 96),
        ]
    )
    vol = pd.Series(0.01, index=bars.index)
    out = triple_barrier(bars, bars.index[[0, 3]], 2.0, 2.0, 2, vol)
    assert list(out.columns) == ["t1", "label", "ret", "touch"]
    assert out.index.tolist() == [bars.index[0], bars.index[3]]
    assert out.loc[bars.index[0], "label"] == 1
    assert out.loc[bars.index[3], "label"] == -1


def test_missing_vol_at_event_raises() -> None:
    bars = _bars([(100, 100, 100, 100)] + [(101, 103, 101, 102)] * 3)
    vol = pd.Series(np.nan, index=bars.index)
    with pytest.raises(ValueError, match="vol"):
        triple_barrier(bars, bars.index[[0]], 2.0, 2.0, 3, vol)


def test_event_not_in_bars_raises() -> None:
    bars = _bars([(100, 100, 100, 100)] * 4)
    stray = pd.DatetimeIndex([pd.Timestamp("2030-01-01", tz="UTC")])
    with pytest.raises(ValueError, match="not in bars"):
        triple_barrier(bars, stray, 2.0, 2.0, 3, pd.Series(0.01, index=bars.index))


# --- meta-labeling -----------------------------------------------------------


def _labels_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    return pd.DataFrame(
        {
            "t1": idx,
            "label": [1, -1, 0],
            "ret": [0.02, -0.02, 0.005],
            "touch": ["pt", "sl", "time"],
        },
        index=idx,
    )


def test_meta_labels_reward_correct_side() -> None:
    labels = _labels_frame()
    long_side = pd.Series(1, index=labels.index)
    meta = meta_labels(long_side, labels)
    # long wins on the +0.02 and +0.005 outcomes, loses on the -0.02 one.
    assert meta.tolist() == [1, 0, 1]
    assert set(meta.unique()) <= {0, 1}


def test_meta_labels_short_side() -> None:
    labels = _labels_frame()
    short_side = pd.Series(-1, index=labels.index)
    meta = meta_labels(short_side, labels)
    assert meta.tolist() == [0, 1, 0]


def test_meta_labels_reject_bad_side() -> None:
    labels = _labels_frame()
    with pytest.raises(ValueError, match="primary_side"):
        meta_labels(pd.Series(2, index=labels.index), labels)


# --- fixed-horizon baseline --------------------------------------------------


def test_fixed_horizon_sign_and_time_barrier() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (101, 101, 101, 103),
            (103, 103, 103, 97),
            (97, 97, 97, 100),
        ]
    )
    out = fixed_horizon_labels(bars, bars.index[[0, 1]], horizon_bars=1, tau=0.0)
    assert out.loc[bars.index[0], "label"] == 1  # 100 -> 103 up
    assert out.loc[bars.index[1], "label"] == -1  # 103 -> 97 down
    assert (out["touch"] == "time").all()
    assert out.loc[bars.index[0], "t1"] == bars.index[1]


def test_fixed_horizon_threshold_band() -> None:
    bars = _bars(
        [
            (100, 100, 100, 100),
            (100, 100, 100, 100.5),  # +0.5% move
        ]
    )
    out = fixed_horizon_labels(bars, bars.index[[0]], horizon_bars=1, tau=0.01)
    assert out.loc[bars.index[0], "label"] == 0  # inside +/-1% band
