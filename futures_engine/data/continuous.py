"""Explicit continuous-contract stitching (Global Constraint G3).

:func:`build_continuous` splices a set of per-contract bar frames into one
continuous series with a fully explicit, recorded method: the roll *rule*
(``volume`` / ``open_interest`` / ``calendar``) decides *when* to switch from
the front to the back contract, and the *adjustment* (``panama_diff`` / ``ratio``
/ ``none``) decides how prior contracts are back-adjusted so the splice has no
artificial gap. Every roll date and both underlying contract symbols are recorded
in the returned :class:`~futures_engine.core.types.ContinuousMeta`.

Roll-rule semantics
-------------------
* ``volume`` / ``open_interest`` -- over the sessions where the front and back
  contracts both trade, roll on the first session on which the back contract's
  volume (resp. open interest) has exceeded the front's for ``confirm_days``
  consecutive sessions (crossover plus confirmation; no single-day spike rolls).
* ``calendar`` -- roll ``calendar_offset_days`` business days before the front
  contract's expiry, snapped back to the last session both contracts trade
  (handles holiday/weekend gaps).

Adjustment semantics
--------------------
Let ``gap_i = back_close(r_i) - front_close(r_i)`` at roll ``r_i`` (both
contracts trade on ``r_i``). The newest contract is left unadjusted; each older
segment ``j`` is shifted by the cumulative gap of the rolls at or after it:

* ``panama_diff`` -- add ``sum_{i>=j} gap_i`` to OHLC (difference back-adjust);
* ``ratio`` -- multiply OHLC by ``prod_{i>=j} back_close(r_i)/front_close(r_i)``
  (multiplicative back-adjust; preserves returns across rolls);
* ``none`` -- raw splice (segments concatenated unchanged; the gap is visible).

Volume / open interest are never adjusted. This constructs a historical
continuous *price* series; it is not itself a point-in-time feature (those are
audited separately in :mod:`futures_engine.data.audit`).
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import pandas as pd

from futures_engine.core.types import BAR_COLUMNS, Bars, ContinuousMeta
from futures_engine.data.provider import ContractInfo
from futures_engine.data.store import DataIntegrityError

RollRule = Literal["volume", "open_interest", "calendar"]
Adjustment = Literal["panama_diff", "ratio", "none"]

# Business days before expiry at which the calendar rule rolls (configurable).
DEFAULT_CALENDAR_OFFSET_DAYS = 4
# Consecutive sessions the back contract must lead before a volume/OI roll
# commits: a crossover day plus one confirming day (brief: "1-day confirmation").
DEFAULT_ROLL_CONFIRM_DAYS = 2

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
_ROLL_COLUMN: dict[RollRule, str] = {"volume": "volume", "open_interest": "open_interest"}


def _ordered_contracts(
    per_contract: dict[str, Bars], specs: list[ContractInfo]
) -> list[tuple[ContractInfo, Bars]]:
    if not per_contract:
        raise DataIntegrityError("build_continuous requires at least one contract")
    by_symbol = {c.symbol: c for c in specs}
    missing = [sym for sym in per_contract if sym not in by_symbol]
    if missing:
        raise DataIntegrityError(f"no ContractInfo spec for contract(s): {missing}")
    ordered: list[tuple[ContractInfo, Bars]] = []
    for sym in per_contract:
        frame = per_contract[sym]
        if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
            raise DataIntegrityError(f"contract {sym!r} bars must have a UTC DatetimeIndex")
        if "close" not in frame.columns or "volume" not in frame.columns:
            raise DataIntegrityError(f"contract {sym!r} bars need close and volume columns")
        ordered.append((by_symbol[sym], frame.sort_index()))
    ordered.sort(key=lambda pair: pair[0].expiry)
    return ordered


def _market_share_roll(front: Bars, back: Bars, column: str, confirm_days: int) -> pd.Timestamp:
    if column not in front.columns or column not in back.columns:
        raise DataIntegrityError(f"roll rule needs a {column!r} column on both contracts")
    common = front.index.intersection(back.index)
    if len(common) == 0:
        raise DataIntegrityError("front and back contracts have no overlapping sessions")
    lead = (back.loc[common, column] > front.loc[common, column]).to_numpy()
    run = 0
    for pos in range(len(lead)):
        run = run + 1 if lead[pos] else 0
        if run >= confirm_days:
            return pd.Timestamp(common[pos])
    raise DataIntegrityError(
        f"back contract volume/OI never led for {confirm_days} consecutive sessions"
    )


def _calendar_roll(front: Bars, back: Bars, expiry: date, offset_days: int) -> pd.Timestamp:
    common = cast(pd.DatetimeIndex, front.index.intersection(back.index))
    if len(common) == 0:
        raise DataIntegrityError("front and back contracts have no overlapping sessions")
    # Nominal target: offset_days Mon-Fri business days before expiry; then snap
    # to the last session both contracts actually trade (handles holidays).
    target = (pd.Timestamp(expiry, tz="UTC") - offset_days * pd.offsets.BDay()).normalize()
    candidates = common[common.normalize() <= target]
    if len(candidates) == 0:
        raise DataIntegrityError("no overlapping session on or before the calendar roll target")
    return pd.Timestamp(candidates[-1])


def _roll_dates(
    ordered: list[tuple[ContractInfo, Bars]],
    roll_rule: RollRule,
    confirm_days: int,
    calendar_offset_days: int,
) -> list[pd.Timestamp]:
    rolls: list[pd.Timestamp] = []
    for (front_spec, front), (_back_spec, back) in pairwise(ordered):
        if roll_rule == "calendar":
            r = _calendar_roll(front, back, front_spec.expiry, calendar_offset_days)
        else:
            r = _market_share_roll(front, back, _ROLL_COLUMN[roll_rule], confirm_days)
        if rolls and r <= rolls[-1]:
            raise DataIntegrityError("roll dates are not strictly increasing across contracts")
        rolls.append(r)
    return rolls


def _segment(frame: Bars, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> Bars:
    mask = pd.Series(True, index=frame.index)
    if lo is not None:
        mask &= frame.index >= lo
    if hi is not None:
        mask &= frame.index < hi
    return frame.loc[mask]


def _adjustments(
    ordered: list[tuple[ContractInfo, Bars]],
    rolls: list[pd.Timestamp],
    adjustment: Adjustment,
) -> tuple[list[float], list[float]]:
    """Return per-segment additive offsets and multiplicative factors."""
    n = len(ordered)
    diffs: list[float] = []
    ratios: list[float] = []
    for i, r in enumerate(rolls):
        front_close = float(cast(float, ordered[i][1].loc[r, "close"]))
        back_close = float(cast(float, ordered[i + 1][1].loc[r, "close"]))
        diffs.append(back_close - front_close)
        if front_close == 0.0:
            raise DataIntegrityError("cannot ratio-adjust across a zero front price")
        ratios.append(back_close / front_close)
    offsets = [0.0] * n
    factors = [1.0] * n
    for j in range(n - 1):
        if adjustment == "panama_diff":
            offsets[j] = float(np.sum(diffs[j:]))
        elif adjustment == "ratio":
            factors[j] = float(np.prod(ratios[j:]))
    return offsets, factors


def build_continuous(
    per_contract: dict[str, Bars],
    specs: list[ContractInfo],
    roll_rule: RollRule,
    adjustment: Adjustment,
    *,
    calendar_offset_days: int = DEFAULT_CALENDAR_OFFSET_DAYS,
    confirm_days: int = DEFAULT_ROLL_CONFIRM_DAYS,
) -> tuple[Bars, ContinuousMeta]:
    """Stitch ``per_contract`` bars into one continuous series (see module docstring).

    Returns the continuous ``Bars`` (columns ``BAR_COLUMNS``, UTC index) and the
    :class:`ContinuousMeta` recording the rule, adjustment, roll dates, and the
    ordered underlying contract symbols. Raises :class:`DataIntegrityError` on any
    structural problem (missing spec, no overlap, non-increasing rolls, empty
    segment, missing roll column).
    """
    if roll_rule == "open_interest":
        for sym, frame in per_contract.items():
            if "open_interest" not in frame.columns:
                raise DataIntegrityError(f"contract {sym!r} lacks open_interest for OI roll rule")

    ordered = _ordered_contracts(per_contract, specs)
    rolls = _roll_dates(ordered, roll_rule, confirm_days, calendar_offset_days)
    offsets, factors = _adjustments(ordered, rolls, adjustment)

    segments: list[Bars] = []
    for i, (_spec, frame) in enumerate(ordered):
        lo = rolls[i - 1] if i > 0 else None
        hi = rolls[i] if i < len(rolls) else None
        seg = _segment(frame, lo, hi).loc[:, list(BAR_COLUMNS)].copy()
        if seg.empty:
            raise DataIntegrityError(f"contract {ordered[i][0].symbol!r} contributes no sessions")
        for col in _PRICE_COLUMNS:
            seg[col] = seg[col] * factors[i] + offsets[i]
        segments.append(seg)

    continuous = pd.concat(segments).sort_index()
    if not continuous.index.is_unique:
        raise DataIntegrityError("continuous series has duplicate timestamps after splicing")

    meta = ContinuousMeta(
        roll_rule=roll_rule,
        adjustment=adjustment,
        roll_dates=[pd.Timestamp(r).date() for r in rolls],
        underlying_contracts=[spec.symbol for spec, _ in ordered],
    )
    return continuous, meta
