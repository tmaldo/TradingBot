"""Prop-account survival Monte-Carlo gate (task T7, Global Constraints G11/G16).

The question this module answers is the highest-stakes one in the whole system:
*will this strategy blow the prop account's trailing-drawdown rule before it
pays out?* We answer it by block-bootstrapping the strategy's realized
trade-level P&L into many synthetic account-equity paths and running each one
through T2's :func:`~futures_engine.prop.rules.simulate_account` oracle.

Method
------
1. **Resample.** The strategy's per-trade ``net_pnl_usd`` series is resampled
   with :class:`arch.bootstrap.CircularBlockBootstrap` (fixed block length so
   serial structure inside a block is preserved; block size defaults to the
   Politis-White :func:`~arch.bootstrap.optimal_block_length`). We do **not**
   hand-roll a resampler. ``arch`` draws are the length of the input series, so
   to build ``n_paths`` paths each long enough to fill ``horizon_days`` we draw
   a stream of independent circular-block resamples, concatenate them, and chop
   the stream into equal path slices. Joins between concatenated draws are the
   only place block continuity is not guaranteed; this is documented and minor.

2. **Trades -> days.** Resampled trades are grouped into days of
   ``trades_per_day`` each. ``trades_per_day`` is inferred from the *original*
   TradeLog: the mean number of trades per calendar day of ``entry_ts`` when
   that column is present, else the documented default of one trade per day
   (:data:`DEFAULT_TRADES_PER_DAY`).

3. **Intraday path.** Each day's :class:`~futures_engine.prop.rules.DayPnL`
   carries ``realized`` (the day's summed trade P&L) and the running extremes of
   the within-day cumulative P&L relative to the day open (which starts at 0).
   When per-trade ``mae_usd`` (<=0) / ``mfe_usd`` (>=0) excursion columns are
   present they widen each trade's contribution to the intraday min/max;
   **absent them the extremes are taken from trade *closes* only**, which
   understates true intraday swings and therefore *understates* bust risk in
   ``eod`` mode but is neutral for the trough test -- documented as a known,
   conservative-leaning approximation.

4. **Size.** ``contracts`` linearly scales every trade's P&L (and any MAE/MFE):
   the logged TradeLog is treated as a one-lot unit strategy, so running it at
   ``k`` contracts multiplies each day's P&L by ``k``. Under this linear scaling
   a bust at size ``k`` implies a bust at every larger size for the *same* path
   (with an unreachable profit target), which is why ``p_survival`` is monotone
   non-increasing in ``contracts`` -- the property the sizer relies on.

Reproducibility (G11/G15): every run is fully determined by ``seed``; identical
inputs and seed give an identical :class:`SurvivalReport`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from arch.bootstrap import CircularBlockBootstrap, optimal_block_length
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.costs.model import TradeLog
from futures_engine.prop.rules import (
    AccountOutcome,
    DayPnL,
    PropRuleSet,
    simulate_account,
)

# Fallback trades-per-day grouping when the TradeLog carries no ``entry_ts``.
DEFAULT_TRADES_PER_DAY: int = 1

# The Politis-White ``optimal_block_length`` estimator is unstable/undefined on
# very short series (it inspects ~sqrt(n) autocorrelations); below this many
# trades we fall back to a block of 1 rather than trust a degenerate estimate.
_MIN_OBS_FOR_OPTIMAL_BLOCK: int = 16

# 90% two-sided normal quantile, for the Wilson score interval on p_survival.
_Z_90: float = 1.6448536269514722


class SurvivalReport(BaseModel):
    """Outcome distribution of a survival Monte Carlo (all fractions in ``[0, 1]``).

    * ``p_survival`` -- fraction of paths that did **not** bust over the horizon
      (they either passed the evaluation or ran the horizon out clean).
    * ``p_target_before_bust`` -- fraction that reached the profit target without
      busting first.
    * ``median_days_to_target`` -- median ``days_elapsed`` among the paths that
      hit the target, or ``None`` if none did.
    * ``bust_reasons`` -- histogram of :class:`AccountOutcome` ``bust_reason``
      over *all* paths (each value is a fraction of ``n_paths``; they sum to
      ``1 - p_survival``).
    * ``ci_90`` -- 90% Wilson score interval on ``p_survival``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_survival: float = Field(ge=0.0, le=1.0)
    p_target_before_bust: float = Field(ge=0.0, le=1.0)
    median_days_to_target: float | None
    bust_reasons: dict[str, float]
    ci_90: tuple[float, float]


def _infer_trades_per_day(trades: TradeLog) -> int:
    """Mean trades per calendar day of ``entry_ts`` (>=1), else the default."""
    if "entry_ts" not in trades.columns:
        return DEFAULT_TRADES_PER_DAY
    entry = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    if entry.isna().any():
        return DEFAULT_TRADES_PER_DAY
    per_day = entry.dt.floor("D").value_counts()
    if per_day.empty:
        return DEFAULT_TRADES_PER_DAY
    return max(1, round(float(per_day.mean())))


def _resample_paths(
    values: np.ndarray, n_paths: int, path_len: int, block_size: int, seed: int
) -> np.ndarray:
    """Draw ``n_paths`` block-bootstrap paths of length ``path_len`` from ``values``.

    Uses :class:`arch.bootstrap.CircularBlockBootstrap`; ``arch`` resamples are
    the length of ``values``, so we concatenate enough independent draws to fill
    ``n_paths * path_len`` observations and reshape. Deterministic in ``seed``.
    """
    n = values.shape[0]
    needed = n_paths * path_len
    reps = math.ceil(needed / n)
    boot = CircularBlockBootstrap(block_size, values, seed=seed)
    chunks: list[np.ndarray] = []
    total = 0
    for pos_data, _kw in boot.bootstrap(reps):
        chunk = np.asarray(pos_data[0], dtype=np.float64).ravel()
        chunks.append(chunk)
        total += chunk.shape[0]
        if total >= needed:
            break
    stream = np.concatenate(chunks)[:needed]
    return stream.reshape(n_paths, path_len)


def _build_days(
    pnl_path: np.ndarray,
    mae_path: np.ndarray | None,
    mfe_path: np.ndarray | None,
    trades_per_day: int,
    contracts: float,
) -> list[DayPnL]:
    """Group one resampled trade path into per-day :class:`DayPnL` objects.

    The within-day cumulative P&L starts at 0 (the day open). For each trade the
    running min/max are updated with the trade's low excursion (``mae``), high
    excursion (``mfe``) and close; absent MAE/MFE only the close is used.
    """
    n_trades = pnl_path.shape[0]
    n_days = n_trades // trades_per_day
    days: list[DayPnL] = []
    for d in range(n_days):
        start = d * trades_per_day
        cum = 0.0
        lo = 0.0
        hi = 0.0
        for i in range(start, start + trades_per_day):
            pnl = float(pnl_path[i]) * contracts
            if mae_path is not None and mfe_path is not None:
                low_exc = cum + float(mae_path[i]) * contracts
                high_exc = cum + float(mfe_path[i]) * contracts
                lo = min(lo, low_exc)
                hi = max(hi, high_exc)
            cum += pnl
            lo = min(lo, cum)
            hi = max(hi, cum)
        realized = cum
        # Guard tiny float drift so DayPnL's lo <= realized <= hi holds exactly.
        lo = min(lo, realized)
        hi = max(hi, realized)
        days.append(DayPnL(realized=realized, intraday_min=lo, intraday_max=hi))
    return days


def _default_block(values: np.ndarray) -> int:
    """Politis-White ``optimal_block_length`` (circular), robust on short series."""
    n = int(values.shape[0])
    if n < _MIN_OBS_FOR_OPTIMAL_BLOCK:
        return 1
    try:
        with np.errstate(all="ignore"):
            optimal = float(optimal_block_length(values)["circular"].iloc[0])
    except (ValueError, ZeroDivisionError):
        return 1
    if not math.isfinite(optimal):
        return 1
    return max(1, min(round(optimal), n))


def _wilson_ci_90(k: int, n: int) -> tuple[float, float]:
    """90% Wilson score interval for a binomial proportion ``k/n``."""
    if n == 0:
        return (0.0, 0.0)
    z = _Z_90
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def monte_carlo_survival(
    trades: TradeLog,
    rules: PropRuleSet,
    contracts: int,
    n_paths: int,
    horizon_days: int,
    block_size: int | None = None,
    seed: int = 0,
) -> SurvivalReport:
    """Monte-Carlo an account's survival of ``rules`` at a given ``contracts`` size.

    Parameters
    ----------
    trades:
        A conforming TradeLog with a ``net_pnl_usd`` column (see
        :mod:`futures_engine.costs.model`). Optional ``mae_usd`` / ``mfe_usd``
        columns sharpen the intraday path (see module docstring).
    rules:
        The prop rule set to test against.
    contracts:
        Position size; linearly scales each trade's P&L. Must be ``>= 0``.
    n_paths:
        Number of synthetic paths (``>= 1``).
    horizon_days:
        Trading days simulated per path (``>= 1``).
    block_size:
        Circular-block length. ``None`` (default) uses the Politis-White
        ``optimal_block_length`` (circular variant), rounded to ``>= 1``.
    seed:
        RNG seed; the whole report is reproducible from it.
    """
    if contracts < 0:
        raise ValueError(f"contracts must be >= 0, got {contracts}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if "net_pnl_usd" not in trades.columns:
        raise ValueError("TradeLog missing required column 'net_pnl_usd'")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size must be >= 1 when given, got {block_size}")

    pnl = np.asarray(trades["net_pnl_usd"], dtype=np.float64).ravel()
    if pnl.shape[0] < 1:
        raise ValueError("TradeLog is empty")

    has_excursions = "mae_usd" in trades.columns and "mfe_usd" in trades.columns
    mae = np.asarray(trades["mae_usd"], dtype=np.float64).ravel() if has_excursions else None
    mfe = np.asarray(trades["mfe_usd"], dtype=np.float64).ravel() if has_excursions else None

    block = _default_block(pnl) if block_size is None else block_size

    trades_per_day = _infer_trades_per_day(trades)
    path_len = horizon_days * trades_per_day

    pnl_paths = _resample_paths(pnl, n_paths, path_len, block, seed)
    if mae is not None and mfe is not None:
        # Same seed + block + length => arch draws identical resample indices, so
        # the MAE/MFE paths stay aligned trade-for-trade with the P&L paths.
        mae_paths: np.ndarray | None = _resample_paths(mae, n_paths, path_len, block, seed)
        mfe_paths: np.ndarray | None = _resample_paths(mfe, n_paths, path_len, block, seed)
    else:
        mae_paths = None
        mfe_paths = None

    survived = 0
    hit_target = 0
    bust_counts: dict[str, int] = {}
    days_to_target: list[int] = []

    for p in range(n_paths):
        days = _build_days(
            pnl_paths[p],
            None if mae_paths is None else mae_paths[p],
            None if mfe_paths is None else mfe_paths[p],
            trades_per_day,
            float(contracts),
        )
        outcome: AccountOutcome = simulate_account(days, rules)
        if outcome.survived:
            survived += 1
            if outcome.hit_target:
                hit_target += 1
                days_to_target.append(outcome.days_elapsed)
        else:
            reason = outcome.bust_reason or "unknown"
            bust_counts[reason] = bust_counts.get(reason, 0) + 1

    p_survival = survived / n_paths
    p_target = hit_target / n_paths
    median_days = float(np.median(days_to_target)) if days_to_target else None
    bust_reasons = {k: v / n_paths for k, v in bust_counts.items()}
    ci_90 = _wilson_ci_90(survived, n_paths)

    return SurvivalReport(
        p_survival=p_survival,
        p_target_before_bust=p_target,
        median_days_to_target=median_days,
        bust_reasons=bust_reasons,
        ci_90=ci_90,
    )


def passes_gate(report: SurvivalReport, min_survival: float = 0.90) -> bool:
    """Whether ``report`` clears the survival gate (G11). Default threshold 0.90."""
    return report.p_survival >= min_survival


def max_contracts_surviving(
    trades: TradeLog,
    rules: PropRuleSet,
    n_paths: int,
    horizon_days: int,
    seed: int,
    block_size: int | None = None,
    min_survival: float = 0.90,
    max_search: int = 50,
) -> int:
    """Largest contract count whose :class:`SurvivalReport` still passes the gate.

    Because ``p_survival`` is monotone non-increasing in ``contracts`` for a
    fixed seed, a linear scan up from 1 finds the boundary; returns 0 if even a
    single contract fails. Deterministic (fully seeded).
    """
    best = 0
    for k in range(1, max_search + 1):
        report = monte_carlo_survival(
            trades,
            rules,
            contracts=k,
            n_paths=n_paths,
            horizon_days=horizon_days,
            block_size=block_size,
            seed=seed,
        )
        if passes_gate(report, min_survival):
            best = k
        else:
            break
    return best
