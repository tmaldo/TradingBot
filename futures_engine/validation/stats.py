"""Anti-overfitting validation statistics (G10).

Mandatory report outputs that decide whether any strategy is ever believed:

* :func:`deflated_sharpe` -- Deflated Sharpe Ratio (Bailey & López de Prado,
  "The Deflated Sharpe Ratio", 2014): the probability that the observed Sharpe
  exceeds what the *best of ``n_trials``* random trials would produce, corrected
  for non-normality via the Probabilistic Sharpe Ratio.
* :func:`pbo` -- Probability of Backtest Overfitting via combinatorially
  symmetric cross-validation (Bailey et al., "The Probability of Backtest
  Overfitting", 2014).
* :func:`bootstrap_sharpe_ci` -- stationary block-bootstrap confidence interval
  for the Sharpe ratio (via :mod:`arch`).
* :func:`red_flags` -- structured warnings for implausible or fragile results.

Only the standard library's :class:`statistics.NormalDist` is used for the
normal CDF / inverse CDF (no scipy). Skewness/kurtosis are supplied by the
caller (e.g. from pandas). ``kurt`` is always **non-excess (Pearson) kurtosis**
(normal = 3), matching the papers' gamma4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from statistics import NormalDist
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from pydantic import BaseModel, ConfigDict, Field

_NORM = NormalDist()
# Euler-Mascheroni constant, used in the Gumbel expected-maximum approximation.
_EULER_MASCHERONI = 0.5772156649015329

FloatArray = npt.NDArray[np.float64]


# --- Deflated Sharpe Ratio ---------------------------------------------------


def _expected_max_sharpe_z(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. standard normals (Gumbel approx.).

    Bailey & López de Prado (2014), eq. for E[max]: using the Euler-Mascheroni
    constant gamma and the Gumbel limit,

        E[max_N Z] ≈ (1 - gamma)·Φ⁻¹(1 - 1/N) + gamma·Φ⁻¹(1 - 1/(N·e)).

    A single trial cannot inflate the maximum, so the term is 0 for ``N == 1``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    quantile_a = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    quantile_b = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - _EULER_MASCHERONI) * quantile_a + _EULER_MASCHERONI * quantile_b


def _sharpe_estimator_std(observed_sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """Standard error of the Sharpe estimator (Lo 2002; Mertens; AFML §5).

        SE(SR̂) = sqrt( (1 - gamma3·SR̂ + (gamma4 - 1)/4·SR̂²) / (n_obs - 1) )

    with skewness gamma3 and non-excess kurtosis gamma4. Raises if the variance term is
    non-positive (a degenerate / mis-specified moment set).
    """
    variance_term = 1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr**2
    if variance_term <= 0.0:
        raise ValueError(
            "non-positive Sharpe variance term "
            f"(1 - skew*SR + (kurt-1)/4*SR^2 = {variance_term:.6g}); "
            "check skew/kurt/observed_sr"
        )
    return math.sqrt(variance_term / (n_obs - 1))


def deflated_sharpe(
    observed_sr: float, n_trials: int, n_obs: int, skew: float, kurt: float
) -> float:
    """Deflated Sharpe Ratio: P(true SR > 0) after correcting for selection bias.

    Implements Bailey & López de Prado (2014). The deflation benchmark is the
    expected maximum Sharpe of ``n_trials`` random trials,
    ``SR₀ = SE(SR̂) · E[max_N Z]`` (:func:`_expected_max_sharpe_z`), where the
    per-trial dispersion is approximated by the Sharpe estimator's own standard
    error :func:`_sharpe_estimator_std`. The Probabilistic Sharpe Ratio then
    gives

        DSR = Φ( (SR̂ - SR₀) / SE(SR̂) ) = Φ( SR̂/SE(SR̂) - E[max_N Z] ),

    a probability in ``[0, 1]``; higher is better. For ``n_trials == 1`` this
    reduces to the plain Probabilistic Sharpe Ratio against a zero benchmark.

    Parameters
    ----------
    observed_sr:
        Observed Sharpe ratio in the **same frequency as ``n_obs``** (per
        observation, not annualised).
    n_trials:
        Honest number of configurations tried (from the trial store); ``>= 1``.
    n_obs:
        Number of return observations; ``>= 2``.
    skew:
        Skewness of the returns (gamma3).
    kurt:
        **Non-excess** kurtosis of the returns (gamma4; normal = 3).
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    se = _sharpe_estimator_std(observed_sr, n_obs, skew, kurt)
    z_observed = observed_sr / se  # PSR z-statistic against a zero benchmark
    return _NORM.cdf(z_observed - _expected_max_sharpe_z(n_trials))


# --- Probability of Backtest Overfitting (CSCV) ------------------------------


def _column_sharpe(sub: FloatArray) -> FloatArray:
    """Per-config (row-wise) Sharpe across the columns of ``sub``."""
    mean = sub.mean(axis=1)
    std = sub.std(axis=1, ddof=1)
    safe = np.where(std > 0.0, std, 1.0)
    return np.where(std > 0.0, mean / safe, 0.0)


def pbo(perf_matrix: np.ndarray, n_partitions: int) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al., 2014).

    ``perf_matrix`` has one row per configuration and one column per time slice
    (``M[i, t]`` = per-period performance of config ``i`` in slice ``t``). The
    columns are split into ``n_partitions`` (which **must be even**) contiguous
    submatrices; for each of the ``C(S, S/2)`` ways to split them into an
    in-sample (IS) half and its complementary out-of-sample (OOS) half we:

    1. rank configs by IS Sharpe and take the IS-best, ``n*``;
    2. find ``n*``'s relative rank ``ω ∈ (0, 1)`` in the OOS Sharpe ordering
       (``ω = rank / (N + 1)``, so it is never degenerate);
    3. record the logit ``λ = ln(ω / (1 - ω))``.

    PBO is the fraction of splits with ``λ < 0`` — i.e. the estimated
    probability that the IS-optimal configuration underperforms the OOS median.
    ``≈ 0.5`` indicates no genuine skill; ``≈ 0`` indicates a robust edge.
    """
    if perf_matrix.ndim != 2:
        raise ValueError(f"perf_matrix must be 2-D, got {perf_matrix.ndim}-D")
    n_configs, n_time = perf_matrix.shape
    if n_configs < 2:
        raise ValueError(f"need >= 2 configs to rank, got {n_configs}")
    if n_partitions < 2:
        raise ValueError(f"n_partitions must be >= 2, got {n_partitions}")
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even (CSCV), got {n_partitions}")
    if n_partitions > n_time:
        raise ValueError(f"n_partitions ({n_partitions}) exceeds number of time slices ({n_time})")

    matrix = np.asarray(perf_matrix, dtype=np.float64)
    col_groups = np.array_split(np.arange(n_time), n_partitions)
    half = n_partitions // 2
    below_zero = 0
    total = 0
    for is_combo in combinations(range(n_partitions), half):
        is_set = set(is_combo)
        is_cols = np.concatenate([col_groups[g] for g in is_combo])
        oos_cols = np.concatenate([col_groups[g] for g in range(n_partitions) if g not in is_set])
        is_perf = _column_sharpe(matrix[:, is_cols])
        oos_perf = _column_sharpe(matrix[:, oos_cols])
        best = int(np.argmax(is_perf))
        # 1-based ascending OOS rank of the IS-best config (1 = worst).
        rank = int(np.sum(oos_perf < oos_perf[best])) + 1
        omega = rank / (n_configs + 1)
        logit = math.log(omega / (1.0 - omega))
        below_zero += int(logit < 0.0)
        total += 1
    return below_zero / total


# --- bootstrap Sharpe confidence interval ------------------------------------


def _per_period_sharpe(sample: np.ndarray) -> FloatArray:
    """Per-period Sharpe of a 1-D returns sample (mean / std, ddof=1).

    Returns a length-1 array: ``arch``'s ``conf_int`` expects the statistic
    function to return an ``ndarray``.
    """
    flat = np.asarray(sample, dtype=np.float64).ravel()
    std = flat.std(ddof=1)
    value = flat.mean() / std if std > 0.0 else 0.0
    return np.array([value], dtype=np.float64)


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    block_size: int | None = None,
    seed: int = 0,
    conf: float = 0.95,
) -> tuple[float, float]:
    """Stationary block-bootstrap confidence interval for the Sharpe ratio.

    Resamples ``returns`` with :class:`arch.bootstrap.StationaryBootstrap`
    (Politis-Romano) so serial correlation is preserved, then returns the
    percentile interval of the per-period Sharpe ratio at confidence ``conf``.

    Parameters
    ----------
    returns:
        Per-period returns; at least 2 observations.
    n_boot:
        Number of bootstrap resamples; ``>= 1``.
    block_size:
        Expected block length. ``None`` (default) uses the Politis-White
        ``optimal_block_length`` (stationary variant), rounded to ``>= 1``. If
        given explicitly it must be ``>= 1``.
    seed:
        Seed for the bootstrap RNG; identical seeds give identical intervals.
    conf:
        Confidence level in ``(0, 1)`` — e.g. ``0.95`` for a 95% interval.

    Returns
    -------
    ``(low, high)`` percentile bounds of the Sharpe ratio.
    """
    if conf <= 0.0 or conf >= 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size must be >= 1 when given, got {block_size}")
    data = np.asarray(returns, dtype=np.float64).ravel()
    if data.shape[0] < 2:
        raise ValueError(f"need >= 2 observations, got {data.shape[0]}")

    if block_size is None:
        optimal = float(optimal_block_length(data)["stationary"].iloc[0])
        block = max(1, round(optimal))
    else:
        block = block_size

    bootstrap = StationaryBootstrap(block, data, seed=seed)
    interval = bootstrap.conf_int(_per_period_sharpe, reps=n_boot, method="percentile", size=conf)
    return float(interval[0, 0]), float(interval[1, 0])


# --- red-flag reporter -------------------------------------------------------


@runtime_checkable
class BacktestResultLike(Protocol):
    """Structural view of a backtest result needed by :func:`red_flags`.

    The concrete ``BacktestResult`` produced by task T6 will satisfy this
    Protocol. Required members:

    * ``returns`` -- per-period returns (:class:`pandas.Series`); used for the
      volatility estimate in the smoothness check.
    * ``equity`` -- the (positive-valued) equity curve
      (:class:`pandas.Series`); used for the maximum-drawdown estimate.
    * ``metrics`` -- a mapping that must contain at least ``"sharpe"`` and
      ``"win_rate"`` (win rate as a fraction in ``[0, 1]``).
    """

    returns: pd.Series
    equity: pd.Series
    metrics: Mapping[str, float]


class RedFlag(BaseModel):
    """A single structured warning emitted by :func:`red_flags`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["warn", "fail"]


class RedFlagConfig(BaseModel):
    """Thresholds for :func:`red_flags` (G10; no magic constants in logic).

    Defaults encode the constraint's guidance: a Sharpe above 3 is implausible;
    a win rate above 70% paired with implausibly smooth equity is suspicious; an
    edge is "smooth" when its maximum drawdown is below ``smoothness_k`` times
    the per-period return volatility; and a real edge should retain at least
    ``min_edge_retention`` of its Sharpe under a 1-bar delay and under costs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sharpe_max: float = Field(default=3.0, gt=0.0)
    win_rate_max: float = Field(default=0.70, gt=0.0, lt=1.0)
    smoothness_k: float = Field(default=10.0, gt=0.0)
    min_edge_retention: float = Field(default=0.5, ge=0.0, le=1.0)


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction of the peak."""
    values = np.asarray(equity, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = np.where(running_max > 0.0, (running_max - values) / running_max, 0.0)
    return float(drawdown.max())


def _edge_retained(reference_sharpe: float, degraded_sharpe: float, retention: float) -> bool:
    """True if a positive reference edge is preserved to within ``retention``."""
    if reference_sharpe <= 0.0:
        return True  # no positive edge to lose
    return degraded_sharpe >= retention * reference_sharpe


def red_flags(
    result: BacktestResultLike,
    delayed: BacktestResultLike | None,
    gross: BacktestResultLike | None,
    config: RedFlagConfig | None = None,
) -> list[RedFlag]:
    """Return structured red flags for a backtest result (G10).

    Emits, in order:

    * ``SHARPE_IMPLAUSIBLE`` (warn) -- ``sharpe`` exceeds ``sharpe_max``.
    * ``SMOOTH_HIGH_WINRATE`` (warn) -- ``win_rate`` exceeds ``win_rate_max``
      while the equity is implausibly smooth (max drawdown below
      ``smoothness_k * volatility``).
    * ``EDGE_FAILS_DELAY`` (fail) -- the ``delayed`` (1-bar-delayed) variant
      fails to retain ``min_edge_retention`` of ``result``'s Sharpe.
    * ``EDGE_FAILS_COSTS`` (fail) -- ``result`` (net of costs) fails to retain
      ``min_edge_retention`` of the ``gross`` (pre-cost) Sharpe.

    ``delayed`` and ``gross`` are optional; their checks are skipped when
    ``None``. An empty list means no red flags fired.
    """
    cfg = config if config is not None else RedFlagConfig()
    flags: list[RedFlag] = []

    sharpe = float(result.metrics["sharpe"])
    win_rate = float(result.metrics["win_rate"])
    volatility = float(np.asarray(result.returns, dtype=np.float64).std(ddof=1))
    max_dd = _max_drawdown(result.equity)

    if sharpe > cfg.sharpe_max:
        flags.append(
            RedFlag(
                code="SHARPE_IMPLAUSIBLE",
                message=(
                    f"Sharpe {sharpe:.2f} exceeds the plausible maximum "
                    f"{cfg.sharpe_max:.2f}; suspect overfitting or a P&L bug."
                ),
                severity="warn",
            )
        )

    if win_rate > cfg.win_rate_max and max_dd < cfg.smoothness_k * volatility:
        flags.append(
            RedFlag(
                code="SMOOTH_HIGH_WINRATE",
                message=(
                    f"win rate {win_rate:.0%} exceeds {cfg.win_rate_max:.0%} with "
                    f"implausibly smooth equity (max drawdown {max_dd:.2%} < "
                    f"{cfg.smoothness_k:g}*vol {volatility:.4f}); suspect hidden "
                    "tail risk or mismarked P&L."
                ),
                severity="warn",
            )
        )

    if delayed is not None:
        delayed_sharpe = float(delayed.metrics["sharpe"])
        if not _edge_retained(sharpe, delayed_sharpe, cfg.min_edge_retention):
            flags.append(
                RedFlag(
                    code="EDGE_FAILS_DELAY",
                    message=(
                        f"Sharpe collapses from {sharpe:.2f} to {delayed_sharpe:.2f} "
                        "under a 1-bar execution delay (retention below "
                        f"{cfg.min_edge_retention:.0%}); edge likely a look-ahead artefact."
                    ),
                    severity="fail",
                )
            )

    if gross is not None:
        gross_sharpe = float(gross.metrics["sharpe"])
        if not _edge_retained(gross_sharpe, sharpe, cfg.min_edge_retention):
            flags.append(
                RedFlag(
                    code="EDGE_FAILS_COSTS",
                    message=(
                        f"net Sharpe {sharpe:.2f} retains less than "
                        f"{cfg.min_edge_retention:.0%} of the gross Sharpe "
                        f"{gross_sharpe:.2f}; the edge is eaten by costs."
                    ),
                    severity="fail",
                )
            )

    return flags
