"""Tests for the Deflated Sharpe Ratio and Probability of Backtest Overfitting.

DSR follows Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014): the
expected-maximum-Sharpe deflation term uses the Euler-Mascheroni / Gumbel
approximation for the expected maximum of ``n_trials`` i.i.d. standard normals,
and the Probabilistic Sharpe Ratio supplies the non-normality correction. PBO
follows Bailey et al., "The Probability of Backtest Overfitting" (2014), via the
combinatorially symmetric cross-validation (CSCV) logit-rank method.

All ``kurt`` inputs are **non-excess (Pearson) kurtosis** (normal = 3), matching
the paper's gamma4; callers using ``pandas.Series.kurt`` (which returns *excess*
kurtosis) must add 3.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from futures_engine.validation.stats import (
    _expected_max_sharpe_z,
    deflated_sharpe,
    pbo,
)

_NORM = NormalDist()


# --- expected-max-Sharpe (Gumbel) term ---------------------------------------


@pytest.mark.parametrize(
    ("n_trials", "expected"),
    [(10, 1.574598), (50, 2.276303), (100, 2.530603), (1000, 3.255122)],
)
def test_expected_max_sharpe_gumbel_values(n_trials: int, expected: float) -> None:
    """Euler-Mascheroni/Gumbel approximation of E[max of N std normals]."""
    assert _expected_max_sharpe_z(n_trials) == pytest.approx(expected, abs=1e-5)


def test_expected_max_sharpe_single_trial_is_zero() -> None:
    # A single trial cannot inflate the max: no deflation.
    assert _expected_max_sharpe_z(1) == 0.0


# --- DSR worked example (Bailey & López de Prado 2014) -----------------------


def test_deflated_sharpe_reproduces_worked_example() -> None:
    """Reproduce a fully specified DSR case within tolerance.

    Inputs (per-observation Sharpe): observed_sr=0.09, n_obs=1000, skew=-0.6,
    non-excess kurt=6.0. The PSR z-statistic against a zero benchmark is
    2.757585; the Gumbel expected-max for N=100 trials is 2.530603; hence
    DSR = Φ(2.757585 - 2.530603) = 0.589781. Recomputed here independently with
    ``statistics.NormalDist`` from the published Bailey-López de Prado formulas.
    """
    sr, n_obs, skew, kurt = 0.09, 1000, -0.6, 6.0
    z0 = sr * ((n_obs - 1) ** 0.5) / ((1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5)
    assert z0 == pytest.approx(2.757585, abs=1e-5)
    expected = _NORM.cdf(z0 - 2.530603)
    assert expected == pytest.approx(0.589781, abs=1e-5)
    got = deflated_sharpe(observed_sr=sr, n_trials=100, n_obs=n_obs, skew=skew, kurt=kurt)
    assert got == pytest.approx(0.589781, abs=1e-6)


def test_deflated_sharpe_single_trial_equals_probabilistic_sharpe() -> None:
    """With one trial there is nothing to deflate: DSR == PSR against zero."""
    sr, n_obs, skew, kurt = 0.09, 1000, -0.6, 6.0
    z0 = sr * ((n_obs - 1) ** 0.5) / ((1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5)
    got = deflated_sharpe(observed_sr=sr, n_trials=1, n_obs=n_obs, skew=skew, kurt=kurt)
    assert got == pytest.approx(_NORM.cdf(z0), abs=1e-9)


def test_deflated_sharpe_is_a_probability() -> None:
    for n_trials in (1, 5, 100, 10_000):
        val = deflated_sharpe(0.09, n_trials=n_trials, n_obs=1000, skew=0.0, kurt=3.0)
        assert 0.0 <= val <= 1.0


# --- DSR properties (hypothesis) ---------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    observed_sr=st.floats(min_value=0.01, max_value=0.3),
    n_obs=st.integers(min_value=50, max_value=5000),
    skew=st.floats(min_value=-1.5, max_value=1.5),
    kurt=st.floats(min_value=1.5, max_value=20.0),
    n_a=st.integers(min_value=1, max_value=500),
    step=st.integers(min_value=1, max_value=5000),
)
def test_deflated_sharpe_monotone_decreasing_in_n_trials(
    observed_sr: float, n_obs: int, skew: float, kurt: float, n_a: int, step: int
) -> None:
    """More trials → more selection bias → DSR can only fall (G10 honesty)."""
    n_b = n_a + step
    dsr_a = deflated_sharpe(observed_sr, n_a, n_obs, skew, kurt)
    dsr_b = deflated_sharpe(observed_sr, n_b, n_obs, skew, kurt)
    assert dsr_b <= dsr_a + 1e-9


@settings(max_examples=100, deadline=None)
@given(
    sr_lo=st.floats(min_value=0.0, max_value=0.15),
    bump=st.floats(min_value=0.01, max_value=0.15),
    n_obs=st.integers(min_value=50, max_value=5000),
    n_trials=st.integers(min_value=1, max_value=1000),
)
def test_deflated_sharpe_increasing_in_observed_sr(
    sr_lo: float, bump: float, n_obs: int, n_trials: int
) -> None:
    lo = deflated_sharpe(sr_lo, n_trials, n_obs, 0.0, 3.0)
    hi = deflated_sharpe(sr_lo + bump, n_trials, n_obs, 0.0, 3.0)
    assert hi >= lo - 1e-9


# --- DSR validation ----------------------------------------------------------


def test_deflated_sharpe_rejects_bad_n_obs() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe(0.09, n_trials=10, n_obs=1, skew=0.0, kurt=3.0)


def test_deflated_sharpe_rejects_bad_n_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(0.09, n_trials=0, n_obs=1000, skew=0.0, kurt=3.0)


def test_deflated_sharpe_rejects_nonpositive_variance_term() -> None:
    # skew*sr large enough to drive 1 - skew*sr + (kurt-1)/4*sr^2 <= 0.
    with pytest.raises(ValueError, match="variance"):
        deflated_sharpe(0.9, n_trials=10, n_obs=1000, skew=2.0, kurt=1.0)


# --- PBO / CSCV --------------------------------------------------------------


def _noise_matrix(rng: np.random.Generator, n_configs: int, n_time: int) -> np.ndarray:
    return rng.normal(size=(n_configs, n_time))


def test_pbo_pure_noise_is_about_one_half() -> None:
    """Symmetric no-skill configs → the IS-best is a coin-flip OOS → PBO ≈ 0.5."""
    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(4000 + seed)
        estimates.append(pbo(_noise_matrix(rng, 30, 300), n_partitions=10))
    mean_pbo = float(np.mean(estimates))
    assert 0.35 <= mean_pbo <= 0.65


def test_pbo_dominant_config_is_about_zero() -> None:
    """A genuinely dominant config is IS-best and OOS-best everywhere → PBO ≈ 0."""
    rng = np.random.default_rng(11)
    m = rng.normal(size=(25, 400)) * 0.4
    m[0] += 1.2  # config 0 dominates in every slice
    assert pbo(m, n_partitions=12) < 0.05


def test_pbo_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    m = rng.normal(size=(20, 200))
    assert pbo(m, n_partitions=8) == pbo(m, n_partitions=8)


def test_pbo_requires_even_partitions() -> None:
    m = np.random.default_rng(0).normal(size=(10, 100))
    with pytest.raises(ValueError, match="even"):
        pbo(m, n_partitions=7)


def test_pbo_rejects_too_few_partitions() -> None:
    m = np.random.default_rng(0).normal(size=(10, 100))
    with pytest.raises(ValueError):
        pbo(m, n_partitions=1)


def test_pbo_rejects_too_few_configs() -> None:
    m = np.random.default_rng(0).normal(size=(1, 100))
    with pytest.raises(ValueError, match="config"):
        pbo(m, n_partitions=4)


def test_pbo_rejects_partitions_exceeding_time() -> None:
    m = np.random.default_rng(0).normal(size=(10, 6))
    with pytest.raises(ValueError):
        pbo(m, n_partitions=8)


def test_pbo_result_is_a_probability() -> None:
    rng = np.random.default_rng(1)
    val = pbo(rng.normal(size=(15, 250)), n_partitions=10)
    assert 0.0 <= val <= 1.0
