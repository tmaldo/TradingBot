"""Tests for the stationary block-bootstrap Sharpe confidence interval.

Uses :func:`futures_engine.validation.stats.bootstrap_sharpe_ci`, which wraps
``arch.bootstrap.StationaryBootstrap`` with a Politis-White
``optimal_block_length`` default (overridable). The bootstrap preserves serial
correlation, so the interval is honest for autocorrelated strategy returns
(G10). All randomness is seeded and reproducible; no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_engine.validation.stats import bootstrap_sharpe_ci


def _sample_sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std(ddof=1))


def _iid_returns(rng: np.random.Generator, mu: float, sigma: float, n: int) -> pd.Series:
    return pd.Series(rng.normal(mu, sigma, n))


# --- basic behaviour ---------------------------------------------------------


def test_ci_is_ordered_and_brackets_sample_sharpe() -> None:
    returns = _iid_returns(np.random.default_rng(1), 0.05, 1.0, 500)
    lo, hi = bootstrap_sharpe_ci(returns, n_boot=500, block_size=None, seed=7, conf=0.95)
    assert lo <= hi
    assert lo <= _sample_sharpe(returns) <= hi


def test_ci_is_reproducible_under_fixed_seed() -> None:
    returns = _iid_returns(np.random.default_rng(2), 0.02, 1.0, 400)
    a = bootstrap_sharpe_ci(returns, n_boot=400, block_size=3, seed=99, conf=0.95)
    b = bootstrap_sharpe_ci(returns, n_boot=400, block_size=3, seed=99, conf=0.95)
    assert a == b


def test_different_seeds_generally_differ() -> None:
    returns = _iid_returns(np.random.default_rng(3), 0.02, 1.0, 400)
    a = bootstrap_sharpe_ci(returns, n_boot=400, block_size=3, seed=1, conf=0.95)
    b = bootstrap_sharpe_ci(returns, n_boot=400, block_size=3, seed=2, conf=0.95)
    assert a != b


def test_optimal_block_length_default_runs() -> None:
    returns = _iid_returns(np.random.default_rng(4), 0.03, 1.0, 600)
    lo, hi = bootstrap_sharpe_ci(returns, n_boot=300, block_size=None, seed=5, conf=0.9)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_higher_confidence_widens_interval() -> None:
    returns = _iid_returns(np.random.default_rng(5), 0.04, 1.0, 500)
    lo90, hi90 = bootstrap_sharpe_ci(returns, n_boot=800, block_size=2, seed=8, conf=0.90)
    lo99, hi99 = bootstrap_sharpe_ci(returns, n_boot=800, block_size=2, seed=8, conf=0.99)
    assert (hi99 - lo99) >= (hi90 - lo90)


# --- coverage sanity (seeded Monte Carlo) ------------------------------------


def test_coverage_on_iid_returns_is_near_nominal() -> None:
    """The 90% interval should cover the population Sharpe ~90% of the time."""
    rng = np.random.default_rng(20260725)
    mu, sigma, n, reps, conf = 0.05, 1.0, 250, 120, 0.90
    true_sr = mu / sigma
    hits = 0
    for r in range(reps):
        returns = _iid_returns(rng, mu, sigma, n)
        lo, hi = bootstrap_sharpe_ci(returns, n_boot=250, block_size=1, seed=1000 + r, conf=conf)
        hits += int(lo <= true_sr <= hi)
    coverage = hits / reps
    assert 0.80 <= coverage <= 0.98


# --- validation --------------------------------------------------------------


def test_rejects_too_short_series() -> None:
    with pytest.raises(ValueError, match="observations"):
        bootstrap_sharpe_ci(pd.Series([0.01]), n_boot=100, block_size=1, seed=0, conf=0.95)


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.1, 1.2])
def test_rejects_bad_confidence(bad_conf: float) -> None:
    returns = _iid_returns(np.random.default_rng(6), 0.0, 1.0, 100)
    with pytest.raises(ValueError, match="conf"):
        bootstrap_sharpe_ci(returns, n_boot=100, block_size=1, seed=0, conf=bad_conf)


def test_rejects_bad_n_boot() -> None:
    returns = _iid_returns(np.random.default_rng(7), 0.0, 1.0, 100)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_sharpe_ci(returns, n_boot=0, block_size=1, seed=0, conf=0.95)


@pytest.mark.parametrize("bad_block", [0, -3])
def test_rejects_bad_block_size(bad_block: int) -> None:
    returns = _iid_returns(np.random.default_rng(8), 0.0, 1.0, 100)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_sharpe_ci(returns, n_boot=100, block_size=bad_block, seed=0, conf=0.95)
