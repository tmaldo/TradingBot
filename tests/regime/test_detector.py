"""Tests for regime detectors (task T4, G4/G6).

The binding requirement is **causality**: ``regimes``/``proba`` may use only data
at or before each timestamp, so both survive the T1 look-ahead shift test. We
also pin determinism under a fixed seed, output shapes/alignment, and that a
regime shift-check is registered into the audit and passes.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from futures_engine.data import audit
from futures_engine.data.audit import (
    reference_history,
    register_pit_check,
    registered_checks,
    run_pit_audit,
    run_pit_check,
)
from futures_engine.regime.detector import (
    ChangePointRegimeDetector,
    HMMRegimeDetector,
    RegimeDetector,
    register_regime_checks,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    audit.clear_registry()
    yield
    audit.clear_registry()


def _regime_bars(n: int = 160, seed: int = 0) -> pd.DataFrame:
    """Two-regime OHLCV: a low-vol uptrend followed by a high-vol downtrend."""
    rng = np.random.default_rng(seed)
    half = n // 2
    up = rng.normal(0.0015, 0.004, half)
    down = rng.normal(-0.0015, 0.02, n - half)
    rets = np.concatenate([up, down])
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC"))
    spread = np.abs(rng.normal(0.0, 0.003, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


# --- protocol conformance ----------------------------------------------------


def test_detectors_satisfy_protocol() -> None:
    assert isinstance(HMMRegimeDetector(n_states=2, seed=0), RegimeDetector)
    assert isinstance(ChangePointRegimeDetector(model="l2", penalty=5.0), RegimeDetector)


# --- HMM: shape, alignment, determinism --------------------------------------


def test_hmm_regimes_shape_and_alignment() -> None:
    bars = _regime_bars()
    det = HMMRegimeDetector(n_states=2, seed=0).fit(bars)
    reg = det.regimes(bars)
    assert reg.index.equals(bars.index)
    assert reg.dtype == np.int64
    assert set(reg.unique()) <= {-1, 0, 1}  # -1 = warm-up (no return at bar 0)


def test_hmm_proba_is_distribution() -> None:
    bars = _regime_bars()
    det = HMMRegimeDetector(n_states=2, seed=0).fit(bars)
    proba = det.proba(bars)
    assert list(proba.columns) == [0, 1]
    assert proba.index.equals(bars.index)
    valid = proba.dropna()
    assert np.allclose(valid.sum(axis=1).to_numpy(), 1.0)
    assert (valid.to_numpy() >= 0.0).all()


def test_hmm_is_deterministic_under_fixed_seed() -> None:
    bars = _regime_bars()
    a = HMMRegimeDetector(n_states=2, seed=7).fit(bars).regimes(bars)
    b = HMMRegimeDetector(n_states=2, seed=7).fit(bars).regimes(bars)
    pd.testing.assert_series_equal(a, b)


def test_hmm_rejects_bad_state_count() -> None:
    with pytest.raises(ValueError, match="n_states"):
        HMMRegimeDetector(n_states=1, seed=0)


# --- HMM: causality (the critical requirement) -------------------------------


def test_hmm_regimes_pass_shift_test() -> None:
    bars = _regime_bars()
    det = HMMRegimeDetector(n_states=2, seed=0).fit(bars)
    run_pit_check(det.regimes, bars, name="regime.hmm")  # must not raise


def test_hmm_proba_pass_shift_test() -> None:
    bars = _regime_bars()
    det = HMMRegimeDetector(n_states=2, seed=0).fit(bars)
    run_pit_check(det.proba, bars, name="regime.hmm.proba")  # must not raise


# --- change-point: shape, determinism, causality -----------------------------


def test_changepoint_regimes_shape_and_alignment() -> None:
    bars = _regime_bars(n=90)
    det = ChangePointRegimeDetector(model="l2", penalty=5.0).fit(bars)
    reg = det.regimes(bars)
    assert reg.index.equals(bars.index)
    assert reg.dtype == np.int64
    assert (reg >= 0).all()


def test_changepoint_detects_a_regime_change() -> None:
    bars = _regime_bars(n=90)
    det = ChangePointRegimeDetector(model="l2", penalty=2.0).fit(bars)
    assert det.regimes(bars).nunique() >= 2


def test_changepoint_is_deterministic() -> None:
    bars = _regime_bars(n=90)
    a = ChangePointRegimeDetector(model="l2", penalty=5.0).fit(bars).regimes(bars)
    b = ChangePointRegimeDetector(model="l2", penalty=5.0).fit(bars).regimes(bars)
    pd.testing.assert_series_equal(a, b)


def test_changepoint_rejects_bad_penalty() -> None:
    with pytest.raises(ValueError, match="penalty"):
        ChangePointRegimeDetector(model="l2", penalty=0.0)


def test_changepoint_regimes_pass_shift_test() -> None:
    bars = _regime_bars(n=80)
    det = ChangePointRegimeDetector(model="l2", penalty=5.0).fit(bars)
    run_pit_check(det.regimes, bars, name="regime.changepoint")  # must not raise


def test_changepoint_proba_is_onehot() -> None:
    bars = _regime_bars(n=80)
    det = ChangePointRegimeDetector(model="l2", penalty=5.0).fit(bars)
    proba = det.proba(bars)
    assert proba.index.equals(bars.index)
    assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0)
    assert set(np.unique(proba.to_numpy())) <= {0.0, 1.0}


# --- audit registration ------------------------------------------------------


def test_register_regime_checks_populates_registry() -> None:
    register_regime_checks()
    assert "regime.hmm" in registered_checks()


def test_registered_regime_check_passes_audit() -> None:
    register_regime_checks()
    report = run_pit_audit(reference_history())
    assert report.ok, report.violations
    assert "regime.hmm" in report.checked


def test_register_regime_checks_idempotent() -> None:
    register_regime_checks()
    n = len(registered_checks())
    register_regime_checks()
    assert len(registered_checks()) == n


def test_import_of_regime_package_registers_check() -> None:
    script = (
        "import futures_engine.regime\n"
        "from futures_engine.data.audit import registered_checks\n"
        "assert 'regime.hmm' in registered_checks(), sorted(registered_checks())\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


def test_leaky_regime_variant_is_caught() -> None:
    """Guard: a smoothed (full-sequence) label leaks; the audit must catch it."""

    def leaky_smoothed(history: pd.DataFrame) -> pd.Series:
        # Centered (two-sided) smoothing peeks ahead, like full-sequence Viterbi
        # decoding -- exactly the non-causal labeling this module must avoid.
        return history["close"].rolling(5, center=True).mean()

    register_pit_check("regime.leaky", leaky_smoothed)
    report = run_pit_audit(reference_history())
    assert "regime.leaky" in report.violations
