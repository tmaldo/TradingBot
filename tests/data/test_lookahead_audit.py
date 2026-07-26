"""Tests for the point-in-time / look-ahead audit (Global Constraint G4).

The audit re-runs a registered feature with its input history truncated at
several cut points and asserts the values at and before each cut are unchanged.
A causal feature passes; a deliberately leaky feature (reads a future bar) is
caught. This is a highest-stakes guard (G16): a leak here silently inflates
every backtest, so the leaky-feature detection test is the crux.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

from futures_engine.data import audit
from futures_engine.data.audit import (
    LookaheadError,
    register_pit_check,
    run_pit_audit,
    run_pit_check,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    audit.clear_registry()
    yield
    audit.clear_registry()


def _history(n: int = 60) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC").tolist())
    close = pd.Series(range(n), index=idx, dtype="float64") + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )


def causal_sma(history: pd.DataFrame) -> pd.Series:
    """A backward-looking 5-bar moving average -- point-in-time safe."""
    return history["close"].rolling(5).mean()


def leaky_next_return(history: pd.DataFrame) -> pd.Series:
    """LEAKY: uses tomorrow's close (shift(-1)) -- must be caught by the audit."""
    return history["close"].shift(-1) - history["close"]


def leaky_full_mean(history: pd.DataFrame) -> pd.Series:
    """LEAKY: subtracts the whole-sample mean (depends on future rows)."""
    return history["close"] - history["close"].mean()


# --- registry ----------------------------------------------------------------


def test_register_and_list() -> None:
    register_pit_check("causal_sma", causal_sma)
    assert "causal_sma" in audit.registered_checks()


def test_duplicate_registration_raises() -> None:
    register_pit_check("causal_sma", causal_sma)
    with pytest.raises(ValueError, match="already registered"):
        register_pit_check("causal_sma", causal_sma)


# --- shift test --------------------------------------------------------------


def test_causal_feature_passes() -> None:
    run_pit_check(causal_sma, _history())  # no raise


def test_leaky_shift_is_caught() -> None:
    with pytest.raises(LookaheadError, match="leaky_next_return"):
        run_pit_check(leaky_next_return, _history(), name="leaky_next_return")


def test_leaky_full_mean_is_caught() -> None:
    with pytest.raises(LookaheadError):
        run_pit_check(leaky_full_mean, _history(), name="leaky_full_mean")


def test_run_pit_audit_reports_violation() -> None:
    register_pit_check("causal_sma", causal_sma)
    register_pit_check("leaky_next_return", leaky_next_return)
    report = run_pit_audit(_history())
    assert set(report.checked) == {"causal_sma", "leaky_next_return"}
    assert not report.ok
    assert "leaky_next_return" in report.violations
    assert "causal_sma" not in report.violations


def test_run_pit_audit_all_clean_is_ok() -> None:
    register_pit_check("causal_sma", causal_sma)
    report = run_pit_audit(_history())
    assert report.ok
    assert report.violations == {}


# --- CLI entry point ---------------------------------------------------------


def test_main_passes_with_clean_registry() -> None:
    register_pit_check("causal_sma", causal_sma)
    assert audit.main([]) == 0


def test_main_fails_on_leak() -> None:
    register_pit_check("leaky_next_return", leaky_next_return)
    assert audit.main([]) == 1


def test_main_empty_registry_passes() -> None:
    assert audit.main([]) == 0
