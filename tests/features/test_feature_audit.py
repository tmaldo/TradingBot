"""Every registered feature must pass the T1 look-ahead (point-in-time) audit.

This is the crux of task T4 (architect note 2): the CI ``audit-run`` goes from
"no checks registered" to shift-testing the whole feature list. We assert the
registry is non-empty *after importing the features package* (in a fresh
interpreter, so the assertion is independent of other tests' registry state),
that ``register_all`` is idempotent, that every registered feature survives the
shift test on the deterministic reference history, and -- the highest-stakes
guard -- that a feature which peeks one bar ahead (``shift(-1)``) is rejected.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import pandas as pd
import pytest

from futures_engine.data import audit
from futures_engine.data.audit import (
    LookaheadError,
    reference_history,
    registered_checks,
    run_pit_audit,
    run_pit_check,
)
from futures_engine.features.builder import FEATURE_COLUMNS, register_all


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    audit.clear_registry()
    yield
    audit.clear_registry()


def test_import_of_features_package_registers_checks() -> None:
    """A fresh interpreter that imports the package has a non-empty registry."""
    script = (
        "import futures_engine.features\n"
        "from futures_engine.data.audit import registered_checks\n"
        "checks = registered_checks()\n"
        "assert any(k.startswith('feature.') for k in checks), sorted(checks)\n"
        "assert len(checks) >= 16, len(checks)\n"
        "print('OK', len(checks))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


def test_register_all_registers_every_feature_column() -> None:
    register_all()
    checks = registered_checks()
    expected = {f"feature.{c}" for c in FEATURE_COLUMNS}
    assert expected <= set(checks)


def test_register_all_is_idempotent() -> None:
    register_all()
    n_first = len(registered_checks())
    register_all()  # must not raise on already-registered names
    assert len(registered_checks()) == n_first


def test_every_registered_feature_passes_shift_audit() -> None:
    register_all()
    report = run_pit_audit(reference_history())
    assert report.ok, report.violations
    assert {f"feature.{c}" for c in FEATURE_COLUMNS} <= set(report.checked)


def _leaky_future_return(history: pd.DataFrame) -> pd.Series:
    """LEAKY: uses tomorrow's close via shift(-1); the audit must reject it."""
    out: pd.Series = history["close"].shift(-1) / history["close"] - 1.0
    return out


def test_leaky_shift_feature_is_rejected_by_audit() -> None:
    with pytest.raises(LookaheadError):
        run_pit_check(_leaky_future_return, reference_history(), name="feature.leaky_future")


def test_leaky_feature_flagged_in_full_audit() -> None:
    register_all()
    report = run_pit_audit(
        reference_history(),
        checks={**dict(registered_checks()), "feature.leaky": _leaky_future_return},
    )
    assert not report.ok
    assert "feature.leaky" in report.violations
