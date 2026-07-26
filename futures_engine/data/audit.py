"""Automated look-ahead (point-in-time) audit (Global Constraint G4).

A *PIT check* is a feature function ``fn(history: Bars) -> Series | DataFrame``
whose value at each timestamp must depend only on information available at that
timestamp. The audit proves this with a **shift test**: it recomputes ``fn`` with
the input history truncated at several cut points and asserts the values at and
before each cut are byte-for-byte unchanged. If truncating the future changes a
past value, the feature peeked ahead -- a :class:`LookaheadError`.

Downstream (T4) feature modules call :func:`register_pit_check` at import time;
CI runs ``pytest tests/data/test_lookahead_audit.py`` plus the ``audit-run``
entry point (:func:`main`), which shift-tests every registered check against a
deterministic reference history and exits non-zero on any leak.

No wall-clock reads and no network: the reference history is a closed-form
function of a fixed origin date (G15).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from futures_engine.core.types import Bars
from futures_engine.data.store import DataIntegrityError

PitCheck = Callable[[Bars], "pd.Series | pd.DataFrame"]

# Fractions of the history length at which to truncate and recompute.
DEFAULT_CUT_FRACTIONS: tuple[float, ...] = (0.5, 0.75, 0.9)
# Absolute tolerance for comparing recomputed values (float noise, not leakage).
_COMPARE_ATOL = 1e-9

_REGISTRY: dict[str, PitCheck] = {}


class LookaheadError(DataIntegrityError):
    """Raised when a feature's past values change once future input is withheld."""


def register_pit_check(name: str, fn: PitCheck) -> None:
    """Register ``fn`` under ``name`` for the look-ahead audit. Rejects duplicates."""
    if name in _REGISTRY:
        raise ValueError(f"PIT check {name!r} is already registered")
    _REGISTRY[name] = fn


def registered_checks() -> Mapping[str, PitCheck]:
    """Return a read-only view of the registered checks."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Remove all registered checks (test isolation)."""
    _REGISTRY.clear()


@dataclass(frozen=True)
class AuditReport:
    """Outcome of a :func:`run_pit_audit` pass."""

    checked: list[str]
    violations: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def _to_frame(result: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, pd.DataFrame):
        return result
    raise DataIntegrityError(
        f"PIT check must return a Series or DataFrame, got {type(result).__name__}"
    )


def _cut_timestamps(index: pd.Index, cut_fractions: tuple[float, ...]) -> list[pd.Timestamp]:
    n = len(index)
    positions = sorted({int(f * (n - 1)) for f in cut_fractions if 0.0 < f < 1.0})
    return [pd.Timestamp(index[p]) for p in positions if 0 < p < n - 1]


def _prefix_mismatch(full: pd.DataFrame, truncated: pd.DataFrame, cut: pd.Timestamp) -> bool:
    """True if ``full`` and ``truncated`` disagree at any timestamp <= ``cut``."""
    common = full.index.intersection(truncated.index)
    common = common[common <= cut]
    if len(common) == 0:
        return False
    a = full.loc[common]
    b = truncated.loc[common]
    if list(a.columns) != list(b.columns):
        return True
    return not np.allclose(
        a.to_numpy(dtype="float64"),
        b.to_numpy(dtype="float64"),
        atol=_COMPARE_ATOL,
        equal_nan=True,
    )


def run_pit_check(
    check: PitCheck,
    history: Bars,
    *,
    name: str | None = None,
    cut_fractions: tuple[float, ...] = DEFAULT_CUT_FRACTIONS,
) -> None:
    """Shift-test a single ``check`` over ``history``; raise on any leak.

    Recomputes ``check`` with history truncated at each cut point and compares the
    at-or-before-cut values to the full-history result. Any difference means a
    past value depended on withheld future data -> :class:`LookaheadError`.
    """
    label = name or getattr(check, "__name__", "pit_check")
    if not history.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{label}: history index must be sorted ascending")
    full = _to_frame(check(history))
    for cut in _cut_timestamps(history.index, cut_fractions):
        truncated_history = history.loc[history.index <= cut]
        recomputed = _to_frame(check(truncated_history))
        if _prefix_mismatch(full, recomputed, cut):
            raise LookaheadError(
                f"{label}: values at/before {cut.isoformat()} changed when future "
                "input was withheld (look-ahead leak)"
            )


def run_pit_audit(
    history: Bars,
    *,
    checks: Mapping[str, PitCheck] | None = None,
    cut_fractions: tuple[float, ...] = DEFAULT_CUT_FRACTIONS,
) -> AuditReport:
    """Shift-test every check (registered ones by default) and return a report."""
    selected = dict(checks) if checks is not None else dict(_REGISTRY)
    violations: dict[str, str] = {}
    for name, check in selected.items():
        try:
            run_pit_check(check, history, name=name, cut_fractions=cut_fractions)
        except LookaheadError as exc:
            violations[name] = str(exc)
    return AuditReport(checked=list(selected), violations=violations)


def reference_history(periods: int = 250) -> Bars:
    """A deterministic OHLCV history for the CLI audit (no randomness/wall clock)."""
    origin = datetime(2020, 1, 1, tzinfo=UTC)
    idx = pd.DatetimeIndex(pd.date_range(origin, periods=periods, freq="B", tz="UTC").tolist())
    steps = np.sin(np.arange(periods) * 0.1) * 10.0 + np.arange(periods) * 0.25
    close = pd.Series(1000.0 + steps, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": pd.Series(np.full(periods, 5000.0), index=idx),
        },
        index=idx,
    )


def main(argv: list[str] | None = None) -> int:
    """``audit-run`` entry point: shift-test all registered checks; exit non-zero on leak."""
    _ = argv  # no options yet; reserved for future dataset selection
    checks = registered_checks()
    if not checks:
        print("look-ahead audit: no PIT checks registered (nothing to verify)")
        return 0
    report = run_pit_audit(reference_history())
    print(f"look-ahead audit: shift-tested {len(report.checked)} check(s)")
    for name in report.checked:
        status = "LEAK" if name in report.violations else "ok"
        print(f"  [{status}] {name}")
    if not report.ok:
        for name, detail in report.violations.items():
            print(f"VIOLATION {name}: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
