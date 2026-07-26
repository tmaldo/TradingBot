"""Market-regime detectors (task T4, G4/G6).

Importing this package registers the causal regime shift-check with the
look-ahead audit (:func:`register_regime_checks`), so CI shift-tests the regime
labeling path alongside the features. Registration is idempotent.
"""

from __future__ import annotations

from futures_engine.regime.detector import (
    ChangePointRegimeDetector,
    HMMRegimeDetector,
    RegimeDetector,
    register_regime_checks,
)

register_regime_checks()

__all__ = [
    "ChangePointRegimeDetector",
    "HMMRegimeDetector",
    "RegimeDetector",
    "register_regime_checks",
]
