"""Feature, indicator and fractional-differencing library (task T4).

Importing this package **registers every point-in-time feature check** into the
look-ahead audit (:mod:`futures_engine.data.audit`) via :func:`register_all`, so
CI's ``audit-run`` shift-tests the full feature list. Registration is idempotent.
"""

from __future__ import annotations

from futures_engine.features.builder import (
    FEATURE_COLUMNS,
    FeatureConfig,
    build_features,
    feature_functions,
    register_all,
)

register_all()

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureConfig",
    "build_features",
    "feature_functions",
    "register_all",
]
