"""Event labeling: triple-barrier, meta-labels, uniqueness weights (task T4, G6)."""

from __future__ import annotations

from futures_engine.labels.triple_barrier import (
    LABEL_COLUMNS,
    Labels,
    fixed_horizon_labels,
    meta_labels,
    triple_barrier,
    uniqueness_weights,
)

__all__ = [
    "LABEL_COLUMNS",
    "Labels",
    "fixed_horizon_labels",
    "meta_labels",
    "triple_barrier",
    "uniqueness_weights",
]
