"""Per-run research report: the GO/NO-GO verdict and the Markdown+HTML artifact.

The report is the project's Definition-of-Done gate. :func:`decide_verdict` is
the highest-stakes logic in the whole system (G16): it turns the anti-overfitting
statistics and the prop-survival Monte Carlo into a strict GO / NO-GO. A strategy
is only ever believed when *every* gate in :class:`GateConfig` passes and no
``severity="fail"`` red flag fired.
"""

from __future__ import annotations

from futures_engine.report.builder import (
    GateConfig,
    GateResult,
    ReportBundle,
    Verdict,
    build_report,
    decide_verdict,
)

__all__ = [
    "GateConfig",
    "GateResult",
    "ReportBundle",
    "Verdict",
    "build_report",
    "decide_verdict",
]
