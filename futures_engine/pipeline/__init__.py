"""Single-command research pipeline: data -> report, all offline.

:func:`run_pipeline` chains the whole system end-to-end for one strategy family:
load a validation-grade snapshot, build features/labels, sweep the parameter grid
(logging one honest trial per combination), confirm the best config on the
event-driven backtester net of costs, run the anti-overfitting validation and the
prop-survival Monte Carlo, and emit the GO/NO-GO report artifact. It is RESEARCH
only -- it never touches the live execution adapters.
"""

from __future__ import annotations

from futures_engine.pipeline.run import (
    PipelineConfig,
    PipelineOutcome,
    SurvivalSettings,
    run_pipeline,
)

__all__ = [
    "PipelineConfig",
    "PipelineOutcome",
    "SurvivalSettings",
    "run_pipeline",
]
