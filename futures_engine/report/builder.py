"""Per-run research report: GO/NO-GO verdict + self-contained Markdown/HTML.

:func:`decide_verdict` is the crux (G16). :func:`build_report` assembles the
mandatory anti-overfitting outputs (G10) -- the Deflated Sharpe Ratio with an
**honest** trial count sourced from the :class:`~futures_engine.trials.logger.TrialLogger`,
the Probability of Backtest Overfitting, a bootstrap Sharpe confidence interval,
the prop-survival probability with its CI, gross-vs-net performance, and every
red flag -- then derives the verdict strictly from :class:`GateConfig` and writes
a run-scoped Markdown + HTML artifact plus the ``RunManifest`` for reproducibility.

Signature refinement (disclosed per the task brief)
---------------------------------------------------
The plan sketched ``build_report(run_id, logger, survival, result, gates) -> Path``.
The report needs a few more *inputs* to compute its mandatory contents without
recomputing them from scratch, so :func:`build_report` keeps that positional core
and the :class:`GateConfig` defaults and adds keyword-only parameters: the
``gross`` / ``delayed`` cost variants (for the red-flag edge-decay checks), the
per-config ``perf_matrix`` (the CSCV input for PBO -- one Sharpe cell per config
per time slice, which a single ``BacktestResult`` cannot carry), the sweep's
``strategy_family`` / ``cv_scheme``, the ``cost_cfg`` shown in the report, and the
output directory. Everything else is derived internally.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.backtest.engine import BacktestResult
from futures_engine.costs.model import CostConfig
from futures_engine.prop.survival import SurvivalReport
from futures_engine.trials.logger import TrialLogger, TrialRecord
from futures_engine.validation.stats import (
    BacktestResultLike,
    RedFlag,
    RedFlagConfig,
    bootstrap_sharpe_ci,
    deflated_sharpe,
    pbo,
    red_flags,
)

FloatArray = npt.NDArray[np.float64]

# Non-excess (Pearson) kurtosis of a normal distribution; deflated_sharpe expects
# gamma4 on this convention, while pandas Series.kurt() returns EXCESS kurtosis.
_NORMAL_KURTOSIS = 3.0

Decision = Literal["GO", "NO-GO"]


# --- gate configuration & verdict -------------------------------------------


class GateConfig(BaseModel):
    """Thresholds the verdict is derived from (G15/G16: no magic constants).

    Defaults encode the project's Definition of Done: the Deflated Sharpe Ratio
    must clear ``min_dsr_p`` (probability the edge is real), the Probability of
    Backtest Overfitting must sit at or below ``max_pbo``, and the prop-account
    survival probability must be at least ``min_survival``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_dsr_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_pbo: float = Field(default=0.5, ge=0.0, le=1.0)
    min_survival: float = Field(default=0.90, ge=0.0, le=1.0)


class GateResult(BaseModel):
    """One gate's pass/fail outcome plus a human-readable justification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class Verdict(BaseModel):
    """The strict GO/NO-GO decision and the evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Decision
    gates: list[GateResult]
    fail_flag_codes: list[str]

    @property
    def is_go(self) -> bool:
        return self.decision == "GO"


def decide_verdict(
    *,
    dsr_p: float,
    pbo: float,
    p_survival: float,
    red_flags: list[RedFlag],
    gates: GateConfig,
) -> Verdict:
    """Derive the strict verdict (G16).

    GO **iff** every gate passes AND no ``severity="fail"`` red flag fired:

    * survival gate: ``p_survival >= gates.min_survival``;
    * DSR gate: ``dsr_p >= gates.min_dsr_p``;
    * PBO gate: ``pbo <= gates.max_pbo``;
    * red-flag gate: no red flag has ``severity == "fail"``.

    Any single failure yields NO-GO. Warn-severity flags never change the verdict.
    """
    fail_codes = [f.code for f in red_flags if f.severity == "fail"]
    gate_results = [
        GateResult(
            name="survival",
            passed=p_survival >= gates.min_survival,
            detail=f"p_survival={p_survival:.4f} vs min {gates.min_survival:.4f}",
        ),
        GateResult(
            name="deflated_sharpe",
            passed=dsr_p >= gates.min_dsr_p,
            detail=f"DSR={dsr_p:.4f} vs min {gates.min_dsr_p:.4f}",
        ),
        GateResult(
            name="pbo",
            passed=pbo <= gates.max_pbo,
            detail=f"PBO={pbo:.4f} vs max {gates.max_pbo:.4f}",
        ),
        GateResult(
            name="no_fail_flags",
            passed=len(fail_codes) == 0,
            detail=(
                "no fail-severity red flags"
                if not fail_codes
                else f"fail-severity red flags: {', '.join(fail_codes)}"
            ),
        ),
    ]
    decision: Decision = "GO" if all(g.passed for g in gate_results) else "NO-GO"
    return Verdict(decision=decision, gates=gate_results, fail_flag_codes=fail_codes)


# --- computed report bundle --------------------------------------------------


@dataclass(frozen=True)
class ReportBundle:
    """Every number the rendered report shows, plus the verdict.

    Kept separate from the rendering so the Markdown and HTML views are two
    projections of one deterministic, testable computation.
    """

    run_id: str
    strategy_family: str
    cv_scheme: str
    verdict: Verdict
    gates: GateConfig
    # deflated Sharpe inputs/outputs
    observed_sharpe: float
    dsr_p: float
    n_trials: int
    n_obs: int
    trial_list_hash: str
    # cross-validation / overfitting
    pbo: float
    pbo_partitions: int
    n_configs: int
    # bootstrap CI
    sharpe_ci_low: float
    sharpe_ci_high: float
    # survival
    survival: SurvivalReport
    # performance
    net_sharpe: float
    gross_sharpe: float
    net_win_rate: float
    net_n_trades: int
    net_max_dd: float
    # red flags
    red_flags: list[RedFlag]
    # provenance
    config_hash: str
    data_snapshot_hashes: list[str]
    git_sha: str
    seed: int
    created_at: datetime


# --- honest DSR helpers ------------------------------------------------------


def _safe_moments(returns: pd.Series) -> tuple[float, float]:
    """Return ``(skew, non_excess_kurtosis)`` of ``returns``; normal on degeneracy.

    pandas needs >=3 observations for skew and >=4 for kurtosis and returns NaN
    otherwise; a NaN or non-finite moment falls back to the normal convention
    (skew 0, kurtosis 3) so the DSR is still computable rather than crashing.
    """
    skew = float(cast("float", returns.skew())) if len(returns) >= 3 else 0.0
    excess_kurt = float(cast("float", returns.kurt())) if len(returns) >= 4 else 0.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(excess_kurt):
        excess_kurt = 0.0
    return skew, excess_kurt + _NORMAL_KURTOSIS


def _honest_dsr(observed_sr: float, n_trials: int, returns: pd.Series) -> float:
    """Deflated Sharpe with an honest ``n_trials``; fail-safe to 0.0 on degeneracy.

    ``n_trials`` MUST originate from :meth:`TrialLogger.count` (G10) -- the caller
    passes exactly that. A degenerate moment set (too few observations, a
    non-positive Sharpe variance term) means the edge is not credibly significant,
    so we report 0.0 -- which fails the DSR gate, the safe default.
    """
    n_obs = len(returns)
    if n_obs < 2 or n_trials < 1:
        return 0.0
    skew, kurt = _safe_moments(returns)
    try:
        return deflated_sharpe(observed_sr, n_trials, n_obs, skew, kurt)
    except ValueError:
        return 0.0


def _trial_list_hash(records: list[TrialRecord]) -> str:
    """Deterministic 64-hex hash of the logged trial list (G10 provenance).

    Hashes each trial's identity + config hash + metrics in insertion order, so
    the report pins *which* trials the honest DSR count was computed over.
    """
    payload = [
        {
            "trial_id": r.trial_id,
            "config_hash": r.config_hash,
            "cv_scheme": r.cv_scheme,
            "metrics": r.metrics,
        }
        for r in records
    ]
    blob = b"FE-TRIALLIST-v1\n" + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


# --- assembly ----------------------------------------------------------------


def compute_bundle(
    run_id: str,
    logger: TrialLogger,
    survival: SurvivalReport,
    result: BacktestResult,
    gates: GateConfig,
    *,
    gross: BacktestResultLike,
    delayed: BacktestResultLike,
    perf_matrix: FloatArray,
    strategy_family: str,
    cv_scheme: str,
    pbo_partitions: int = 8,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
    red_flag_config: RedFlagConfig | None = None,
) -> ReportBundle:
    """Compute every report number and the verdict (no I/O)."""
    observed_sharpe = float(result.metrics["sharpe"])

    # Honest DSR: n_trials comes straight from the trial store (G10).
    n_trials = logger.count(strategy_family)
    records = logger.all(strategy_family)
    dsr_p = _honest_dsr(observed_sharpe, n_trials, result.returns)

    matrix = np.asarray(perf_matrix, dtype=np.float64)
    n_configs = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    pbo_value = pbo(matrix, pbo_partitions)

    if len(result.returns) >= 2:
        ci_low, ci_high = bootstrap_sharpe_ci(
            result.returns, n_boot=bootstrap_n, seed=bootstrap_seed
        )
    else:
        ci_low, ci_high = 0.0, 0.0

    flags = red_flags(cast("BacktestResultLike", result), delayed, gross, red_flag_config)

    verdict = decide_verdict(
        dsr_p=dsr_p,
        pbo=pbo_value,
        p_survival=survival.p_survival,
        red_flags=flags,
        gates=gates,
    )

    manifest = result.manifest
    return ReportBundle(
        run_id=run_id,
        strategy_family=strategy_family,
        cv_scheme=cv_scheme,
        verdict=verdict,
        gates=gates,
        observed_sharpe=observed_sharpe,
        dsr_p=dsr_p,
        n_trials=n_trials,
        n_obs=len(result.returns),
        trial_list_hash=_trial_list_hash(records),
        pbo=pbo_value,
        pbo_partitions=pbo_partitions,
        n_configs=n_configs,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        survival=survival,
        net_sharpe=observed_sharpe,
        gross_sharpe=float(gross.metrics["sharpe"]),
        net_win_rate=float(result.metrics["win_rate"]),
        net_n_trades=int(result.metrics.get("n_trades", len(result.returns))),
        net_max_dd=float(result.metrics.get("max_dd", 0.0)),
        red_flags=flags,
        config_hash=manifest.config_hash,
        data_snapshot_hashes=list(manifest.data_snapshot_hashes),
        git_sha=manifest.git_sha,
        seed=manifest.seed,
        created_at=manifest.created_at,
    )


# --- rendering ---------------------------------------------------------------


def _flags_rows_md(flags: list[RedFlag]) -> str:
    if not flags:
        return "_None._"
    lines = ["| code | severity | message |", "| --- | --- | --- |"]
    lines += [f"| {f.code} | {f.severity} | {f.message} |" for f in flags]
    return "\n".join(lines)


def render_markdown(b: ReportBundle, cost_cfg: CostConfig) -> str:
    """Render the report bundle as a self-contained Markdown document."""
    v = b.verdict
    gate_lines = "\n".join(
        f"- [{'x' if g.passed else ' '}] **{g.name}** — {g.detail}" for g in v.gates
    )
    bust = (
        "\n".join(f"  - `{k}`: {val:.3f}" for k, val in sorted(b.survival.bust_reasons.items()))
        or "  - _none_"
    )
    return f"""# Research report — `{b.run_id}`

## Verdict: {v.decision}

A run is **GO only if every gate passes and no fail-severity red flag fired**
(thresholds from `GateConfig`: DSR ≥ {b.gates.min_dsr_p:.2f}, PBO ≤ \
{b.gates.max_pbo:.2f}, survival ≥ {b.gates.min_survival:.2f}).

{gate_lines}

## Performance — gross vs net

| metric | value |
| --- | --- |
| net Sharpe (per trade) | {b.net_sharpe:.4f} |
| gross Sharpe (per trade) | {b.gross_sharpe:.4f} |
| net win rate | {b.net_win_rate:.2%} |
| net trades | {b.net_n_trades} |
| net max drawdown | {b.net_max_dd:.2%} |

## Deflated Sharpe Ratio (honest trial count, G10)

| quantity | value |
| --- | --- |
| observed Sharpe | {b.observed_sharpe:.4f} |
| **n_trials (from TrialLogger)** | **{b.n_trials}** |
| n_obs | {b.n_obs} |
| DSR (P[edge is real]) | {b.dsr_p:.4f} |
| trial-list hash | `{b.trial_list_hash}` |

The trial count is read directly from the append-only trial store for the
`{b.strategy_family}` family — never hard-coded — so the deflation benchmark
reflects every configuration actually tried.

## Cross-validation & overfitting

| quantity | value |
| --- | --- |
| CV scheme (sweep) | `{b.cv_scheme}` |
| configurations swept | {b.n_configs} |
| PBO partitions (CSCV) | {b.pbo_partitions} |
| PBO (P[backtest overfit]) | {b.pbo:.4f} |

## Bootstrap Sharpe confidence interval

Stationary block-bootstrap 95% CI of the per-trade Sharpe:
**[{b.sharpe_ci_low:.4f}, {b.sharpe_ci_high:.4f}]**.

## Prop-account survival Monte Carlo (G11)

| quantity | value |
| --- | --- |
| p_survival | {b.survival.p_survival:.4f} |
| 90% CI | [{b.survival.ci_90[0]:.4f}, {b.survival.ci_90[1]:.4f}] |
| p_target_before_bust | {b.survival.p_target_before_bust:.4f} |
| median days to target | {b.survival.median_days_to_target} |

Bust reasons (fraction of paths):
{bust}

## Red flags

{_flags_rows_md(b.red_flags)}

## Costs applied

| field | value |
| --- | --- |
| commission / side | {cost_cfg.commission_per_side_usd} |
| exchange fee / side | {cost_cfg.exchange_fee_per_side_usd} |
| NFA fee / side | {cost_cfg.nfa_fee_per_side_usd} |
| spread (ticks) | {cost_cfg.spread_ticks} |
| slippage | {cost_cfg.slippage} ({cost_cfg.slippage_ticks}) |
| delay bars | {cost_cfg.delay_bars} |

## Reproducibility (RunManifest)

| field | value |
| --- | --- |
| config hash | `{b.config_hash}` |
| data snapshot hashes | {", ".join(f"`{h}`" for h in b.data_snapshot_hashes)} |
| git SHA | `{b.git_sha}` |
| seed | {b.seed} |
| created at | {b.created_at.isoformat()} |
"""


def _flags_rows_html(flags: list[RedFlag]) -> str:
    if not flags:
        return "<p><em>None.</em></p>"
    rows = "".join(
        f"<tr><td>{f.code}</td><td>{f.severity}</td><td>{f.message}</td></tr>" for f in flags
    )
    return (
        "<table><thead><tr><th>code</th><th>severity</th><th>message</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(b: ReportBundle, cost_cfg: CostConfig) -> str:
    """Render the bundle as a single self-contained HTML page (no external assets)."""
    v = b.verdict
    banner_bg = "#0b7a34" if v.is_go else "#8a1f1f"
    gate_items = "".join(
        f"<li><strong>{g.name}</strong>: {'PASS' if g.passed else 'FAIL'} — {g.detail}</li>"
        for g in v.gates
    )
    snapshots_html = "</code>, <code>".join(b.data_snapshot_hashes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research report — {b.run_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 900px;
         padding: 1.5rem; color: #1a1a1a; }}
  .banner {{ background: {banner_bg}; color: #fff; padding: 1rem 1.25rem;
             border-radius: 8px; font-size: 1.6rem; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: left;
            font-size: 0.92rem; }}
  th {{ background: #f2f2f2; }}
  code {{ background: #f4f4f4; padding: 0 0.2rem; word-break: break-all; }}
  h2 {{ margin-top: 1.8rem; border-bottom: 1px solid #eee; padding-bottom: 0.2rem; }}
</style></head><body>
<div class="banner">Verdict: {v.decision}</div>
<p>Run <code>{b.run_id}</code> — strategy family <code>{b.strategy_family}</code>.
GO only if every gate passes and no fail-severity red flag fired.</p>
<h2>Gates</h2><ul>{gate_items}</ul>
<h2>Performance — gross vs net</h2>
<table><tbody>
<tr><th>net Sharpe</th><td>{b.net_sharpe:.4f}</td></tr>
<tr><th>gross Sharpe</th><td>{b.gross_sharpe:.4f}</td></tr>
<tr><th>net win rate</th><td>{b.net_win_rate:.2%}</td></tr>
<tr><th>net trades</th><td>{b.net_n_trades}</td></tr>
<tr><th>net max drawdown</th><td>{b.net_max_dd:.2%}</td></tr>
</tbody></table>
<h2>Deflated Sharpe (honest trial count)</h2>
<table><tbody>
<tr><th>observed Sharpe</th><td>{b.observed_sharpe:.4f}</td></tr>
<tr><th>n_trials (from TrialLogger)</th><td>{b.n_trials}</td></tr>
<tr><th>n_obs</th><td>{b.n_obs}</td></tr>
<tr><th>DSR</th><td>{b.dsr_p:.4f}</td></tr>
<tr><th>trial-list hash</th><td><code>{b.trial_list_hash}</code></td></tr>
</tbody></table>
<h2>Cross-validation &amp; overfitting</h2>
<table><tbody>
<tr><th>CV scheme</th><td><code>{b.cv_scheme}</code></td></tr>
<tr><th>configs swept</th><td>{b.n_configs}</td></tr>
<tr><th>PBO partitions</th><td>{b.pbo_partitions}</td></tr>
<tr><th>PBO</th><td>{b.pbo:.4f}</td></tr>
</tbody></table>
<h2>Bootstrap Sharpe CI</h2>
<p>95% stationary block-bootstrap CI:
<strong>[{b.sharpe_ci_low:.4f}, {b.sharpe_ci_high:.4f}]</strong>.</p>
<h2>Prop-account survival (G11)</h2>
<table><tbody>
<tr><th>p_survival</th><td>{b.survival.p_survival:.4f}</td></tr>
<tr><th>90% CI</th><td>[{b.survival.ci_90[0]:.4f}, {b.survival.ci_90[1]:.4f}]</td></tr>
<tr><th>p_target_before_bust</th><td>{b.survival.p_target_before_bust:.4f}</td></tr>
<tr><th>median days to target</th><td>{b.survival.median_days_to_target}</td></tr>
</tbody></table>
<h2>Red flags</h2>{_flags_rows_html(b.red_flags)}
<h2>Reproducibility</h2>
<table><tbody>
<tr><th>config hash</th><td><code>{b.config_hash}</code></td></tr>
<tr><th>data snapshots</th><td><code>{snapshots_html}</code></td></tr>
<tr><th>git SHA</th><td><code>{b.git_sha}</code></td></tr>
<tr><th>seed</th><td>{b.seed}</td></tr>
<tr><th>created at</th><td>{b.created_at.isoformat()}</td></tr>
<tr><th>delay bars</th><td>{cost_cfg.delay_bars}</td></tr>
</tbody></table>
</body></html>
"""


def build_report(
    run_id: str,
    logger: TrialLogger,
    survival: SurvivalReport,
    result: BacktestResult,
    gates: GateConfig,
    *,
    out_dir: str | Path,
    gross: BacktestResultLike,
    delayed: BacktestResultLike,
    perf_matrix: FloatArray,
    strategy_family: str,
    cv_scheme: str,
    cost_cfg: CostConfig,
    pbo_partitions: int = 8,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
    red_flag_config: RedFlagConfig | None = None,
) -> Path:
    """Write the run-scoped Markdown + HTML report and return its directory (G10/G16).

    Computes the mandatory outputs -- honest DSR (trial count from ``logger``), PBO,
    bootstrap Sharpe CI, survival, gross-vs-net, and red flags -- derives the strict
    verdict from ``gates``, and writes ``report.md``, ``report.html``, ``verdict.json``
    and ``manifest.json`` under ``<out_dir>/<run_id>/``. Deterministic given the same
    inputs and seeds.
    """
    bundle = compute_bundle(
        run_id,
        logger,
        survival,
        result,
        gates,
        gross=gross,
        delayed=delayed,
        perf_matrix=perf_matrix,
        strategy_family=strategy_family,
        cv_scheme=cv_scheme,
        pbo_partitions=pbo_partitions,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        red_flag_config=red_flag_config,
    )

    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.md").write_text(render_markdown(bundle, cost_cfg), encoding="utf-8")
    (run_dir / "report.html").write_text(render_html(bundle, cost_cfg), encoding="utf-8")
    (run_dir / "verdict.json").write_text(
        bundle.verdict.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        result.manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return run_dir
