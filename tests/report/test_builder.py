"""build_report artifact + honest-DSR-count acceptance (no Nautilus needed).

Constructs a ``BacktestResult`` by hand so the report builder is exercised in
isolation from the event-driven engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from futures_engine.backtest.engine import BacktestResult
from futures_engine.core.manifest import RunManifest
from futures_engine.costs.model import CostConfig
from futures_engine.prop.survival import SurvivalReport
from futures_engine.report.builder import GateConfig, build_report, compute_bundle
from futures_engine.trials.logger import TrialLogger, TrialRecord

_FAMILY = "trend_momentum"
_CREATED = datetime(2026, 7, 27, tzinfo=UTC)
_COST = CostConfig(
    commission_per_side_usd=0.15,
    exchange_fee_per_side_usd=0.35,
    nfa_fee_per_side_usd=0.01,
    spread_ticks=1.0,
    slippage="fixed_ticks",
    slippage_ticks=1.0,
    delay_bars=0,
)


@dataclass
class _FakeResult:
    """Minimal BacktestResultLike stand-in for gross/delayed variants."""

    returns: pd.Series
    equity: pd.Series
    metrics: Mapping[str, float]


def _returns(mean: float, seed: int, n: int = 40) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="1h", tz="UTC")
    return pd.Series(rng.normal(mean, 0.01, size=n), index=idx, name="returns")


def _metrics(returns: pd.Series) -> dict[str, float]:
    std = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / std) if std > 0 else 0.0
    return {
        "sharpe": sharpe,
        "win_rate": float((returns > 0).mean()),
        "n_trades": float(len(returns)),
        "max_dd": 0.05,
    }


def _result(returns: pd.Series, n_snap: str = "deadbeef") -> BacktestResult:
    equity = pd.Series(100_000.0 + np.cumsum(returns.to_numpy() * 100_000.0), index=returns.index)
    metrics = _metrics(returns)
    manifest = RunManifest.create(
        created_at=_CREATED,
        config_hash="c" * 64,
        data_snapshot_hashes=[n_snap],
        seed=7,
        trial_ids=["t-0"],
        run_id="run-0",
        git_sha="a" * 40,
    )
    return BacktestResult(
        trades=pd.DataFrame({"net_pnl_usd": returns.to_numpy() * 100_000.0}),
        equity=equity,
        returns=returns,
        fills=pd.DataFrame(),
        metrics=metrics,
        manifest=manifest,
    )


def _logger_with_trials(tmp_path: Path, n: int) -> TrialLogger:
    tmp_path.mkdir(parents=True, exist_ok=True)
    logger = TrialLogger(tmp_path / "trials.db")
    for i in range(n):
        logger.log(
            TrialRecord(
                trial_id=f"trial-{i:04d}",
                run_id="run-0",
                ts=_CREATED,
                strategy_family=_FAMILY,
                config_hash=f"{i:064d}",
                params={"window": 10 + i},
                data_snapshot_hashes=["deadbeef"],
                cv_scheme="PurgedKFold(embargo_frac=0.02,n_splits=5)",
                metrics={"sharpe_net": 0.01 * i},
                seed=7,
                git_sha="a" * 40,
            )
        )
    return logger


def _survival(p: float) -> SurvivalReport:
    return SurvivalReport(
        p_survival=p,
        p_target_before_bust=0.05,
        median_days_to_target=None,
        bust_reasons={"trailing_drawdown": 1.0 - p},
        ci_90=(max(0.0, p - 0.05), min(1.0, p + 0.05)),
    )


def test_build_report_writes_artifacts_and_honest_trial_count(tmp_path: Path) -> None:
    net = _returns(0.0002, seed=1)
    result = _result(net)
    gross = _FakeResult(net, result.equity, _metrics(_returns(0.0004, seed=2)))
    delayed = _FakeResult(net, result.equity, _metrics(_returns(0.00019, seed=3)))
    logger = _logger_with_trials(tmp_path, n=17)
    perf = np.random.default_rng(0).normal(0.0, 1.0, size=(6, 8))

    run_dir = build_report(
        "demo-run",
        logger,
        _survival(0.80),
        result,
        GateConfig(),
        out_dir=tmp_path / "reports",
        gross=gross,
        delayed=delayed,
        perf_matrix=perf,
        strategy_family=_FAMILY,
        cv_scheme="PurgedKFold(embargo_frac=0.02,n_splits=5)",
        cost_cfg=_COST,
        pbo_partitions=8,
        bootstrap_n=200,
        bootstrap_seed=0,
    )

    md = (run_dir / "report.md").read_text(encoding="utf-8")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert (run_dir / "verdict.json").exists()
    assert (run_dir / "manifest.json").exists()

    # Honest DSR n_trials must be exactly the logged count, not a constant.
    assert "**17**" in md
    assert logger.count(_FAMILY) == 17
    # Mandatory report contents (G10).
    tokens = ("Deflated Sharpe", "PBO", "Bootstrap", "survival", "Red flags", "trial-list hash")
    for token in tokens:
        assert token in md
    assert "Verdict:" in html
    assert "config hash" in md


def test_bundle_dsr_uses_logger_count_not_hardcoded(tmp_path: Path) -> None:
    net = _returns(0.0002, seed=5)
    result = _result(net)
    gross = _FakeResult(net, result.equity, _metrics(_returns(0.00025, seed=6)))
    delayed = _FakeResult(net, result.equity, _metrics(net))
    perf = np.random.default_rng(1).normal(0.0, 1.0, size=(4, 8))

    small = _logger_with_trials(tmp_path / "a", n=3)
    big = _logger_with_trials(tmp_path / "b", n=250)

    def dsr_for(logger: TrialLogger) -> float:
        return compute_bundle(
            "r",
            logger,
            _survival(0.9),
            result,
            GateConfig(),
            gross=gross,
            delayed=delayed,
            perf_matrix=perf,
            strategy_family=_FAMILY,
            cv_scheme="cv",
            bootstrap_n=100,
        ).dsr_p

    # More trials => a higher deflation benchmark => a strictly lower DSR. This
    # only holds if n_trials is read from the logger (G10), never hard-coded.
    assert dsr_for(big) < dsr_for(small)
