"""End-to-end research pipeline (data -> features/labels -> sweep -> backtest ->
validation -> survival -> report). RESEARCH only: no live execution adapters.

Everything is driven by a validated :class:`PipelineConfig` (YAML, no magic
constants, G15) plus the repo's :class:`~futures_engine.core.config.Settings`
(instrument spec, cost profile, prop-rule preset). One command, fully offline and
deterministic (fixed seed + fixed ``created_at``), producing the GO/NO-GO report.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.backtest.engine import BacktestResult, BacktestRunner, StrategyConfig
from futures_engine.backtest.strategy_adapter import SIGNAL_REGISTRY
from futures_engine.core.config import Settings, load_yaml
from futures_engine.core.manifest import current_git_sha
from futures_engine.core.types import Bars, InstrumentSpec
from futures_engine.costs.model import CostConfig, apply_costs
from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.features.builder import FeatureConfig, build_features
from futures_engine.labels.triple_barrier import triple_barrier, uniqueness_weights
from futures_engine.prop.rules import PropRuleSet
from futures_engine.prop.survival import SurvivalReport, monte_carlo_survival
from futures_engine.report.builder import GateConfig, Verdict, build_report
from futures_engine.research.harness import causal_positions, positions_to_trades
from futures_engine.research.harness import sweep as run_sweep
from futures_engine.trials.logger import TrialLogger
from futures_engine.validation.splitters import PurgedKFold
from futures_engine.validation.stats import BacktestResultLike

FloatArray = npt.NDArray[np.float64]

# Zero every friction: the "gross" backtest variant the red-flag EDGE_FAILS_COSTS
# check compares the net result against (G8/G10).
_GROSS_COST = CostConfig(
    commission_per_side_usd=0.0,
    exchange_fee_per_side_usd=0.0,
    nfa_fee_per_side_usd=0.0,
    spread_ticks=0.0,
    slippage="fixed_ticks",
    slippage_ticks=0.0,
    delay_bars=0,
)

# Deterministic wall-clock stand-in so a run's manifest/report are reproducible.
_DEFAULT_CREATED_AT = datetime(2026, 7, 27, tzinfo=UTC)


# --- config ------------------------------------------------------------------


class SurvivalSettings(BaseModel):
    """Monte-Carlo survival parameters for the pipeline (all from YAML)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_paths: int = Field(gt=0)
    horizon_days: int = Field(gt=0)
    contracts: int = Field(default=1, ge=0)


class PipelineConfig(BaseModel):
    """Validated end-to-end run configuration (G15: no magic constants).

    ``grid`` is the integer parameter sweep for ``signal`` (a key in
    :data:`~futures_engine.backtest.strategy_adapter.SIGNAL_REGISTRY`). ``instrument``
    and ``prop_preset`` resolve against the repo :class:`Settings`. Loaded through
    the same ``extra="forbid"`` pydantic path as every other config, so a typo in
    the demo YAML fails loudly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    # Display/label only: the honest DSR trial count is sourced from the signal's
    # own ``.family`` (see run_pipeline), never this field, so a stale value here
    # cannot silently zero the count.
    strategy_family: str = Field(min_length=1)
    grid: dict[str, list[int]]
    snapshot_hash: str = Field(min_length=1)
    prop_preset: str = Field(min_length=1)
    seed: int = 7
    n_splits: int = Field(default=5, ge=2)
    embargo_frac: float = Field(default=0.02, ge=0.0, lt=1.0)
    pbo_partitions: int = Field(default=8, ge=2)
    bootstrap_n: int = Field(default=500, ge=1)
    survival: SurvivalSettings
    gates: GateConfig = GateConfig()

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a pipeline config YAML."""
        return cls.model_validate(load_yaml(path))


# --- outcome -----------------------------------------------------------------


@dataclass(frozen=True)
class PipelineOutcome:
    """What one end-to-end run produced (the report dir plus the verdict)."""

    run_id: str
    run_dir: Path
    verdict: Verdict
    n_trials: int
    best_params: dict[str, int]

    @property
    def decision(self) -> str:
        return self.verdict.decision


# --- stages ------------------------------------------------------------------


def _features_and_labels(
    bars: Bars, spec: InstrumentSpec, feature_config: FeatureConfig
) -> tuple[int, int]:
    """Run the features + triple-barrier labelling stage; return (n_features, n_labels).

    The reference trend signals are price-based, so the feature matrix and labels
    are not consumed by the sweep here; building them keeps the full research
    chain (and its point-in-time guarantees) exercised end-to-end.
    """
    features = build_features(bars, spec, feature_config)
    vol = bars["close"].pct_change(fill_method=None).rolling(50).std()
    valid = vol.dropna()
    if len(valid) < 3:
        return int(features.shape[1]), 0
    step = max(1, len(valid) // 200)
    events = pd.DatetimeIndex(valid.index[::step], name=bars.index.name)
    labels = triple_barrier(bars, events, pt_mult=2.0, sl_mult=2.0, max_holding_bars=60, vol=vol)
    # exercised for the point-in-time audit trail
    _ = uniqueness_weights(labels, cast("pd.DatetimeIndex", bars.index))
    return int(features.shape[1]), len(labels)


def _combos(grid: dict[str, list[int]]) -> list[dict[str, int]]:
    keys = list(grid.keys())
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def _perf_matrix(
    bars: Bars,
    spec: InstrumentSpec,
    cost_cfg: CostConfig,
    signal: Any,
    combos: list[dict[str, int]],
    n_slices: int,
) -> FloatArray:
    """CSCV input for PBO: one mean net-PnL cell per config per contiguous time slice.

    Row ``i`` = config ``i``; column ``t`` = the mean net PnL of that config's trades
    whose entry bar falls in the ``t``-th contiguous slice of the sample (0 if none).
    Reuses the T5 vectorized fill/cost path so the numbers agree with the sweep.
    """
    n = len(bars)
    edges = np.linspace(0, n, n_slices + 1).astype(int)
    index = bars.index
    rows: list[FloatArray] = []
    for params in combos:
        row = np.zeros(n_slices, dtype=np.float64)
        raw = signal.generate(bars, params)
        held = causal_positions(raw)
        trades = positions_to_trades(held, bars, spec, 1.0)
        if not trades.empty:
            priced = apply_costs(trades, spec, cost_cfg)
            entry_pos = index.get_indexer(pd.Index(priced["entry_ts"]))
            net = priced["net_pnl_usd"].to_numpy(dtype=np.float64)
            for s in range(n_slices):
                mask = (entry_pos >= edges[s]) & (entry_pos < edges[s + 1])
                if mask.any():
                    row[s] = float(net[mask].mean())
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _best_params(best: dict[str, Any], grid: dict[str, list[int]]) -> dict[str, int]:
    """Extract the sweep's best combination as clean ints keyed by the grid axes."""
    if not best:
        raise ValueError("sweep produced no non-errored combination; cannot pick a best config")
    return {key: int(best[key]) for key in grid}


def _cv_scheme(splitter: PurgedKFold) -> str:
    return f"PurgedKFold(n_splits={splitter.n_splits},embargo_frac={splitter.embargo_frac})"


def _run_survival(net: BacktestResult, rules: PropRuleSet, cfg: PipelineConfig) -> SurvivalReport:
    return monte_carlo_survival(
        net.trades,
        rules,
        contracts=cfg.survival.contracts,
        n_paths=cfg.survival.n_paths,
        horizon_days=cfg.survival.horizon_days,
        seed=cfg.seed,
    )


# --- the pipeline ------------------------------------------------------------


def run_pipeline(
    config_path: str | Path,
    *,
    settings_dir: str | Path,
    snapshot_root: str | Path,
    out_dir: str | Path,
    created_at: datetime | None = None,
    feature_config: FeatureConfig | None = None,
    repo: str | Path | None = None,
) -> PipelineOutcome:
    """Run the full research chain offline and write the GO/NO-GO report.

    Parameters
    ----------
    config_path:
        The :class:`PipelineConfig` YAML.
    settings_dir:
        Directory of ``instruments``/``costs``/``prop_rules`` YAML (repo ``configs/``).
    snapshot_root:
        Root of the content-addressed :class:`SnapshotStore` holding the bundled
        validation-grade snapshot named by ``config.snapshot_hash``.
    out_dir:
        Where the run-scoped report directory is written.
    created_at:
        Deterministic timestamp stamped into trials/manifest (defaults to a fixed
        instant so runs are reproducible; never reads the wall clock implicitly).
    feature_config:
        Feature windows/calendar; defaults to a 1-minute :class:`FeatureConfig`.
    repo:
        Repo path for the git-SHA provenance (defaults to cwd).
    """
    cfg = PipelineConfig.load(config_path)
    settings = Settings.load(settings_dir)
    spec = settings.instruments[cfg.instrument]
    cost_cfg = settings.costs[cfg.instrument]
    rules = settings.prop_rules[cfg.prop_preset]
    resolved_created = created_at if created_at is not None else _DEFAULT_CREATED_AT
    fcfg = feature_config if feature_config is not None else FeatureConfig(interval="1m")
    git_sha = current_git_sha(repo)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    store = SnapshotStore(snapshot_root)
    bars, meta = store.load(cfg.snapshot_hash)
    require_validation_grade(meta)  # G1/G3: refuse dev / non-continuous data

    # 1) features + labels (point-in-time chain exercised end-to-end).
    _features_and_labels(bars, spec, fcfg)

    # 2) triage sweep -- logs exactly one trial per grid combination. This is the
    # SELECTION search whose size is the honest DSR trial count, so it gets its own
    # logger; the confirmation backtests below (re-runs of the single winner) use a
    # separate logger and must not inflate the deflation benchmark.
    logger = TrialLogger(out_path / f"{cfg.run_id}.trials.db")
    signal = SIGNAL_REGISTRY[cfg.signal]()
    # G10 (honest DSR): the trial count MUST be sourced from the SIGNAL's own
    # family, which is exactly the family the sweep logs each trial under (the
    # harness logs ``strategy_family=signal.family``). Deriving it here -- rather
    # than trusting the independently-supplied ``cfg.strategy_family`` -- makes the
    # logged family and ``logger.count(...)`` identical by construction, so a
    # mismatched config field can never silently collapse the count to 0 and fake
    # a statistical NO-GO. ``cfg.strategy_family`` is retained as a display/label
    # field only.
    strategy_family = signal.family
    splitter = PurgedKFold(n_splits=cfg.n_splits, embargo_frac=cfg.embargo_frac)
    sweep_result = run_sweep(
        signal,
        cfg.grid,
        [cfg.snapshot_hash],
        cost_cfg,
        splitter,
        logger,
        cfg.seed,
        store=store,
        spec=spec,
        run_id=cfg.run_id,
        git_sha=git_sha,
        ts=resolved_created,
    )
    best_params = _best_params(sweep_result.best, cfg.grid)

    # 3) event-driven backtest of the selected config: net, gross, and 1-bar-delay.
    # A separate logger keeps these confirmation re-runs out of the DSR trial count.
    bt_logger = TrialLogger(out_path / f"{cfg.run_id}.backtest.trials.db")
    runner = BacktestRunner(store, bt_logger, repo=repo)
    strat = StrategyConfig(signal=cfg.signal, params=dict(best_params), qty=1, seed=cfg.seed)
    net = runner.run(cfg.snapshot_hash, strat, cost_cfg, spec, created_at=resolved_created)
    gross = runner.run(cfg.snapshot_hash, strat, _GROSS_COST, spec, created_at=resolved_created)
    delayed_cost = cost_cfg.model_copy(update={"delay_bars": 1})
    delayed = runner.run(cfg.snapshot_hash, strat, delayed_cost, spec, created_at=resolved_created)

    # 4) validation input: the per-config CSCV performance matrix for PBO.
    perf = _perf_matrix(bars, spec, cost_cfg, signal, _combos(cfg.grid), cfg.pbo_partitions)

    # 5) prop-survival Monte Carlo on the selected config's realised trades (G11).
    survival = _run_survival(net, rules, cfg)

    # 6) the report + strict verdict.
    run_dir = build_report(
        cfg.run_id,
        logger,
        survival,
        net,
        cfg.gates,
        out_dir=out_dir,
        gross=cast("BacktestResultLike", gross),
        delayed=cast("BacktestResultLike", delayed),
        perf_matrix=perf,
        strategy_family=strategy_family,
        cv_scheme=_cv_scheme(splitter),
        cost_cfg=cost_cfg,
        pbo_partitions=cfg.pbo_partitions,
        bootstrap_n=cfg.bootstrap_n,
        bootstrap_seed=cfg.seed,
    )
    verdict = Verdict.model_validate_json((run_dir / "verdict.json").read_text(encoding="utf-8"))
    return PipelineOutcome(
        run_id=cfg.run_id,
        run_dir=run_dir,
        verdict=verdict,
        n_trials=logger.count(strategy_family),
        best_params=best_params,
    )
