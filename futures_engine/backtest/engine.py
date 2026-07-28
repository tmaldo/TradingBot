"""The event-driven validation backtester (T6).

:class:`BacktestRunner` runs a reference signal through the Nautilus event loop
(:mod:`futures_engine.backtest.strategy_adapter`) as the order/position **state
machine**, then prices the resulting round-turns with T2's canonical
:func:`~futures_engine.costs.model.delayed_fill_prices` + ``apply_costs`` so that
every reported number is net of costs via the single T2 authority (G8). It refuses
non-validation-grade / non-continuous data (G3), populates a fully-reproducible
:class:`~futures_engine.core.manifest.RunManifest`, and logs exactly one
:class:`~futures_engine.trials.logger.TrialRecord` per run.

The produced :class:`BacktestResult` satisfies
:class:`~futures_engine.validation.stats.BacktestResultLike`, so T3's ``red_flags``
consumes it directly.

Accounting seam (Path 3)
------------------------
* Nautilus runs with a **zero** ``FeeModel`` and deterministic fills, so its
  realised PnL is pure gross ``(exit - entry) * multiplier * qty`` on the *bar
  close* prices it fills at. :func:`reconcile_gross` cross-checks that against T2's
  ``gross_pnl_usd`` on those same prices (a test-only assertion; never reported).
* The **reported** trade log takes only the round-turn *boundaries* from Nautilus
  and re-prices them at the next-bar open via ``delayed_fill_prices`` -> the exact
  same fills the T5 vectorized path uses, which is what makes the two paths agree
  (see :mod:`futures_engine.backtest.parity`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.backtest.instrument import assert_spec_parity, build_nautilus_instrument
from futures_engine.backtest.strategy_adapter import (
    SIGNAL_REGISTRY,
    NautilusRun,
    bar_timestamps_ns,
    run_event_loop,
)
from futures_engine.core.manifest import RunManifest, current_git_sha
from futures_engine.core.types import Bars, InstrumentSpec
from futures_engine.costs.model import (
    COST_COLUMNS,
    TRADE_COLUMNS,
    CostConfig,
    apply_costs,
    delayed_fill_prices,
)
from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.research.harness import causal_positions
from futures_engine.trials.logger import TrialLogger, TrialRecord

try:  # pragma: no cover - import indirection for a clearer error surface
    from nautilus_trader.model.identifiers import Venue
except ImportError as exc:  # pragma: no cover
    raise ImportError("nautilus_trader is required for the T6 backtest engine") from exc


class StrategyConfig(BaseModel):
    """Config for one backtest run: which signal, its params, and sizing (G15).

    A run names its signal by a stable registry key (``signal``) rather than
    smuggling in an object, so the config fully determines the run and hashes
    reproducibly. ``qty`` is the contract size a unit target position maps to;
    ``seed`` participates in the manifest's config hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: str = Field(min_length=1)
    params: dict[str, Any]
    qty: int = Field(default=1, ge=1)
    seed: int = 0
    starting_balance: float = Field(default=100_000.0, gt=0)


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a single :meth:`BacktestRunner.run`.

    Satisfies :class:`~futures_engine.validation.stats.BacktestResultLike`
    (``returns`` / ``equity`` / ``metrics`` with at least ``sharpe`` and
    ``win_rate``). ``returns`` is a **per-trade** net-return series (net USD PnL of
    each round-turn as a fraction of the starting balance); ``equity`` is the
    positive-valued cumulative equity curve after each trade.
    """

    trades: pd.DataFrame  # TRADE_COLUMNS + COST_COLUMNS (priced, net of costs)
    equity: pd.Series
    returns: pd.Series
    fills: pd.DataFrame
    metrics: dict[str, float]
    manifest: RunManifest


# --- trade-log assembly ------------------------------------------------------


def _ns_to_pos(bars: Bars) -> dict[int, int]:
    return {int(ns): i for i, ns in enumerate(bar_timestamps_ns(bars))}


def build_trade_log(run: NautilusRun, bars: Bars) -> pd.DataFrame:
    """Turn captured Nautilus positions into a raw round-turn frame.

    Columns: ``entry_ts``, ``exit_ts`` (canonical ``bars.index`` timestamps),
    ``side``, ``qty``. Closed positions come straight from the event stream; a
    still-open final position is closed **mark-to-open at the last bar** -- the
    identical convention :func:`research.harness.positions_to_trades` uses -- and a
    degenerate final run (opened on the very last bar) is dropped.
    """
    ns_to_pos = _ns_to_pos(bars)
    index = bars.index
    n = len(index)
    entry_ts: list[pd.Timestamp] = []
    exit_ts: list[pd.Timestamp] = []
    sides: list[str] = []
    qtys: list[float] = []

    for trade in run.closed:
        entry_ts.append(index[ns_to_pos[trade.entry_ts_ns]])
        exit_ts.append(index[ns_to_pos[trade.exit_ts_ns]])
        sides.append(trade.side)
        qtys.append(trade.qty)

    if run.final_open is not None:
        entry_ns, side, qty = run.final_open
        entry_pos = ns_to_pos[int(entry_ns)]
        if entry_pos != n - 1:  # drop a zero-length final run (mark-to-open)
            entry_ts.append(index[entry_pos])
            exit_ts.append(index[n - 1])
            sides.append(side)
            qtys.append(qty)

    frame = pd.DataFrame({"entry_ts": entry_ts, "exit_ts": exit_ts, "side": sides, "qty": qtys})
    return frame.sort_values("entry_ts").reset_index(drop=True)


def reconcile_gross(run: NautilusRun, spec: InstrumentSpec) -> float:
    """Max abs difference between Nautilus realised PnL and T2's gross formula.

    With ``fee_model=0`` Nautilus reports gross PnL; on the *same* fill prices and
    multiplier it must equal ``sign * (exit - entry) * multiplier * qty`` to
    floating-point tolerance. Returns the worst-case delta (0.0 when no trades).
    """
    worst = 0.0
    for trade in run.closed:
        sign = 1.0 if trade.side == "long" else -1.0
        formula = sign * (trade.exit_px_close - trade.entry_px_close) * spec.multiplier * trade.qty
        worst = max(worst, abs(trade.nautilus_pnl - formula))
    return worst


# --- fill model (T2 authority) ----------------------------------------------


def _empty_priced() -> pd.DataFrame:
    return pd.DataFrame({col: [] for col in (*TRADE_COLUMNS, *COST_COLUMNS)})


def price_trades(
    raw_trades: pd.DataFrame,
    bars: Bars,
    spec: InstrumentSpec,
    cost_cfg: CostConfig,
) -> pd.DataFrame:
    """Price raw round-turns at the T2 delayed fill, then net of costs (the fill model).

    Fills happen at bar ``t + cost_cfg.delay_bars``'s open via T2's
    :func:`~futures_engine.costs.model.delayed_fill_prices` (``delay_bars=1`` is
    the binding next-bar-open convention), then :func:`apply_costs` subtracts every
    friction. Trades whose delayed fill would run off the end of the data are
    dropped (documented residual, mirroring the T5 harness's ``room`` filter), so a
    tail trade never raises.
    """
    if raw_trades.empty:
        return _empty_priced()
    delay = cost_cfg.delay_bars
    pos_of = _ns_to_pos(bars)
    ns_index = {ts: int(ns) for ts, ns in zip(bars.index, bar_timestamps_ns(bars), strict=True)}
    n = len(bars.index)
    entry_pos = np.array([pos_of[ns_index[ts]] for ts in raw_trades["entry_ts"]])
    exit_pos = np.array([pos_of[ns_index[ts]] for ts in raw_trades["exit_ts"]])
    room = (entry_pos + delay < n) & (exit_pos + delay < n)
    filtered = raw_trades.loc[room].reset_index(drop=True)
    if filtered.empty:
        return _empty_priced()
    delayed = delayed_fill_prices(filtered, bars, spec, delay)
    return apply_costs(delayed, spec, cost_cfg)


# --- metrics -----------------------------------------------------------------


def _sharpe(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0.0 else 0.0


def _max_drawdown_fraction(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0.0, (running_max - equity) / running_max, 0.0)
    return float(drawdown.max())


def _metrics_and_curves(
    priced: pd.DataFrame, starting_balance: float
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    if priced.empty:
        empty = pd.Series([], dtype=float)
        metrics = {
            "sharpe": 0.0,
            "sharpe_gross": 0.0,
            "win_rate": 0.0,
            "n_trades": 0.0,
            "max_dd": 0.0,
        }
        return metrics, empty, empty

    net = priced["net_pnl_usd"].to_numpy(dtype=float)
    gross = priced["gross_pnl_usd"].to_numpy(dtype=float)
    net_returns = net / starting_balance
    equity_values = starting_balance + np.cumsum(net)
    exit_ts = priced["exit_ts"].to_numpy()
    returns = pd.Series(net_returns, index=exit_ts, name="returns")
    equity = pd.Series(equity_values, index=exit_ts, name="equity")
    metrics = {
        "sharpe": _sharpe(net_returns),
        "sharpe_gross": _sharpe(gross / starting_balance),
        "win_rate": float((net > 0.0).mean()),
        "n_trades": float(len(priced)),
        "max_dd": _max_drawdown_fraction(equity_values),
    }
    return metrics, equity, returns


# --- config hash -------------------------------------------------------------


def config_hash(strategy_cfg: StrategyConfig, cost_cfg: CostConfig) -> str:
    """Canonical 64-hex content hash of the run's economics + strategy + seed."""
    payload = {
        "strategy": strategy_cfg.model_dump(mode="json"),
        "cost_cfg": cost_cfg.model_dump(mode="json"),
        "seed": strategy_cfg.seed,
    }
    blob = b"FE-BACKTESTCFG-v1\n" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --- runner ------------------------------------------------------------------


class BacktestRunner:
    """Run a reference signal event-driven and report a costed, reproducible result."""

    def __init__(
        self,
        store: SnapshotStore,
        logger: TrialLogger,
        *,
        venue: str = "GLBX",
        repo: str | Path | None = None,
    ) -> None:
        self._store = store
        self._logger = logger
        self._venue = Venue(venue)
        self._repo = repo

    def _held_targets(
        self, bars: Bars, strategy_cfg: StrategyConfig
    ) -> tuple[pd.Series, dict[int, int]]:
        if strategy_cfg.signal not in SIGNAL_REGISTRY:
            raise KeyError(
                f"unknown signal {strategy_cfg.signal!r}; known: {sorted(SIGNAL_REGISTRY)}"
            )
        signal = SIGNAL_REGISTRY[strategy_cfg.signal]()
        raw = signal.generate(bars, strategy_cfg.params)
        held = causal_positions(raw)
        # Whole-contract integer targets (Nautilus trades integer contracts). A
        # fractional target position is rounded to the nearest whole contract -- a
        # legitimate residual vs the fractional-qty vectorized path, documented in
        # parity.py; the reference parity signal is integer-valued so it is nil.
        targets_int = np.rint(held.to_numpy(dtype=float) * strategy_cfg.qty).astype("int64")
        ts_ns = bar_timestamps_ns(bars)
        targets = {int(ns): int(t) for ns, t in zip(ts_ns, targets_int, strict=True)}
        return held, targets

    def run(
        self,
        snapshot_hash: str,
        strategy_cfg: StrategyConfig,
        cost_cfg: CostConfig,
        spec: InstrumentSpec,
        *,
        created_at: datetime | None = None,
    ) -> BacktestResult:
        """Execute the backtest and return a costed :class:`BacktestResult`.

        Refuses non-validation-grade / non-continuous snapshots (G3), builds the
        Nautilus instrument from ``spec`` and re-asserts spec parity, runs the event
        loop, prices trades net of ``cost_cfg`` via T2, logs one TrialRecord, and
        attaches a fully-populated manifest.
        """
        bars, meta = self._store.load(snapshot_hash)
        require_validation_grade(meta)  # G3: loud failure on dev / non-continuous data

        instrument = build_nautilus_instrument(spec, self._venue)
        assert_spec_parity(instrument, spec)  # runtime drift guard before any run

        _held, targets = self._held_targets(bars, strategy_cfg)
        run = run_event_loop(
            bars,
            targets,
            instrument,
            self._venue,
            starting_balance=strategy_cfg.starting_balance,
        )
        raw_trades = build_trade_log(run, bars)
        priced = price_trades(raw_trades, bars, spec, cost_cfg)
        metrics, equity, returns = _metrics_and_curves(priced, strategy_cfg.starting_balance)

        cfg_hash = config_hash(strategy_cfg, cost_cfg)
        resolved_created = created_at if created_at is not None else datetime.now(UTC)
        git_sha = current_git_sha(self._repo)
        signal_family = SIGNAL_REGISTRY[strategy_cfg.signal]().family
        trial_id = f"{cfg_hash[:16]}-{snapshot_hash[:8]}"

        self._logger.log(
            TrialRecord(
                trial_id=trial_id,
                run_id=cfg_hash[:16],
                ts=resolved_created,
                strategy_family=signal_family,
                config_hash=cfg_hash,
                params=strategy_cfg.params,
                data_snapshot_hashes=[snapshot_hash],
                cv_scheme="event_backtest",
                metrics=metrics,
                seed=strategy_cfg.seed,
                git_sha=git_sha,
            )
        )
        manifest = RunManifest.create(
            created_at=resolved_created,
            config_hash=cfg_hash,
            data_snapshot_hashes=[snapshot_hash],
            seed=strategy_cfg.seed,
            trial_ids=[trial_id],
            run_id=cfg_hash[:16],
            git_sha=git_sha,
        )
        return BacktestResult(
            trades=priced,
            equity=equity,
            returns=returns,
            fills=priced[list(TRADE_COLUMNS)].copy(),
            metrics=metrics,
            manifest=manifest,
        )
