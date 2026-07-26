"""Vectorized research harness: signal sweeps with honest trial logging (G10).

This module is the signal-triage stage. It turns a parameterised vector signal
into round-turn trades under the project's **binding delay-fill causality
convention**, prices them through the T2 cost model, and evaluates a whole grid
of parameter combinations against a T3 splitter -- logging *exactly one*
:class:`~futures_engine.trials.logger.TrialRecord` per evaluated combination so
the Deflated Sharpe Ratio's trial count is honest.

Causality convention (from the T2 review, implemented here)
-----------------------------------------------------------
``entry_ts`` / ``exit_ts`` name the bar **at whose OPEN the order acts**. A
:class:`VectorSignal` may compute a raw target position using data through bar
``t``'s close; the harness makes it causal by holding that decision only from
bar ``t + 1`` onward (:func:`causal_positions` is a one-bar shift). The resulting
*held* position series -- ``positions[t]`` is the position held during bar ``t``
-- is converted to trades by :func:`positions_to_trades`, whose fills feed
:func:`~futures_engine.costs.model.delayed_fill_prices` (delay 0 = fill at the
signal bar's own open; delay 1 = next bar's open).

Config-hash recipe (``config_hash``)
------------------------------------
Mirroring the style of ``data/store.py``'s snapshot hash: a version tag followed
by canonical JSON (sorted keys, no whitespace) of the essential trial inputs::

    payload = b"FE-TRIALCFG-v1\\n" + canonical_json({params, cost_cfg, seed})
    config_hash = sha256(payload).hexdigest()          # 64 lowercase hex chars

It is a pure function of the parameter combination, the cost configuration, and
the seed -- deterministic across sessions and sensitive to any of those inputs.

OOS evaluation
--------------
For a pure signal sweep there is nothing to *fit*, so each bar's label interval
is its own timestamp span (``t0 == t1 == bar``). The injected splitter still
governs which observations count: metrics are computed over the union of the
splitter's test folds (the out-of-sample mask), never a plain full-sample pass.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd

from futures_engine.core.manifest import current_git_sha
from futures_engine.core.types import Bars, InstrumentSpec
from futures_engine.costs.model import CostConfig, apply_costs
from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.trials.logger import TrialLogger, TrialRecord

_METRIC_KEYS: tuple[str, ...] = (
    "sharpe_gross",
    "sharpe_net",
    "sharpe_net_delay1",
    "n_trades",
    "win_rate",
    "max_dd",
    "error",
)


@runtime_checkable
class VectorSignal(Protocol):
    """A vectorized signal family: a stable ``family`` label and ``generate``."""

    family: str

    def generate(self, bars: Bars, params: dict[str, Any]) -> pd.Series:
        """Return a RAW float target position in ``[-1, 1]`` indexed like ``bars``."""
        ...


class Splitter(Protocol):
    """Structural view of a T3 splitter (``PurgedKFold`` / ``WalkForward`` / CPCV)."""

    def split(
        self, x: object, intervals: pd.Series, /
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True)
class SweepResult:
    """Outcome of a parameter sweep.

    ``table`` has one row per evaluated combination (its params, metrics, the
    ``error`` flag, ``config_hash`` and ``trial_id``); ``best`` is the row that
    maximises ``sharpe_net`` among non-errored combos (empty if none); and
    ``n_trials_logged`` is the number of TrialRecords written -- which equals the
    grid's cross-product size, errors included.
    """

    table: pd.DataFrame
    best: dict[str, Any]
    n_trials_logged: int


# --- position -> trades ------------------------------------------------------


def _empty_tradelog() -> pd.DataFrame:
    """An empty TradeLog carrying the canonical :data:`TRADE_COLUMNS` in order."""
    from futures_engine.costs.model import TRADE_COLUMNS

    return pd.DataFrame({col: [] for col in TRADE_COLUMNS})


def causal_positions(raw: pd.Series) -> pd.Series:
    """Make a raw signal causal: hold bar ``t``'s decision from bar ``t+1`` on.

    A one-bar shift (with a flat leading bar) is the entire causality transform;
    the buggy legacy path is simply *omitting* this shift (acting on same-bar
    information), which :mod:`tests.research.test_causality` shows is measurably
    -- and dishonestly -- better.
    """
    return raw.shift(1).fillna(0.0)


def positions_to_trades(
    positions: pd.Series,
    bars: Bars,
    spec: InstrumentSpec,
    qty: float,
) -> pd.DataFrame:
    """Convert a *held* target-position series into a T2 :data:`TradeLog`.

    ``positions`` is aligned to ``bars.index`` and interpreted causally:
    ``positions[t]`` is the (float, in ``[-1, 1]``) position **held during bar
    ``t``**, i.e. entered at bar ``t``'s open. A trade is one maximal run of a
    constant, non-zero position value:

    * ``side`` = ``"long"`` if the value is positive else ``"short"``;
    * contract ``qty`` = ``|value| * qty`` (a float position scales the size);
    * ``entry_ts`` = the run's first bar; ``exit_ts`` = the bar at whose open the
      position next changes (a run reaching the final bar is closed at that bar's
      open -- mark-to-open).

    A **flip** (e.g. ``+1`` then ``-1``) is therefore *two* trades that share the
    crossover bar: the old position exits and the new one enters at that one open.
    ``gross_pnl_usd`` is ``sign * (exit_open - entry_open) * multiplier * qty`` --
    identical to :func:`~futures_engine.costs.model.delayed_fill_prices` at
    ``delay_bars=0``, so the two agree by construction. Fully vectorised (run
    boundaries via ``numpy.diff``); no per-bar Python loop.
    """
    if not positions.index.equals(bars.index):
        raise ValueError("positions index must equal bars index")
    n = len(bars)
    if n == 0:
        return _empty_tradelog()

    values = positions.to_numpy(dtype=float)
    opens = bars["open"].to_numpy(dtype=float)
    index = bars.index

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))  # exclusive run end == exit fill bar

    seg_values = values[starts]
    is_trade = seg_values != 0.0
    entry_pos = starts[is_trade]
    exit_pos = np.minimum(ends[is_trade], n - 1)  # final run closes at last open
    seg_values = seg_values[is_trade]

    keep = entry_pos != exit_pos  # drop degenerate zero-length final run
    entry_pos, exit_pos, seg_values = entry_pos[keep], exit_pos[keep], seg_values[keep]

    contracts = np.abs(seg_values) * float(qty)
    sign = np.where(seg_values > 0.0, 1.0, -1.0)
    entry_px = opens[entry_pos]
    exit_px = opens[exit_pos]
    gross = sign * (exit_px - entry_px) * spec.multiplier * contracts

    return pd.DataFrame(
        {
            "entry_ts": index[entry_pos],
            "exit_ts": index[exit_pos],
            "side": np.where(seg_values > 0.0, "long", "short"),
            "qty": contracts,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "gross_pnl_usd": gross,
        }
    )


# --- metrics -----------------------------------------------------------------


def _sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe (mean / std, ddof=1); 0.0 when undefined."""
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    return float(returns.mean() / std) if std > 0.0 else 0.0


def _max_drawdown_usd(net: np.ndarray) -> float:
    """Max peak-to-trough decline of the cumulative net USD PnL (>= 0)."""
    if net.size == 0:
        return 0.0
    equity = np.cumsum(net)
    running_max = np.maximum.accumulate(equity)
    return float((running_max - equity).max())


def _bar_positions(index: pd.Index, timestamps: pd.Series) -> np.ndarray:
    """Vectorised, hashtable-cached position lookup of trade timestamps in bars."""
    return index.get_indexer(pd.Index(timestamps))


def _delayed_fill_prices_vec(
    trades: pd.DataFrame,
    index: pd.Index,
    opens: np.ndarray,
    spec: InstrumentSpec,
    delay: int,
) -> pd.DataFrame:
    """Vectorised twin of :func:`~futures_engine.costs.model.delayed_fill_prices`.

    Identical delay semantic -- a signal on bar ``t`` fills at bar ``t + delay``'s
    open, timestamps/prices/gross recomputed from the fill bar -- but O(trades)
    per call against a *precomputed* index + opens array, instead of T2's helper
    which rebuilds an O(bars) Python dict on every call (289 ms at 350k bars, a
    per-row loop that would blow the 500-combo/5y runtime budget). Byte-for-byte
    equivalence with the T2 helper is pinned by ``test_delay_semantic_matches_t2``
    so the "one delay semantic everywhere" guarantee holds. Callers must ensure
    every fill stays in range (the sweep filters trades that lack ``delay`` room).
    """
    entry_pos = _bar_positions(index, trades["entry_ts"]) + delay
    exit_pos = _bar_positions(index, trades["exit_ts"]) + delay
    entry_px = opens[entry_pos]
    exit_px = opens[exit_pos]
    sign = np.where(trades["side"].astype(str).str.lower().to_numpy() == "long", 1.0, -1.0)
    qty = trades["qty"].to_numpy(dtype=float)
    out = trades.copy()
    out["entry_ts"] = index[entry_pos]
    out["exit_ts"] = index[exit_pos]
    out["entry_px"] = entry_px
    out["exit_px"] = exit_px
    out["gross_pnl_usd"] = sign * (exit_px - entry_px) * spec.multiplier * qty
    return out


def _evaluate(
    trades: pd.DataFrame,
    bars: Bars,
    spec: InstrumentSpec,
    cost_cfg: CostConfig,
) -> dict[str, float]:
    """Metrics for one combo's OOS trades: gross, net, 1-bar-delay net, and stats.

    ``delay_bars=0`` fills are exactly the prices :func:`positions_to_trades`
    produced (fill at the signal bar's own open), so the delay-0 leg needs no
    re-fill; the delay-1 leg is recomputed vectorised over the fill-in-range
    subset.
    """
    if len(trades) == 0:
        return {k: 0.0 for k in _METRIC_KEYS}

    index = bars.index
    opens = bars["open"].to_numpy(dtype=float)
    n = opens.shape[0]

    priced = apply_costs(trades, spec, cost_cfg)  # delay-0 fills == input trades
    gross = priced["gross_pnl_usd"].to_numpy(dtype=float)
    net = priced["net_pnl_usd"].to_numpy(dtype=float)

    # 1-bar-delay variant: drop trades whose delayed fill would run off the data.
    entry_pos = _bar_positions(index, trades["entry_ts"])
    exit_pos = _bar_positions(index, trades["exit_ts"])
    room = np.flatnonzero((entry_pos + 1 < n) & (exit_pos + 1 < n))
    if room.size > 0:
        delayed = apply_costs(
            _delayed_fill_prices_vec(trades.iloc[room], index, opens, spec, 1), spec, cost_cfg
        )
        sharpe_net_delay1 = _sharpe(delayed["net_pnl_usd"].to_numpy(dtype=float))
    else:
        sharpe_net_delay1 = 0.0

    return {
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "sharpe_net_delay1": sharpe_net_delay1,
        "n_trades": float(len(trades)),
        "win_rate": float((net > 0.0).mean()),
        "max_dd": _max_drawdown_usd(net),
        "error": 0.0,
    }


# --- config hash + sweep -----------------------------------------------------


def config_hash(params: dict[str, Any], cost_cfg: CostConfig, seed: int) -> str:
    """Canonical 64-hex content hash of a trial's params + cost config + seed."""
    payload = {
        "params": params,
        "cost_cfg": cost_cfg.model_dump(mode="json"),
        "seed": seed,
    }
    blob = b"FE-TRIALCFG-v1\n" + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def _cv_scheme(splitter: Splitter) -> str:
    """A stable string naming the splitter and its scalar hyper-parameters."""
    scalars = {
        k: v
        for k, v in vars(splitter).items()
        if not k.startswith("_") and isinstance(v, (int, float, bool))
    }
    inner = ",".join(f"{k}={v}" for k, v in sorted(scalars.items()))
    return f"{type(splitter).__name__}({inner})"


def _load_bars(store: SnapshotStore, snapshot_hashes: list[str]) -> Bars:
    """Load + concatenate snapshots, refusing any non-validation-grade data (G1)."""
    frames = []
    for snapshot_hash in snapshot_hashes:
        bars, meta = store.load(snapshot_hash)
        require_validation_grade(meta)  # raises DataIntegrityError on dev-grade data
        frames.append(bars)
    combined = pd.concat(frames).sort_index()
    if not combined.index.is_unique:
        from futures_engine.data.store import DataIntegrityError

        raise DataIntegrityError("combined snapshot bars have a non-unique index")
    return combined


def _oos_mask(bars: Bars, splitter: Splitter) -> np.ndarray:
    """Boolean mask over bar positions = union of the splitter's test folds."""
    intervals = pd.Series(bars.index, index=bars.index)
    mask = np.zeros(len(bars), dtype=bool)
    for _train_idx, test_idx in splitter.split(bars, intervals):
        mask[test_idx] = True
    return mask


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid axes -> one param dict per combination."""
    keys = list(grid.keys())
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def sweep(
    signal: VectorSignal,
    grid: dict[str, list[Any]],
    snapshot_hashes: list[str],
    cost_cfg: CostConfig,
    splitter: Splitter,
    logger: TrialLogger,
    seed: int,
    *,
    store: SnapshotStore,
    spec: InstrumentSpec,
    qty: float = 1.0,
    run_id: str | None = None,
    git_sha: str | None = None,
    ts: datetime | None = None,
) -> SweepResult:
    """Evaluate every ``grid`` combination of ``signal`` and log one trial each.

    Loads (and refuses non-validation-grade) snapshots via ``store``, builds the
    causal held-position series per combo, prices trades net of ``cost_cfg`` with
    gross and 1-bar-delay variants alongside, evaluates over the splitter's OOS
    mask, and logs exactly one :class:`TrialRecord` per combination -- combos that
    raise are logged with ``metrics={"error": 1.0}`` and surfaced in the table.

    ``store`` and ``spec`` are keyword-only: the contract's positional parameters
    identify the *data* (snapshot hashes) and *economics* (cost config) of a
    sweep, while ``store``/``spec`` are the injected resources that resolve hashes
    to bars and price PnL. ``run_id`` / ``git_sha`` / ``ts`` default to a fresh
    UUID, the repo HEAD SHA, and the current UTC time; tests inject them.
    """
    bars = _load_bars(store, snapshot_hashes)
    oos = _oos_mask(bars, splitter)
    cv_scheme = _cv_scheme(splitter)
    combos = _expand_grid(grid)

    resolved_run_id = run_id if run_id is not None else uuid.uuid4().hex
    resolved_git_sha = git_sha if git_sha is not None else current_git_sha()
    resolved_ts = ts if ts is not None else datetime.now(UTC)

    rows: list[dict[str, Any]] = []
    n_logged = 0
    for i, params in enumerate(combos):
        combo_hash = config_hash(params, cost_cfg, seed)
        try:
            raw = signal.generate(bars, params)
            held = causal_positions(raw)
            trades = positions_to_trades(held, bars, spec, qty)
            entry_pos = _bar_positions(bars.index, trades["entry_ts"])
            oos_trades = trades.iloc[np.flatnonzero(oos[entry_pos])]
            metrics = _evaluate(oos_trades, bars, spec, cost_cfg)
        except Exception:
            metrics = {"error": 1.0}

        logger.log(
            TrialRecord(
                trial_id=f"{resolved_run_id}-{i:06d}",
                run_id=resolved_run_id,
                ts=resolved_ts,
                strategy_family=signal.family,
                config_hash=combo_hash,
                params=params,
                data_snapshot_hashes=snapshot_hashes,
                cv_scheme=cv_scheme,
                metrics=metrics,
                seed=seed,
                git_sha=resolved_git_sha,
            )
        )
        n_logged += 1
        rows.append({**params, **metrics, "config_hash": combo_hash})

    table = pd.DataFrame(rows)
    if "error" not in table.columns:
        table["error"] = 0.0
    ok = table[table["error"] == 0.0]
    best: dict[str, Any] = (
        cast("dict[str, Any]", ok.loc[ok["sharpe_net"].idxmax()].to_dict()) if not ok.empty else {}
    )
    return SweepResult(table=table, best=best, n_trials_logged=n_logged)
