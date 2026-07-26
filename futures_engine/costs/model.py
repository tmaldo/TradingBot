"""Transaction-cost model: every reported result flows through this (G8/G16).

A backtest produces a :data:`TradeLog` of gross round-turn trades. :func:`apply_costs`
decomposes the frictions of executing each round-turn -- broker commission, exchange
and NFA fees, the bid/ask spread, and slippage -- into explicit USD columns and reports
``net_pnl_usd = gross_pnl_usd - Σ costs``. Gross is always retained alongside net so a
reviewer can see exactly what costs did to an edge (Global Constraint G8).

Cost arithmetic (per round-turn trade row; ``qty`` = number of contracts):

* **commission / fees** are charged on every *fill*, so they apply to both the entry
  and the exit -- two sides::

      commission_usd = commission_per_side_usd * 2 * qty
      fees_usd       = (exchange_fee_per_side_usd + nfa_fee_per_side_usd) * 2 * qty

* **spread / slippage** are price-impact costs expressed in *ticks per round-turn* and
  converted to USD via the instrument's ``tick_value``. ``spread_ticks`` is the full
  bid/ask spread, paid once over a round-turn (half on entry, half on exit)::

      spread_cost_usd = spread_ticks * tick_value * qty

  Slippage has two models (``CostConfig.slippage``):

  * ``fixed_ticks``  -> ``slippage_ticks * tick_value * qty``
  * ``vol_scaled``   -> ``slippage_ticks * atr_ticks_at_entry * tick_value * qty``

    Here ``slippage_ticks`` is a dimensionless coefficient and each trade row must carry
    an ``atr_ticks_at_entry`` column (per-trade volatility at entry, in ticks). A missing
    column with ``vol_scaled`` selected is a loud error, never a silent zero.

``delay_bars`` execution semantics are defined by :func:`delayed_fill_prices` (a signal
on bar ``t`` fills at bar ``t+delay`` open); the research/backtest layers (T5/T6) reuse
that helper so the "1-bar delay" edge-decay check is computed one way everywhere.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from futures_engine.core.types import InstrumentSpec

# Column contract for a TradeLog frame. ``apply_costs`` adds cost columns to a frame
# carrying at least these; ``gross_pnl_usd`` is the pre-cost round-turn P&L in USD.
TradeLog = pd.DataFrame
TRADE_COLUMNS: tuple[str, ...] = (
    "entry_ts",
    "exit_ts",
    "side",
    "qty",
    "entry_px",
    "exit_px",
    "gross_pnl_usd",
)

# Cost columns added by ``apply_costs``.
COST_COLUMNS: tuple[str, ...] = (
    "commission_usd",
    "fees_usd",
    "spread_cost_usd",
    "slippage_usd",
    "net_pnl_usd",
)

_SIDES_PER_ROUND_TURN = 2


class CostConfig(BaseModel):
    """Per-instrument execution-cost parameters (all values live in ``costs.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commission_per_side_usd: float = Field(ge=0)
    exchange_fee_per_side_usd: float = Field(ge=0)
    nfa_fee_per_side_usd: float = Field(ge=0)
    spread_ticks: float = Field(ge=0)
    slippage: Literal["fixed_ticks", "vol_scaled"]
    slippage_ticks: float = Field(ge=0)
    delay_bars: int = Field(ge=0, le=1)  # in {0, 1}


def _require_columns(trades: TradeLog, required: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"TradeLog missing required column(s): {', '.join(missing)}")


def apply_costs(trades: TradeLog, spec: InstrumentSpec, cfg: CostConfig) -> TradeLog:
    """Return a copy of ``trades`` with per-trade cost columns and ``net_pnl_usd`` added.

    Adds ``commission_usd``, ``fees_usd``, ``spread_cost_usd``, ``slippage_usd`` and
    ``net_pnl_usd``. The input frame is not mutated.
    """
    _require_columns(trades, TRADE_COLUMNS)

    out = trades.copy()
    qty = out["qty"].astype(float)

    out["commission_usd"] = cfg.commission_per_side_usd * _SIDES_PER_ROUND_TURN * qty
    out["fees_usd"] = (
        (cfg.exchange_fee_per_side_usd + cfg.nfa_fee_per_side_usd) * _SIDES_PER_ROUND_TURN * qty
    )
    out["spread_cost_usd"] = cfg.spread_ticks * spec.tick_value * qty

    if cfg.slippage == "fixed_ticks":
        slip_ticks = pd.Series(cfg.slippage_ticks, index=out.index)
    else:  # vol_scaled
        if "atr_ticks_at_entry" not in out.columns:
            raise ValueError(
                "slippage='vol_scaled' requires an 'atr_ticks_at_entry' column on the TradeLog"
            )
        slip_ticks = cfg.slippage_ticks * out["atr_ticks_at_entry"].astype(float)
    out["slippage_usd"] = slip_ticks * spec.tick_value * qty

    out["net_pnl_usd"] = (
        out["gross_pnl_usd"].astype(float)
        - out["commission_usd"]
        - out["fees_usd"]
        - out["spread_cost_usd"]
        - out["slippage_usd"]
    )
    return out


def delayed_fill_prices(
    trades: TradeLog,
    bars: pd.DataFrame,
    spec: InstrumentSpec,
    delay_bars: int,
) -> TradeLog:
    """Shift executions by ``delay_bars`` and recompute gross P&L from the fill prices.

    Semantic (reused verbatim by T5/T6): a signal observed on bar ``t`` is *executed at
    the open of bar ``t + delay_bars``*. ``delay_bars=0`` fills at the signal bar's own
    open; ``delay_bars=1`` fills at the next bar's open. ``entry_ts``/``exit_ts`` in the
    input name the *signal* bars (must exist in ``bars.index``); the returned copy carries
    the fill-bar timestamps, the fill-bar opens as ``entry_px``/``exit_px``, and a
    ``gross_pnl_usd`` recomputed as ``(exit-entry) * multiplier * qty`` for longs (sign
    flipped for shorts).

    ``bars`` must be indexed by a unique, sorted timestamp index with an ``open`` column.
    """
    if delay_bars < 0:
        raise ValueError(f"delay_bars must be >= 0, got {delay_bars}")
    _require_columns(trades, ("entry_ts", "exit_ts", "side", "qty"))

    out = trades.copy()
    n = len(bars.index)
    pos_of: dict[object, int] = {ts: i for i, ts in enumerate(bars.index)}
    opens = bars["open"].to_numpy(dtype=float)

    def fill_positions(col: str) -> np.ndarray:
        result = np.empty(len(out), dtype=np.int64)
        for i, ts in enumerate(out[col]):
            if ts not in pos_of:
                raise ValueError(f"signal timestamp not found in bars index: {ts!r}")
            filled = pos_of[ts] + delay_bars
            if filled >= n:
                raise ValueError(
                    f"delayed fill bar (position {filled}) is out of range for {n} bars"
                )
            result[i] = filled
        return result

    e_pos = fill_positions("entry_ts")
    x_pos = fill_positions("exit_ts")
    e_px = opens[e_pos]
    x_px = opens[x_pos]
    qty = out["qty"].to_numpy(dtype=float)

    sides = out["side"].astype(str).str.lower()
    valid = sides.isin(["long", "short"])
    if not bool(valid.all()):
        bad = sorted(set(sides[~valid]))
        raise ValueError(f"unknown side(s) {bad}; expected 'long' or 'short'")
    sign = np.where(sides.to_numpy() == "long", 1.0, -1.0)

    out["entry_ts"] = bars.index[e_pos]
    out["exit_ts"] = bars.index[x_pos]
    out["entry_px"] = e_px
    out["exit_px"] = x_px
    out["gross_pnl_usd"] = sign * (x_px - e_px) * spec.multiplier * qty
    return out
