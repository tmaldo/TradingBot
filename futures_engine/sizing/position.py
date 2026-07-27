"""Volatility-targeted, survival-dominated position sizing (task T7, G12).

The sizer answers "how many contracts?" and is deliberately conservative: it
takes the **minimum** of four independent legs so that whichever constraint is
tightest wins.

1. **Volatility target.** ``vol_estimate`` is the *per-contract* daily P&L
   standard deviation in USD. The number of contracts whose combined daily P&L
   vol matches the desired ``target_daily_vol_usd`` budget is simply
   ``target_daily_vol_usd / vol_estimate`` (both sides in USD, so the tick
   economics already live inside ``vol_estimate`` -- no extra tick scaling),
   floored to a whole contract.

2. **Fractional Kelly (capped at quarter-Kelly, G12).** From :class:`EdgeStats`
   the full-Kelly fraction of a Bernoulli win/loss bet is

       f* = p_win - (1 - p_win) * avg_loss / avg_win

   (with ``avg_loss`` a positive magnitude). ``f*`` is floored at 0 -- a
   non-positive edge sizes to nothing. The **capital / risk basis** is stated
   explicitly and uses only the inputs available to this function: the daily
   risk *budget* ``target_daily_vol_usd`` stands in for the capital put at risk
   per day, and ``avg_loss`` is the USD risk of one contract on a losing trade.
   Kelly contracts are therefore

       floor( kelly_fraction_cap * f* * target_daily_vol_usd / avg_loss ).

   ``kelly_fraction_cap`` is validated at construction to be in ``(0, 0.25]`` --
   a request for more than quarter-Kelly is rejected loudly (G12).

3. **Survival cap.** ``survival_max_contracts`` is the largest size whose
   :class:`~futures_engine.prop.survival.SurvivalReport` still passes the
   survival gate (computed by the caller via
   :func:`~futures_engine.prop.survival.max_contracts_surviving`). Because it is
   the ``min``, the drawdown/survival rule dominates Kelly whenever the two
   disagree -- the whole point of G12.

4. **Hard ceiling.** ``max_contracts`` is an absolute cap (margin / venue
   limit).

The result is ``max(0, min(all four legs))`` as a plain ``int``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from futures_engine.core.types import InstrumentSpec

# Quarter-Kelly ceiling (G12): no configuration may size above this fraction of
# the full Kelly bet. Lives here as the single source of truth for the cap.
QUARTER_KELLY: float = 0.25


class EdgeStats(BaseModel):
    """A strategy's per-trade edge as a Bernoulli win/loss bet.

    ``avg_win`` and ``avg_loss`` are **positive USD magnitudes** (the average
    winning trade's profit and the average losing trade's loss, per contract).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_win: float = Field(ge=0.0, le=1.0)
    avg_win: float = Field(gt=0.0)
    avg_loss: float = Field(gt=0.0)

    @property
    def kelly_fraction(self) -> float:
        """Full-Kelly fraction ``f*``, floored at 0 (no negative sizing)."""
        payoff = self.avg_win / self.avg_loss
        f_star = self.p_win - (1.0 - self.p_win) / payoff
        return max(0.0, f_star)


class SizingConfig(BaseModel):
    """Sizing knobs. All values are config-driven (G15); the quarter-Kelly cap
    (G12) is enforced here at construction so no caller can exceed it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_daily_vol_usd: float = Field(gt=0.0)
    # (0, 0.25]: strictly positive, at most quarter-Kelly.
    kelly_fraction_cap: float = Field(gt=0.0, le=QUARTER_KELLY)
    max_contracts: int = Field(ge=0)


def position_size(
    vol_estimate: float,
    edge: EdgeStats,
    spec: InstrumentSpec,
    cfg: SizingConfig,
    survival_max_contracts: int,
) -> int:
    """Return the sized contract count: ``min`` of the four legs, floored at 0.

    ``vol_estimate`` is the per-contract daily P&L standard deviation in USD and
    must be strictly positive. ``spec`` is accepted for interface symmetry and
    future tick-aware extensions; the current formulation carries all tick
    economics inside ``vol_estimate`` and ``edge`` (both already in USD).
    """
    if vol_estimate <= 0.0:
        raise ValueError(
            f"vol_estimate must be > 0 (per-contract daily $ stdev), got {vol_estimate}"
        )

    vol_target_contracts = math.floor(cfg.target_daily_vol_usd / vol_estimate)

    kelly_contracts = math.floor(
        cfg.kelly_fraction_cap * edge.kelly_fraction * cfg.target_daily_vol_usd / edge.avg_loss
    )

    legs = (
        vol_target_contracts,
        kelly_contracts,
        survival_max_contracts,
        cfg.max_contracts,
    )
    return max(0, min(legs))
