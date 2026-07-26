"""Shared, validated value types for the futures engine.

These are the authoritative type definitions that downstream tasks code against.
``DatasetMeta`` / ``ContinuousMeta`` / ``Bars`` are *type definitions only* in
task T0 -- the data-layer behaviours (snapshot store, continuous builder) arrive
in a later task. Do not build the store here; just define the shapes.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bar sampling intervals supported across the engine.
BarInterval = Literal["1m", "5m", "15m", "1h", "1d"]

# A price/volume frame: UTC DatetimeIndex; columns open, high, low, close, volume
# (all float64). This is a plain pandas type alias -- validation of the frame's
# shape/dtypes belongs to the data layer (later task), not here.
Bars = pd.DataFrame

# Column contract for a ``Bars`` frame, exposed for downstream builders/validators.
BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class InstrumentSpec(BaseModel):
    """Tick economics and identity for a single tradable instrument.

    Invariant: ``tick_value == multiplier * tick_size`` (the dollar value of one
    tick must equal the contract multiplier applied to one tick of price). This
    is enforced so a mis-typed config cannot silently corrupt every P&L figure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol_root: str = Field(min_length=1)
    exchange: str = Field(min_length=1)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    multiplier: float = Field(gt=0)
    currency: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_tick_economics(self) -> InstrumentSpec:
        expected = self.multiplier * self.tick_size
        if not math.isclose(self.tick_value, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"tick_value ({self.tick_value}) must equal multiplier * tick_size "
                f"({self.multiplier} * {self.tick_size} = {expected}) for "
                f"{self.symbol_root}"
            )
        return self


class ContinuousMeta(BaseModel):
    """How a continuous contract series was stitched together (Global Constraint G3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roll_rule: Literal["volume", "open_interest", "calendar"]
    adjustment: Literal["panama_diff", "ratio", "none"]
    roll_dates: list[date]
    underlying_contracts: list[str]


class DatasetMeta(BaseModel):
    """Provenance for a dataset snapshot (Global Constraints G3/G4).

    ``continuous`` is required for futures series; backtests fail loudly on
    futures data lacking ``ContinuousMeta`` (enforced by the backtest layer, a
    later task). ``validation_grade=False`` marks dev-only data that the
    pipeline must refuse in research/backtest/validation paths (G1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol_root: str = Field(min_length=1)
    source: str = Field(min_length=1)
    interval: BarInterval
    start: datetime
    end: datetime
    continuous: ContinuousMeta | None
    snapshot_hash: str = Field(min_length=1)
    as_of: datetime
    validation_grade: bool
