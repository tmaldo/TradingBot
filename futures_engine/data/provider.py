"""The vendor-agnostic market-data provider interface (Global Constraint G2).

Every source of bars -- Databento, Norgate, or the dev-only yfinance fetcher --
implements :class:`MarketDataProvider`. Nothing outside
:mod:`futures_engine.data.adapters` may import a vendor SDK; the rest of the
engine depends only on this protocol, so vendor selection is a config choice and
never leaks into research/backtest code.

``validation_grade`` is part of the interface on purpose: it lets
:func:`futures_engine.data.store.require_validation_grade` refuse dev-grade data
in any research/backtest/validation path (G1).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from futures_engine.core.types import BarInterval, Bars


class ContractInfo(BaseModel):
    """Identity and lifecycle dates for a single futures contract month.

    ``expiry`` is the contract's last trade / expiration date; ``first_trade``
    is the first date the contract was listed for trading (``None`` when the
    vendor does not report it). These drive continuous-contract roll scheduling
    (see :func:`futures_engine.data.continuous.build_continuous`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    expiry: date
    first_trade: date | None = None


@runtime_checkable
class MarketDataProvider(Protocol):
    """A source of historical futures bars.

    Implementations live in :mod:`futures_engine.data.adapters` and lazily import
    their vendor SDK inside methods only. ``name`` identifies the source (stored
    in ``DatasetMeta.source``); ``validation_grade`` is ``False`` only for the
    dev fetcher, which the pipeline refuses in validation paths (G1).
    """

    name: str
    validation_grade: bool

    def fetch_bars(
        self,
        contract: str,
        start: datetime,
        end: datetime,
        interval: BarInterval,
    ) -> Bars:
        """Return OHLCV bars for a single ``contract`` over ``[start, end]``.

        The returned frame has a UTC :class:`~pandas.DatetimeIndex` and the
        columns in ``futures_engine.core.types.BAR_COLUMNS``.
        """
        ...

    def list_contracts(
        self,
        symbol_root: str,
        start: date,
        end: date,
    ) -> list[ContractInfo]:
        """List the contract months of ``symbol_root`` active within ``[start, end]``."""
        ...
