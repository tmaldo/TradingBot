"""Vendor adapters -- the ONLY place vendor SDKs may be imported (G1/G2).

Each adapter implements :class:`futures_engine.data.provider.MarketDataProvider`.
Vendor SDKs (``databento``, ``norgatedata``, ``yfinance``) are imported lazily
*inside* methods so importing this package never requires an optional extra, and
so a test can assert no vendor import exists anywhere else in ``futures_engine``.

Normalisation from each vendor's raw payload to ``Bars`` is a pure function
(``parse_*``) tested against recorded fixtures; the network call is isolated in a
private ``_fetch_raw`` / ``_list_raw`` method that tests monkeypatch.
"""

from __future__ import annotations

from futures_engine.data.adapters.databento_adapter import (
    DatabentoAdapter,
    parse_databento_bars,
    parse_databento_contracts,
)
from futures_engine.data.adapters.norgate_adapter import NorgateAdapter, parse_norgate_bars
from futures_engine.data.adapters.yfinance_dev import YFinanceDevFetcher, parse_yfinance_bars

__all__ = [
    "DatabentoAdapter",
    "NorgateAdapter",
    "YFinanceDevFetcher",
    "parse_databento_bars",
    "parse_databento_contracts",
    "parse_norgate_bars",
    "parse_yfinance_bars",
]
