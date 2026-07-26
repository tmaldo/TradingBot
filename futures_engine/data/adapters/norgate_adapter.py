"""Norgate Data adapter -- validation-grade continuous/individual futures bars.

The vendor SDK (``norgatedata``) is imported lazily inside the network methods
only. :func:`parse_norgate_bars` is a pure normaliser (tested against a recorded
fixture) turning a ``price_timeseries``-shaped frame into ``Bars`` with an
optional ``open_interest`` column.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import pandas as pd

from futures_engine.core.types import BAR_COLUMNS, BarInterval, Bars
from futures_engine.data.provider import ContractInfo

# Norgate title-case columns -> our lower_snake schema.
_COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Open Interest": "open_interest",
}
_MISSING_SDK_MSG = (
    "the 'norgatedata' SDK is not installed; install the optional extra: "
    "pip install futures-engine[norgate]"
)


def _require_norgatedata() -> Any:
    try:
        import norgatedata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_SDK_MSG) from exc
    return norgatedata


def parse_norgate_bars(frame: pd.DataFrame) -> Bars:
    """Normalise a norgatedata ``price_timeseries`` frame into ``Bars`` (UTC index).

    Accepts either a ``Date``-column frame (as read from CSV) or a date-indexed
    frame (as returned by ``norgatedata``). Carries ``Open Interest`` through as
    an ``open_interest`` column when present.
    """
    data = frame.copy()
    if "Date" in data.columns:
        data = data.set_index("Date")
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in data.columns]
    if missing:
        raise ValueError(f"Norgate frame missing column(s): {missing}")
    out = pd.DataFrame(index=index)
    for src, dst in _COLUMN_MAP.items():
        if src in data.columns:
            out[dst] = data[src].to_numpy(dtype="float64")
    out.index.name = "timestamp"
    keep = [*BAR_COLUMNS, "open_interest"] if "open_interest" in out.columns else list(BAR_COLUMNS)
    return out.loc[:, keep].sort_index()


def parse_norgate_contracts(records: list[dict[str, Any]]) -> list[ContractInfo]:
    """Normalise Norgate contract-metadata records into ``ContractInfo``."""
    contracts = []
    for rec in records:
        first = rec.get("first_trade")
        contracts.append(
            ContractInfo(
                symbol=str(rec["symbol"]),
                expiry=pd.to_datetime(rec["expiration"], utc=True).date(),
                first_trade=pd.to_datetime(first, utc=True).date() if first else None,
            )
        )
    return contracts


class NorgateAdapter:
    """:class:`MarketDataProvider` backed by Norgate Data."""

    name = "norgate"
    validation_grade = True

    def _fetch_raw(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> pd.DataFrame:  # pragma: no cover - network path, mocked in tests
        norgatedata = _require_norgatedata()
        frame = norgatedata.price_timeseries(
            contract,
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            interval=interval,
        )
        return cast(pd.DataFrame, frame)

    def _list_raw(
        self, symbol_root: str, start: date, end: date
    ) -> list[dict[str, Any]]:  # pragma: no cover - network path, mocked in tests
        norgatedata = _require_norgatedata()
        return list(norgatedata.futures_market_session_contracts(symbol_root))

    def fetch_bars(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> Bars:
        return parse_norgate_bars(self._fetch_raw(contract, start, end, interval))

    def list_contracts(self, symbol_root: str, start: date, end: date) -> list[ContractInfo]:
        return parse_norgate_contracts(self._list_raw(symbol_root, start, end))
