"""Databento adapter -- validation-grade CME futures bars.

The vendor SDK (``databento``) is imported lazily inside :meth:`_fetch_raw` /
:meth:`_list_raw` only. Parsing is pure: :func:`parse_databento_bars` and
:func:`parse_databento_contracts` turn raw DBN records (as dicts) into ``Bars`` /
``ContractInfo`` and are tested against recorded fixtures offline.

Databento encodes prices as fixed-point integers scaled by 1e-9 and timestamps
as int64 nanoseconds since the UTC epoch (``ts_event``); both are normalised
here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

from futures_engine.core.types import BAR_COLUMNS, BarInterval, Bars
from futures_engine.data.provider import ContractInfo

# Databento fixed-point price scale: raw integer * 1e-9 == price in index points.
_DATABENTO_PRICE_SCALE = 1e-9
_PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
_INTERVAL_TO_SCHEMA: dict[BarInterval, str] = {
    "1m": "ohlcv-1m",
    "5m": "ohlcv-5m",
    "15m": "ohlcv-15m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}
_MISSING_SDK_MSG = (
    "the 'databento' SDK is not installed; install the optional extra: "
    "pip install futures-engine[databento]"
)


def _require_databento() -> Any:
    try:
        import databento
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_SDK_MSG) from exc
    return databento


def parse_databento_bars(records: Iterable[Mapping[str, Any]]) -> Bars:
    """Normalise raw Databento OHLCV records into ``Bars`` (UTC index, float64)."""
    rows = list(records)
    if not rows:
        raise ValueError("no Databento OHLCV records to parse")
    frame = pd.DataFrame(rows)
    required = {"ts_event", *BAR_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Databento records missing field(s): {sorted(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame["ts_event"], utc=True, unit="ns"))
    out = pd.DataFrame(index=index)
    for col in _PRICE_FIELDS:
        out[col] = frame[col].to_numpy(dtype="float64") * _DATABENTO_PRICE_SCALE
    out["volume"] = frame["volume"].to_numpy(dtype="float64")
    out.index.name = "ts_event"
    return out.loc[:, list(BAR_COLUMNS)].sort_index()


def parse_databento_contracts(records: Iterable[Mapping[str, Any]]) -> list[ContractInfo]:
    """Normalise raw Databento definition records into ``ContractInfo``."""

    def _as_date(value: Any) -> date:
        return pd.Timestamp(value, unit="ns", tz="UTC").date()

    contracts = []
    for rec in records:
        activation = rec.get("activation")
        contracts.append(
            ContractInfo(
                symbol=str(rec["raw_symbol"]),
                expiry=_as_date(rec["expiration"]),
                first_trade=_as_date(activation) if activation is not None else None,
            )
        )
    return contracts


class DatabentoAdapter:
    """:class:`MarketDataProvider` backed by Databento historical data."""

    name = "databento"
    validation_grade = True

    def __init__(self, api_key: str | None = None, dataset: str = "GLBX.MDP3") -> None:
        self._api_key = api_key
        self._dataset = dataset

    def _fetch_raw(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> list[Mapping[str, Any]]:  # pragma: no cover - network path, mocked in tests
        databento = _require_databento()
        client = databento.Historical(self._api_key)
        store = client.timeseries.get_range(
            dataset=self._dataset,
            symbols=[contract],
            schema=_INTERVAL_TO_SCHEMA[interval],
            start=start,
            end=end,
        )
        return [dict(rec) for rec in store]

    def _list_raw(
        self, symbol_root: str, start: date, end: date
    ) -> list[Mapping[str, Any]]:  # pragma: no cover - network path, mocked in tests
        databento = _require_databento()
        client = databento.Historical(self._api_key)
        store = client.timeseries.get_range(
            dataset=self._dataset,
            symbols=[f"{symbol_root}.FUT"],
            stype_in="parent",
            schema="definition",
            start=str(start),
            end=str(end),
        )
        return [dict(rec) for rec in store]

    def fetch_bars(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> Bars:
        return parse_databento_bars(self._fetch_raw(contract, start, end, interval))

    def list_contracts(self, symbol_root: str, start: date, end: date) -> list[ContractInfo]:
        return parse_databento_contracts(self._list_raw(symbol_root, start, end))
