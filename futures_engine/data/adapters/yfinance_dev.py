"""yfinance DEV fetcher -- NOT FOR VALIDATION (Global Constraint G1).

yfinance is a convenience source for local exploration ONLY. Its data is not
contract-accurate, not point-in-time, and unsuitable for research, backtest, or
validation. This fetcher advertises ``validation_grade = False`` so
:func:`futures_engine.data.store.require_validation_grade` refuses any dataset
sourced from it in those paths. Do NOT add this to any research/backtest code.

The vendor SDK (``yfinance``) is imported lazily inside :meth:`_fetch_raw` only.
:func:`parse_yfinance_bars` is a pure normaliser tested against a recorded
fixture.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import pandas as pd

from futures_engine.core.types import BAR_COLUMNS, BarInterval, Bars
from futures_engine.data.provider import ContractInfo

_COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
_MISSING_SDK_MSG = (
    "the 'yfinance' dev SDK is not installed; install the optional extra: "
    "pip install futures-engine[dev-data]"
)


def _require_yfinance() -> Any:
    try:
        import yfinance
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING_SDK_MSG) from exc
    return yfinance


def parse_yfinance_bars(frame: pd.DataFrame) -> Bars:
    """Normalise a yfinance ``download`` frame into ``Bars`` (UTC index, float64).

    Drops ``Adj Close``; keeps only the OHLCV columns. Marked NOT FOR VALIDATION.
    """
    data = frame.copy()
    if "Date" in data.columns:
        data = data.set_index("Date")
    elif "Datetime" in data.columns:
        data = data.set_index("Datetime")
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    missing = [c for c in _COLUMN_MAP if c not in data.columns]
    if missing:
        raise ValueError(f"yfinance frame missing column(s): {missing}")
    out = pd.DataFrame(index=index)
    for src, dst in _COLUMN_MAP.items():
        out[dst] = data[src].to_numpy(dtype="float64")
    out.index.name = "timestamp"
    return out.loc[:, list(BAR_COLUMNS)].sort_index()


class YFinanceDevFetcher:
    """Dev-only :class:`MarketDataProvider`; ``validation_grade`` is always False."""

    name = "yfinance-dev"
    validation_grade = False

    def _fetch_raw(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> pd.DataFrame:  # pragma: no cover - network path, mocked in tests
        yfinance = _require_yfinance()
        frame = yfinance.download(
            contract,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=interval,
            progress=False,
        )
        return cast(pd.DataFrame, frame)

    def fetch_bars(
        self, contract: str, start: datetime, end: datetime, interval: BarInterval
    ) -> Bars:
        return parse_yfinance_bars(self._fetch_raw(contract, start, end, interval))

    def list_contracts(self, symbol_root: str, start: date, end: date) -> list[ContractInfo]:
        raise NotImplementedError(
            "yfinance-dev does not enumerate futures contracts; it is a dev-only "
            "price source and NOT FOR VALIDATION"
        )
