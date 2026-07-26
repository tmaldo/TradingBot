"""Tests for the provider protocol and vendor adapters (G1/G2).

Covers: no vendor SDK import anywhere outside ``adapters/`` (grepped); every
adapter satisfies :class:`MarketDataProvider`; pure parsers normalise recorded
vendor payloads correctly; the yfinance fetcher is dev-grade and refused by
``require_validation_grade``; a missing SDK raises a clear extra-naming error;
and ``fetch_bars`` works fully offline when the network call is mocked.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from futures_engine.core.types import BAR_COLUMNS, ContinuousMeta, DatasetMeta
from futures_engine.data.adapters import (
    DatabentoAdapter,
    NorgateAdapter,
    YFinanceDevFetcher,
    parse_databento_bars,
    parse_databento_contracts,
    parse_norgate_bars,
    parse_yfinance_bars,
)
from futures_engine.data.adapters import yfinance_dev as yfinance_dev_mod
from futures_engine.data.adapters.databento_adapter import _require_databento
from futures_engine.data.adapters.norgate_adapter import (
    _require_norgatedata,
    parse_norgate_contracts,
)
from futures_engine.data.adapters.yfinance_dev import _require_yfinance
from futures_engine.data.provider import MarketDataProvider
from futures_engine.data.store import (
    PENDING_SNAPSHOT_HASH,
    DataIntegrityError,
    require_validation_grade,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "futures_engine"
_VENDOR_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:databento|norgatedata|yfinance)\b", re.M)


# --- G2: no vendor import outside adapters/ ----------------------------------


def test_no_vendor_import_outside_adapters() -> None:
    adapters_dir = PACKAGE_ROOT / "data" / "adapters"
    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if adapters_dir in path.parents:
            continue
        if _VENDOR_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == [], f"vendor SDK imported outside adapters/: {offenders}"


# --- adapters satisfy the protocol -------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "expected_grade"),
    [
        (DatabentoAdapter(), True),
        (NorgateAdapter(), True),
        (YFinanceDevFetcher(), False),
    ],
)
def test_adapter_implements_protocol(adapter: object, expected_grade: bool) -> None:
    assert isinstance(adapter, MarketDataProvider)
    assert isinstance(adapter.name, str)  # type: ignore[attr-defined]
    assert adapter.validation_grade is expected_grade  # type: ignore[attr-defined]


# --- Databento parsers -------------------------------------------------------


def test_parse_databento_bars_scales_prices() -> None:
    records = json.loads((FIXTURES / "databento_ohlcv.json").read_text(encoding="utf-8"))
    bars = parse_databento_bars(records)
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert str(bars.index.tz) == "UTC"
    # raw open 4999000000000 * 1e-9 == 4999.0; first close == 5000.0
    assert bars["open"].iloc[0] == pytest.approx(4999.0)
    assert bars["close"].iloc[0] == pytest.approx(5000.0)
    assert bars.index[0] == pd.Timestamp("2024-06-03T21:00:00Z")


def test_parse_databento_contracts() -> None:
    records = json.loads((FIXTURES / "databento_defs.json").read_text(encoding="utf-8"))
    contracts = parse_databento_contracts(records)
    assert [c.symbol for c in contracts] == ["ESM4", "ESU4"]
    assert contracts[0].expiry == date(2024, 6, 21)
    assert contracts[0].first_trade == date(2023, 6, 16)


def test_parse_databento_bars_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing field"):
        parse_databento_bars([{"ts_event": 1, "open": 1, "high": 2, "low": 0}])


# --- Norgate parsers ---------------------------------------------------------


def test_parse_norgate_bars_carries_open_interest() -> None:
    frame = pd.read_csv(FIXTURES / "norgate_bars.csv")
    bars = parse_norgate_bars(frame)
    assert list(bars.columns) == [*BAR_COLUMNS, "open_interest"]
    assert str(bars.index.tz) == "UTC"
    assert bars["close"].iloc[0] == pytest.approx(5000.0)
    assert bars["open_interest"].iloc[0] == pytest.approx(2005000.0)


def test_parse_norgate_contracts() -> None:
    records = [
        {"symbol": "ESM4", "expiration": "2024-06-21", "first_trade": "2023-06-16"},
        {"symbol": "ESU4", "expiration": "2024-09-20", "first_trade": None},
    ]
    contracts = parse_norgate_contracts(records)
    assert contracts[0].expiry == date(2024, 6, 21)
    assert contracts[1].first_trade is None


# --- yfinance dev fetcher ----------------------------------------------------


def test_parse_yfinance_drops_adj_close() -> None:
    frame = pd.read_csv(FIXTURES / "yfinance_ohlc.csv")
    bars = parse_yfinance_bars(frame)
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert "adj_close" not in bars.columns
    assert str(bars.index.tz) == "UTC"


def test_yfinance_module_marked_not_for_validation() -> None:
    assert "NOT FOR VALIDATION" in (yfinance_dev_mod.__doc__ or "")


def test_require_validation_grade_refuses_yfinance_source() -> None:
    cont = ContinuousMeta(
        roll_rule="volume", adjustment="none", roll_dates=[], underlying_contracts=["MESU24"]
    )
    meta = DatasetMeta(
        symbol_root="MES",
        source=YFinanceDevFetcher.name,
        interval="1d",
        start=datetime(2024, 6, 3, tzinfo=UTC),
        end=datetime(2024, 6, 5, tzinfo=UTC),
        continuous=cont,
        snapshot_hash=PENDING_SNAPSHOT_HASH,
        as_of=datetime(2026, 7, 25, tzinfo=UTC),
        validation_grade=YFinanceDevFetcher.validation_grade,
    )
    with pytest.raises(DataIntegrityError, match="validation"):
        require_validation_grade(meta)


def test_yfinance_list_contracts_not_supported() -> None:
    with pytest.raises(NotImplementedError, match="NOT FOR VALIDATION"):
        YFinanceDevFetcher().list_contracts("MES", date(2024, 1, 1), date(2024, 12, 31))


# --- missing SDK errors name the extra ---------------------------------------


def test_missing_sdk_errors_name_extra() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"databento\]"):
        _require_databento()
    with pytest.raises(ModuleNotFoundError, match=r"norgate\]"):
        _require_norgatedata()
    with pytest.raises(ModuleNotFoundError, match=r"dev-data\]"):
        _require_yfinance()


# --- fetch_bars offline (network isolated & mockable) ------------------------


def test_databento_fetch_bars_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    records = json.loads((FIXTURES / "databento_ohlcv.json").read_text(encoding="utf-8"))
    adapter = DatabentoAdapter(api_key="unused")
    monkeypatch.setattr(adapter, "_fetch_raw", lambda *a, **k: records)
    bars = adapter.fetch_bars(
        "ESM4", datetime(2024, 6, 3, tzinfo=UTC), datetime(2024, 6, 5, tzinfo=UTC), "1d"
    )
    assert bars["close"].iloc[0] == pytest.approx(5000.0)
    assert list(bars.columns) == list(BAR_COLUMNS)


def test_norgate_fetch_bars_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.read_csv(FIXTURES / "norgate_bars.csv")
    adapter = NorgateAdapter()
    monkeypatch.setattr(adapter, "_fetch_raw", lambda *a, **k: frame)
    bars = adapter.fetch_bars(
        "ESM4", datetime(2024, 6, 3, tzinfo=UTC), datetime(2024, 6, 5, tzinfo=UTC), "1d"
    )
    assert "open_interest" in bars.columns
    assert bars["close"].iloc[-1] == pytest.approx(4995.25)
