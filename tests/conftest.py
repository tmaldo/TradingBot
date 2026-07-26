"""Shared pytest fixtures: recorded per-contract bar data for the data layer.

The CSVs under ``tests/fixtures/continuous/`` are produced by
``tests/fixtures/_generate.py`` (five MES quarterly contracts, four rolls,
contango prices, engineered volume/OI crossovers, US-holiday gaps, spanning the
2024 DST change). Loaders here return them as UTC-indexed ``Bars`` frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from futures_engine.data.provider import ContractInfo

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CONTINUOUS_DIR = FIXTURES_DIR / "continuous"

# symbol -> (expiry, first_trade); mirrors tests/fixtures/_generate.py::CONTRACTS.
_CONTRACT_META: list[tuple[str, date, date]] = [
    ("MESH24", date(2024, 3, 15), date(2023, 12, 18)),
    ("MESM24", date(2024, 6, 21), date(2024, 2, 26)),
    ("MESU24", date(2024, 9, 20), date(2024, 5, 28)),
    ("MESZ24", date(2024, 12, 20), date(2024, 8, 26)),
    ("MESH25", date(2025, 3, 21), date(2024, 11, 25)),
]


def _load_contract(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(CONTINUOUS_DIR / f"{symbol}.csv")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True))
    frame.index.name = "timestamp"
    return frame


@dataclass(frozen=True)
class ContinuousFixture:
    """The five-contract roll fixture: per-contract bars plus contract specs."""

    per_contract: dict[str, pd.DataFrame]
    specs: list[ContractInfo]

    @property
    def symbols(self) -> list[str]:
        return [c.symbol for c in self.specs]


@pytest.fixture
def continuous_fixture() -> ContinuousFixture:
    per_contract = {sym: _load_contract(sym) for sym, _, _ in _CONTRACT_META}
    specs = [
        ContractInfo(symbol=sym, expiry=expiry, first_trade=first)
        for sym, expiry, first in _CONTRACT_META
    ]
    return ContinuousFixture(per_contract=per_contract, specs=specs)
