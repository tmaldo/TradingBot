"""Deterministic generator for the continuous-contract test fixtures.

Run with the project venv to (re)create ``tests/fixtures/continuous/*.csv``::

    python tests/fixtures/_generate.py

The fixtures are five MES quarterly contracts (four roll events) with:

- daily bars stamped at 21:00 UTC on the US-business-day calendar, minus a
  curated set of 2024/25 exchange holidays (holiday gaps in the series);
- a **contango** price structure: contract *k* sits exactly ``+PRICE_STEP*k``
  above a shared base path, so every inter-contract gap equals ``PRICE_STEP``
  (exercises panama/ratio back-adjustment);
- volume / open-interest ramps in each overlap so the back contract's volume
  crosses above the front's at 50% through the overlap, while its open interest
  crosses earlier (30%) -- so the volume and open-interest roll dates differ;
- the first roll's overlap spans the 2024-03-10 US DST change.

Everything is a closed-form function of the calendar (no randomness, no
wall-clock reads), so the CSVs are reproducible byte-for-byte.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "continuous"

# Contango step (index points) between adjacent contracts -> every roll gap.
PRICE_STEP = 25.0
BASE_PRICE = 5000.0
DRIFT_PER_DAY = 0.5
BAR_HOUR_UTC = 21

# (symbol, expiry, first_trade) -- real-ish 3rd-Friday quarterly schedule.
CONTRACTS: list[tuple[str, date, date]] = [
    ("MESH24", date(2024, 3, 15), date(2023, 12, 18)),
    ("MESM24", date(2024, 6, 21), date(2024, 2, 26)),
    ("MESU24", date(2024, 9, 20), date(2024, 5, 28)),
    ("MESZ24", date(2024, 12, 20), date(2024, 8, 26)),
    ("MESH25", date(2025, 3, 21), date(2024, 11, 25)),
]

# Curated US equity-index holidays (weekdays) in range -> gaps in the calendar.
HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 3, 29),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 11, 28),
        date(2024, 12, 25),
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
    }
)

CALENDAR_START = date(2023, 12, 18)
CALENDAR_END = date(2025, 3, 21)


def _sessions() -> list[date]:
    days = pd.bdate_range(CALENDAR_START, CALENDAR_END).date
    return [d for d in days if d not in HOLIDAYS]


def _base_path(sessions: list[date]) -> dict[date, float]:
    origin = sessions[0].toordinal()
    return {
        d: BASE_PRICE + DRIFT_PER_DAY * (d.toordinal() - origin) + 2.0 * math.sin(0.3 * i)
        for i, d in enumerate(sessions)
    }


# Crossover fractions through an overlap (back contract overtakes the front).
VOL_CROSS_FRAC = 0.50
OI_CROSS_FRAC = 0.30
SOLO_VOLUME = 10_000.0
SOLO_OPEN_INTEREST = 15_000.0


def _ramp(x: float, cross: float, scale: float, *, back: bool) -> float:
    """A line through ``(cross, scale)``; back rises with x, front falls -- so a
    back/front pair crosses exactly at fraction ``cross`` through the overlap."""
    sign = 1.0 if back else -1.0
    return scale + sign * scale * (x - cross)


def build_contract(idx: int, sessions: list[date], base: dict[date, float]) -> pd.DataFrame:
    _symbol, expiry, first_trade = CONTRACTS[idx]
    my_sessions = [d for d in sessions if first_trade <= d <= expiry]

    prev_expiry = CONTRACTS[idx - 1][1] if idx > 0 else None
    next_start = CONTRACTS[idx + 1][2] if idx + 1 < len(CONTRACTS) else None

    rows = []
    for d in my_sessions:
        close = base[d] + PRICE_STEP * idx
        # Volume / OI: full when sole front month; crossing ramps inside overlaps.
        volume = SOLO_VOLUME
        open_interest = SOLO_OPEN_INTEREST
        overlap: list[date] | None = None
        back = False
        if prev_expiry is not None and d <= prev_expiry:
            overlap = [s for s in sessions if first_trade <= s <= prev_expiry]
            back = True  # I am the back contract vs the previous one
        elif next_start is not None and d >= next_start:
            overlap = [s for s in sessions if next_start <= s <= expiry]
            back = False  # I am the front contract vs the next one
        if overlap is not None:
            m = len(overlap)
            x = 0.5 if m == 1 else overlap.index(d) / (m - 1)
            volume = _ramp(x, VOL_CROSS_FRAC, SOLO_VOLUME / 2, back=back)
            open_interest = _ramp(x, OI_CROSS_FRAC, SOLO_OPEN_INTEREST / 2, back=back)
        ts = datetime.combine(d, time(BAR_HOUR_UTC, 0), tzinfo=UTC)
        rows.append(
            {
                "timestamp": ts,
                "open": close - 0.25,
                "high": max(close, close - 0.25) + 0.5,
                "low": min(close, close - 0.25) - 0.5,
                "close": close,
                "volume": round(volume, 2),
                "open_interest": round(open_interest, 2),
            }
        )
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index.name = "timestamp"
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = _sessions()
    base = _base_path(sessions)
    for idx, (symbol, _expiry, _first) in enumerate(CONTRACTS):
        frame = build_contract(idx, sessions, base)
        frame.to_csv(OUT_DIR / f"{symbol}.csv")
        print(f"{symbol}: {len(frame)} bars {frame.index[0].date()}..{frame.index[-1].date()}")

    # Report detected volume/OI crossovers per adjacent pair (design aid).
    for i in range(len(CONTRACTS) - 1):
        front = pd.read_csv(
            OUT_DIR / f"{CONTRACTS[i][0]}.csv", index_col="timestamp", parse_dates=["timestamp"]
        )
        back = pd.read_csv(
            OUT_DIR / f"{CONTRACTS[i + 1][0]}.csv",
            index_col="timestamp",
            parse_dates=["timestamp"],
        )
        common = front.index.intersection(back.index)
        for col in ("volume", "open_interest"):
            lead = (back.loc[common, col] > front.loc[common, col]).to_numpy()
            first_true = int(np.argmax(lead)) if lead.any() else -1
            # first session completing a 2-in-a-row run:
            roll_pos = next(
                (k for k in range(1, len(lead)) if lead[k] and lead[k - 1]),
                -1,
            )
            print(
                f"  {CONTRACTS[i][0]}->{CONTRACTS[i + 1][0]} {col}: "
                f"first_cross={common[first_true].date()} "
                f"roll(confirm=2)={common[roll_pos].date()}"
            )


if __name__ == "__main__":
    main()
