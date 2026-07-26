"""Generate hand-shaped vendor-payload fixtures for the adapter parser tests.

These mirror the *raw* shapes each SDK emits (not already-normalised frames):

- ``databento_ohlcv.json``    -- DBN OHLCV records: ns ``ts_event`` + int prices
  fixed-point scaled by 1e-9.
- ``databento_defs.json``     -- DBN definition records: raw_symbol + ns
  ``activation`` / ``expiration``.
- ``norgate_bars.csv``        -- norgatedata ``price_timeseries`` frame: naive
  ``Date`` index, title-case columns incl. ``Open Interest``.
- ``yfinance_ohlc.csv``       -- yfinance ``download`` frame: naive ``Date``
  index, title-case columns incl. ``Adj Close``.

No network; everything is closed-form. Run: ``python tests/fixtures/_generate_vendor.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
_SCALE = 1_000_000_000  # databento fixed-point price scale (1e-9)


def _ns(ts: str) -> int:
    return int(pd.Timestamp(ts).value)


def main() -> None:
    days = ["2024-06-03", "2024-06-04", "2024-06-05"]
    closes = [5000.0, 5012.5, 4995.25]

    # --- Databento OHLCV records (raw DBN/JSON shape) ---
    ohlcv = []
    for day, close in zip(days, closes, strict=True):
        o, h, low, c = close - 1.0, close + 3.0, close - 4.0, close
        ohlcv.append(
            {
                "ts_event": _ns(f"{day}T21:00:00Z"),
                "rtype": 34,
                "instrument_id": 12345,
                "open": round(o * _SCALE),
                "high": round(h * _SCALE),
                "low": round(low * _SCALE),
                "close": round(c * _SCALE),
                "volume": 100000 + int(close),
            }
        )
    (OUT / "databento_ohlcv.json").write_text(json.dumps(ohlcv, indent=2), encoding="utf-8")

    # --- Databento definition records ---
    defs = [
        {
            "raw_symbol": "ESM4",
            "instrument_id": 12345,
            "activation": _ns("2023-06-16T00:00:00Z"),
            "expiration": _ns("2024-06-21T13:30:00Z"),
        },
        {
            "raw_symbol": "ESU4",
            "instrument_id": 12346,
            "activation": _ns("2023-09-15T00:00:00Z"),
            "expiration": _ns("2024-09-20T13:30:00Z"),
        },
    ]
    (OUT / "databento_defs.json").write_text(json.dumps(defs, indent=2), encoding="utf-8")

    # --- norgatedata price_timeseries frame ---
    norgate = pd.DataFrame(
        {
            "Date": days,
            "Open": [c - 1.0 for c in closes],
            "High": [c + 3.0 for c in closes],
            "Low": [c - 4.0 for c in closes],
            "Close": closes,
            "Volume": [100000 + int(c) for c in closes],
            "Open Interest": [2000000 + int(c) for c in closes],
        }
    )
    norgate.to_csv(OUT / "norgate_bars.csv", index=False)

    # --- yfinance download frame ---
    yf = pd.DataFrame(
        {
            "Date": days,
            "Open": [c - 1.0 for c in closes],
            "High": [c + 3.0 for c in closes],
            "Low": [c - 4.0 for c in closes],
            "Close": closes,
            "Adj Close": [c - 0.5 for c in closes],
            "Volume": [100000 + int(c) for c in closes],
        }
    )
    yf.to_csv(OUT / "yfinance_ohlc.csv", index=False)
    print("wrote databento/norgate/yfinance fixtures to", OUT)


if __name__ == "__main__":
    main()
