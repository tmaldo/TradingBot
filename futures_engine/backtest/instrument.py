"""Build a Nautilus ``FuturesContract`` *from* our :class:`InstrumentSpec`.

The single biggest risk in wiring Nautilus into this system is **instrument-spec
drift**: Nautilus's own contract multiplier / tick size / price precision can
silently differ from :class:`~futures_engine.core.types.InstrumentSpec` and scale
every PnL figure without any error (Nautilus's own ``TestInstrumentProvider``
ships an ES contract with ``multiplier = 1`` where the real ES is ``50``; MES = 5,
MNQ = 2). To make that impossible, *all* Nautilus instruments are built through
:func:`build_nautilus_instrument`, which copies the spec's economics verbatim, and
:func:`assert_spec_parity` re-checks them at runtime before any backtest runs.
"""

from __future__ import annotations

import math

import pandas as pd
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Price, Quantity

from futures_engine.core.types import InstrumentSpec


class InstrumentSpecMismatchError(Exception):
    """Raised when a Nautilus instrument's economics diverge from the spec."""


def _price_precision(tick_size: float) -> int:
    """Decimal places implied by ``tick_size`` (e.g. 0.25 -> 2, 0.01 -> 2)."""
    text = format(tick_size, "f").rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def build_nautilus_instrument(spec: InstrumentSpec, venue: Venue) -> FuturesContract:
    """Construct a Nautilus ``FuturesContract`` whose economics equal ``spec``.

    ``multiplier``, ``price_increment`` (tick size) and ``price_precision`` are
    taken directly from ``spec`` so Nautilus and T2 agree by construction; the
    result is still re-validated by :func:`assert_spec_parity`.
    """
    if not float(spec.multiplier).is_integer():
        raise InstrumentSpecMismatchError(
            f"Nautilus requires an integer contract multiplier; spec multiplier "
            f"is {spec.multiplier} for {spec.symbol_root}"
        )
    precision = _price_precision(spec.tick_size)
    raw = Symbol(f"{spec.symbol_root}=FUT")
    instrument = FuturesContract(
        instrument_id=InstrumentId(symbol=raw, venue=venue),
        raw_symbol=raw,
        asset_class=AssetClass.INDEX,
        exchange=spec.exchange,
        currency=USD,
        price_precision=precision,
        price_increment=Price(spec.tick_size, precision),
        multiplier=Quantity.from_int(int(spec.multiplier)),
        lot_size=Quantity.from_int(1),
        underlying=spec.symbol_root,
        activation_ns=0,
        expiration_ns=pd.Timestamp("2100-01-01", tz="UTC").value,
        ts_event=0,
        ts_init=0,
    )
    assert_spec_parity(instrument, spec)
    return instrument


def assert_spec_parity(instrument: FuturesContract, spec: InstrumentSpec) -> None:
    """Assert a Nautilus instrument's economics equal ``spec`` (raises on drift).

    Guards multiplier, tick size, and the implied tick value
    (``multiplier * tick_size``) -- the three quantities that scale PnL. Called
    both inside :func:`build_nautilus_instrument` and by the runner before every
    run, so spec drift can never silently corrupt a result.
    """
    nautilus_mult = float(instrument.multiplier)
    if not math.isclose(nautilus_mult, spec.multiplier, rel_tol=1e-12, abs_tol=1e-12):
        raise InstrumentSpecMismatchError(
            f"multiplier mismatch: Nautilus {nautilus_mult} != spec {spec.multiplier} "
            f"for {spec.symbol_root}"
        )
    nautilus_tick = float(instrument.price_increment.as_double())
    if not math.isclose(nautilus_tick, spec.tick_size, rel_tol=1e-12, abs_tol=1e-12):
        raise InstrumentSpecMismatchError(
            f"tick_size mismatch: Nautilus {nautilus_tick} != spec {spec.tick_size} "
            f"for {spec.symbol_root}"
        )
    nautilus_tick_value = nautilus_mult * nautilus_tick
    if not math.isclose(nautilus_tick_value, spec.tick_value, rel_tol=1e-9, abs_tol=1e-9):
        raise InstrumentSpecMismatchError(
            f"tick_value mismatch: Nautilus multiplier*tick {nautilus_tick_value} != "
            f"spec tick_value {spec.tick_value} for {spec.symbol_root}"
        )
