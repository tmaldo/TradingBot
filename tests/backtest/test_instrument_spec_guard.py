"""The instrument-spec drift guard (T6's single biggest risk).

Nautilus contract economics MUST equal our :class:`InstrumentSpec`, or PnL scales
silently. These tests pin the factory + runtime assertion.
"""

from __future__ import annotations

import pytest
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from futures_engine.backtest.instrument import (
    InstrumentSpecMismatchError,
    assert_spec_parity,
    build_nautilus_instrument,
)
from futures_engine.core.types import InstrumentSpec

MES = InstrumentSpec(
    symbol_root="MES",
    exchange="CME",
    tick_size=0.25,
    tick_value=1.25,
    multiplier=5,
    currency="USD",
)
MNQ = InstrumentSpec(
    symbol_root="MNQ",
    exchange="CME",
    tick_size=0.25,
    tick_value=0.5,
    multiplier=2,
    currency="USD",
)


@pytest.mark.parametrize("spec", [MES, MNQ])
def test_factory_copies_spec_economics(spec: InstrumentSpec) -> None:
    inst = build_nautilus_instrument(spec, Venue("GLBX"))
    assert float(inst.multiplier) == spec.multiplier
    assert float(inst.price_increment.as_double()) == spec.tick_size
    # Built instrument passes its own parity guard.
    assert_spec_parity(inst, spec)


def test_guard_catches_multiplier_drift() -> None:
    # Nautilus's stock ES future has multiplier=1; the real MES is 5 -> must raise.
    stock_es = TestInstrumentProvider.es_future(2024, 3)
    assert float(stock_es.multiplier) == 1  # documents the footgun
    with pytest.raises(InstrumentSpecMismatchError, match="multiplier mismatch"):
        assert_spec_parity(stock_es, MES)


def test_guard_catches_tick_value_drift() -> None:
    # Same multiplier & tick but a corrupted tick_value must be caught.
    bad = InstrumentSpec(
        symbol_root="MES",
        exchange="CME",
        tick_size=0.25,
        tick_value=1.25,
        multiplier=5,
        currency="USD",
    )
    inst = build_nautilus_instrument(bad, Venue("GLBX"))
    # Fabricate a spec that lies about tick_value relative to multiplier*tick.
    liar = bad.model_copy(update={"tick_size": 0.5, "tick_value": 2.5, "multiplier": 5})
    with pytest.raises(InstrumentSpecMismatchError, match="tick_size mismatch"):
        assert_spec_parity(inst, liar)
