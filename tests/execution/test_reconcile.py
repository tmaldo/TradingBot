"""Reconciliation on reconnect (G16): one test per mismatch class.

Broker state is the source of truth. Each mismatch class -- an extra broker fill
local never saw, a local fill the broker never got, and a net-position mismatch
-- is detected and the corrective (broker-authoritative) positions are produced.
"""

from __future__ import annotations

from futures_engine.execution.client import Position
from futures_engine.execution.reconcile import (
    BrokerSnapshot,
    Fill,
    LocalState,
    reconcile,
)


def test_in_sync_states_produce_no_discrepancies() -> None:
    fills = [Fill("f1", "MES", 2)]
    pos = [Position("MES", 2, 5000.0)]
    report = reconcile(LocalState(fills, pos), BrokerSnapshot(fills, pos))
    assert report.in_sync
    assert report.discrepancies == []


def test_extra_broker_fill_detected_and_corrected() -> None:
    # The broker filled f2 during our outage; local never recorded it.
    local = LocalState(fills=[Fill("f1", "MES", 2)], positions=[Position("MES", 2, 5000.0)])
    broker = BrokerSnapshot(
        fills=[Fill("f1", "MES", 2), Fill("f2", "MES", 1)],
        positions=[Position("MES", 3, 5010.0)],
    )
    report = reconcile(local, broker)
    kinds = {d.kind for d in report.discrepancies}
    assert "extra_fill" in kinds
    assert any(d.kind == "extra_fill" and "f2" in d.detail for d in report.discrepancies)
    # broker is authoritative: local should be corrected to the broker position.
    assert report.corrected_positions == broker.positions


def test_missed_local_fill_detected() -> None:
    # Local believes f9 filled; the broker has no record of it.
    local = LocalState(
        fills=[Fill("f1", "MES", 2), Fill("f9", "MES", 1)],
        positions=[Position("MES", 3, 5000.0)],
    )
    broker = BrokerSnapshot(fills=[Fill("f1", "MES", 2)], positions=[Position("MES", 2, 5000.0)])
    report = reconcile(local, broker)
    kinds = {d.kind for d in report.discrepancies}
    assert "missed_fill" in kinds
    assert any(d.kind == "missed_fill" and "f9" in d.detail for d in report.discrepancies)
    assert report.corrected_positions == broker.positions


def test_position_mismatch_with_matching_fills_detected() -> None:
    # Identical fill sets, but the local net position is wrong (bookkeeping bug).
    fills = [Fill("f1", "MES", 2)]
    local = LocalState(fills=fills, positions=[Position("MES", 4, 5000.0)])
    broker = BrokerSnapshot(fills=fills, positions=[Position("MES", 2, 5000.0)])
    report = reconcile(local, broker)
    kinds = {d.kind for d in report.discrepancies}
    assert kinds == {"position_mismatch"}
    assert any("MES" in d.detail for d in report.discrepancies)
    assert report.corrected_positions == broker.positions


def test_missing_instrument_position_is_a_mismatch() -> None:
    # Broker holds a position in an instrument local thinks is flat.
    local = LocalState(fills=[], positions=[])
    broker = BrokerSnapshot(fills=[], positions=[Position("MNQ", -1, 18000.0)])
    report = reconcile(local, broker)
    assert any(d.kind == "position_mismatch" and "MNQ" in d.detail for d in report.discrepancies)
    assert report.corrected_positions == broker.positions
