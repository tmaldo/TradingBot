"""Broker-vs-local reconciliation on reconnect (G16).

After a disconnect the local view and the broker's view can diverge. On
reconnect we pull the broker snapshot and diff it against local state, classifying
every discrepancy and producing the broker-authoritative corrected positions (the
broker is always the source of truth for what actually happened at the venue).

Three mismatch classes are detected, each independently tested:

* ``extra_fill`` -- a fill the broker has that local never recorded (filled during
  the outage). Local must adopt it.
* ``missed_fill`` -- a fill local recorded that the broker has no record of (a
  local optimistic fill that never reached / was rejected by the venue).
* ``position_mismatch`` -- a per-instrument net-position quantity that differs
  between local and broker even after accounting for fills (a bookkeeping error).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from futures_engine.execution.client import Position

_log = logging.getLogger("futures_engine.execution.reconcile")

MismatchKind = Literal["extra_fill", "missed_fill", "position_mismatch"]


@dataclass(frozen=True)
class Fill:
    """A single fill keyed by a venue-unique ``fill_id`` (``qty`` signed)."""

    fill_id: str
    instrument: str
    qty: int


@dataclass(frozen=True)
class LocalState:
    """The engine's local view: the fills it recorded and its net positions."""

    fills: list[Fill]
    positions: list[Position]


@dataclass(frozen=True)
class BrokerSnapshot:
    """The broker's authoritative view pulled on reconnect."""

    fills: list[Fill]
    positions: list[Position]


@dataclass(frozen=True)
class Discrepancy:
    """One classified difference between local and broker state."""

    kind: MismatchKind
    instrument: str
    detail: str


@dataclass(frozen=True)
class ReconcileReport:
    """The diff: every discrepancy plus the broker-authoritative corrections."""

    discrepancies: list[Discrepancy]
    corrected_positions: list[Position]

    @property
    def in_sync(self) -> bool:
        """Whether local and broker agreed (no discrepancies)."""
        return not self.discrepancies


def _net_by_instrument(positions: list[Position]) -> dict[str, int]:
    return {p.instrument: p.qty for p in positions if p.qty != 0}


def reconcile(local: LocalState, broker: BrokerSnapshot) -> ReconcileReport:
    """Diff ``local`` against the authoritative ``broker`` snapshot.

    Returns a :class:`ReconcileReport` whose ``corrected_positions`` are the
    broker's positions (the source of truth), and whose ``discrepancies`` list one
    entry per detected mismatch across the three classes.
    """
    discrepancies: list[Discrepancy] = []

    local_fills = {f.fill_id: f for f in local.fills}
    broker_fills = {f.fill_id: f for f in broker.fills}

    for fid in broker_fills.keys() - local_fills.keys():
        f = broker_fills[fid]
        discrepancies.append(
            Discrepancy(
                kind="extra_fill",
                instrument=f.instrument,
                detail=f"broker fill {fid} ({f.qty:+d} {f.instrument}) absent locally",
            )
        )

    for fid in local_fills.keys() - broker_fills.keys():
        f = local_fills[fid]
        discrepancies.append(
            Discrepancy(
                kind="missed_fill",
                instrument=f.instrument,
                detail=f"local fill {fid} ({f.qty:+d} {f.instrument}) absent at broker",
            )
        )

    local_net = _net_by_instrument(local.positions)
    broker_net = _net_by_instrument(broker.positions)
    for instrument in local_net.keys() | broker_net.keys():
        lq = local_net.get(instrument, 0)
        bq = broker_net.get(instrument, 0)
        if lq != bq:
            discrepancies.append(
                Discrepancy(
                    kind="position_mismatch",
                    instrument=instrument,
                    detail=f"{instrument} local net {lq:+d} != broker net {bq:+d}",
                )
            )

    for d in discrepancies:
        _log.warning("reconcile %s: %s", d.kind, d.detail)

    return ReconcileReport(
        discrepancies=discrepancies,
        corrected_positions=list(broker.positions),
    )
