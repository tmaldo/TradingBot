"""Live broker adapters -- thin, fixture-tested implementations of the shared
:class:`~futures_engine.execution.client.ExecutionClient` Protocol (G14 parity).

Scope (architect decision #6, do not re-litigate):

* :class:`~futures_engine.execution.adapters.tradovate.TradovateExecutionClient`
  -- FULL implementation (MFFU route; paper/demo REST+WS endpoints).
* :class:`~futures_engine.execution.adapters.topstepx.TopstepXExecutionClient`
  -- typed + fixture-tested to the extent the public ProjectX gateway docs allow.
* :class:`~futures_engine.execution.adapters.rithmic_stub.RithmicExecutionClient`
  -- typed stub; every method raises ``NotImplementedError``.

Every adapter is driven by an injected :class:`RestTransport`; in CI that is a
recorded-fixture transport, so **no network call is ever made** (G15). Every
outbound order carries ``is_automated=True`` (CME Rule 575), enforced at
serialization and asserted in the adapter tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RestTransport(Protocol):
    """Minimal request/response seam an adapter uses to reach a venue REST API.

    Production wires this to an authenticated HTTP client; tests wire it to a
    recorded-fixture transport so the adapter logic is exercised fully offline.
    """

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``path`` and return the decoded JSON response."""
        ...

    def get(self, path: str) -> dict[str, Any]:
        """GET ``path`` and return the decoded JSON response."""
        ...


@dataclass(frozen=True)
class MarketTick:
    """A normalized market-data tick emitted by an adapter's WS handler.

    ``ts`` is epoch seconds; the staleness clock is driven from these.
    """

    instrument: str
    price: float
    ts: float


class AdapterError(RuntimeError):
    """Raised when a venue response cannot be interpreted."""


__all__ = ["AdapterError", "MarketTick", "RestTransport"]
