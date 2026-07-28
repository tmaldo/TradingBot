"""Adapter serialization / parsing against recorded fixtures (no network, G14/G15).

Tradovate is exercised fully (serialize, ack ok+reject, positions, account, WS
quote normalization); TopstepX to the doc-limited extent; Rithmic is a typed stub
whose every method raises. Every outbound order asserts ``isAutomated`` (CME 575).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from futures_engine.execution.adapters import AdapterError, MarketTick, RestTransport
from futures_engine.execution.adapters.rithmic_stub import RithmicExecutionClient
from futures_engine.execution.adapters.topstepx import TopstepXExecutionClient
from futures_engine.execution.adapters.tradovate import TradovateExecutionClient
from futures_engine.execution.client import ExecutionClient, Order
from futures_engine.execution.live_config import TopstepXAdapterConfig, TradovateAdapterConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class RecordedTransport:
    """A RestTransport that replays recorded fixtures keyed by path (offline)."""

    def __init__(self, posts: dict[str, str], gets: dict[str, str]) -> None:
        self._posts = posts
        self._gets = gets
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.sent.append((path, body))
        return _fixture(self._posts[path])

    def get(self, path: str) -> dict[str, Any]:
        return _fixture(self._gets[path])


def _protocol_smoke() -> None:
    assert isinstance(RecordedTransport({}, {}), RestTransport)


TRADOVATE_CFG = TradovateAdapterConfig(
    base_url="https://demo.tradovateapi.com/v1",
    ws_url="wss://demo.tradovateapi.com/v1/websocket",
    account_spec="MFFU_DEMO",
    account_id=1000001,
)
TOPSTEPX_CFG = TopstepXAdapterConfig(
    base_url="https://gateway-api-demo.s2f.projectx.com", account_id=2000002
)


def _order(oid: str = "o1", side: str = "buy", qty: int = 2, otype: str = "market") -> Order:
    return Order(client_order_id=oid, instrument="MESU4", side=side, qty=qty, type=otype)  # type: ignore[arg-type]


# --- Tradovate (full) --------------------------------------------------------


def test_tradovate_is_execution_client() -> None:
    adapter = TradovateExecutionClient(TRADOVATE_CFG, RecordedTransport({}, {}))
    assert isinstance(adapter, ExecutionClient)


def test_tradovate_serialize_sets_is_automated_true() -> None:
    adapter = TradovateExecutionClient(TRADOVATE_CFG, RecordedTransport({}, {}))
    body = adapter.serialize_order(_order(side="buy", qty=3))
    assert body["isAutomated"] is True  # CME Rule 575
    assert body["action"] == "Buy"
    assert body["orderQty"] == 3
    assert body["orderType"] == "Market"
    assert body["accountId"] == 1000001


def test_tradovate_serialize_rejects_non_automated_order() -> None:
    adapter = TradovateExecutionClient(TRADOVATE_CFG, RecordedTransport({}, {}))
    manual = Order(
        client_order_id="m1",
        instrument="MESU4",
        side="buy",
        qty=1,
        type="market",
        is_automated=False,
    )
    with pytest.raises(AdapterError):
        adapter.serialize_order(manual)


def test_tradovate_submit_accepts_from_ack_fixture() -> None:
    transport = RecordedTransport({"/order/placeorder": "tradovate_place_ack.json"}, {})
    adapter = TradovateExecutionClient(TRADOVATE_CFG, transport)
    ack = adapter.submit(_order("o1"))
    assert ack.accepted and ack.reason is None
    # every outbound order carried the automated flag.
    assert transport.sent[0][1]["isAutomated"] is True


def test_tradovate_submit_rejects_from_reject_fixture() -> None:
    transport = RecordedTransport({"/order/placeorder": "tradovate_place_reject.json"}, {})
    adapter = TradovateExecutionClient(TRADOVATE_CFG, transport)
    ack = adapter.submit(_order("o2"))
    assert not ack.accepted
    assert "max position" in (ack.reason or "").lower()


def test_tradovate_positions_parsed_and_zero_filtered() -> None:
    transport = RecordedTransport({}, {"/position/list": "tradovate_positions.json"})
    adapter = TradovateExecutionClient(TRADOVATE_CFG, transport)
    positions = adapter.positions()
    assert len(positions) == 1  # the netPos==0 row is filtered out
    assert positions[0].instrument == "MESU4"
    assert positions[0].qty == 2
    assert positions[0].avg_px == 5012.25


def test_tradovate_account_parsed() -> None:
    transport = RecordedTransport(
        {},
        {
            "/cashBalance/getcashbalance": "tradovate_account.json",
            "/position/list": "tradovate_positions.json",
        },
    )
    adapter = TradovateExecutionClient(TRADOVATE_CFG, transport)
    state = adapter.account()
    assert state.balance == 50000.0
    assert state.equity == pytest.approx(50125.50)


def test_tradovate_normalize_quote() -> None:
    adapter = TradovateExecutionClient(TRADOVATE_CFG, RecordedTransport({}, {}))
    tick = adapter.normalize_quote(_fixture("tradovate_md_quote.json"))
    assert isinstance(tick, MarketTick)
    assert tick.instrument == "MESU4"
    assert tick.price == 5013.5
    assert tick.ts == 1719849600.0


def test_tradovate_normalize_quote_bad_frame_raises() -> None:
    adapter = TradovateExecutionClient(TRADOVATE_CFG, RecordedTransport({}, {}))
    with pytest.raises(AdapterError):
        adapter.normalize_quote({"garbage": True})


# --- TopstepX (doc-limited) --------------------------------------------------


def test_topstepx_is_execution_client() -> None:
    adapter = TopstepXExecutionClient(TOPSTEPX_CFG, RecordedTransport({}, {}))
    assert isinstance(adapter, ExecutionClient)


def test_topstepx_serialize_sets_automated_and_enum_mapping() -> None:
    adapter = TopstepXExecutionClient(TOPSTEPX_CFG, RecordedTransport({}, {}))
    body = adapter.serialize_order(_order(side="sell", qty=1, otype="market"))
    assert body["isAutomated"] is True  # CME Rule 575 (best-effort field)
    assert body["type"] == 2  # Market
    assert body["side"] == 1  # Ask/sell
    assert body["size"] == 1
    assert body["accountId"] == 2000002


def test_topstepx_submit_accepts_and_rejects() -> None:
    ok_t = RecordedTransport({"/api/Order/place": "topstepx_place_ack.json"}, {})
    assert TopstepXExecutionClient(TOPSTEPX_CFG, ok_t).submit(_order("t1")).accepted
    bad_t = RecordedTransport({"/api/Order/place": "topstepx_place_reject.json"}, {})
    ack = TopstepXExecutionClient(TOPSTEPX_CFG, bad_t).submit(_order("t2"))
    assert not ack.accepted
    assert "buying power" in (ack.reason or "").lower()


def test_topstepx_positions_signed_by_type() -> None:
    transport = RecordedTransport({"/api/Position/searchOpen": "topstepx_positions.json"}, {})
    adapter = TopstepXExecutionClient(TOPSTEPX_CFG, transport)
    positions = {p.instrument: p.qty for p in adapter.positions()}
    assert positions["CON.F.US.MES.U24"] == 2  # type 1 = long
    assert positions["CON.F.US.MNQ.U24"] == -1  # type 2 = short


def test_topstepx_account_parsed() -> None:
    transport = RecordedTransport(
        {
            "/api/Account/search": "topstepx_account.json",
            "/api/Position/searchOpen": "topstepx_positions.json",
        },
        {},
    )
    state = TopstepXExecutionClient(TOPSTEPX_CFG, transport).account()
    assert state.balance == 50250.0


# --- Rithmic (typed stub) ----------------------------------------------------


def test_rithmic_is_execution_client_type() -> None:
    assert isinstance(RithmicExecutionClient(), ExecutionClient)


def test_rithmic_methods_raise_not_implemented() -> None:
    adapter = RithmicExecutionClient()
    with pytest.raises(NotImplementedError):
        adapter.submit(_order())
    with pytest.raises(NotImplementedError):
        adapter.cancel("x")
    with pytest.raises(NotImplementedError):
        adapter.positions()
    with pytest.raises(NotImplementedError):
        adapter.account()
    with pytest.raises(NotImplementedError):
        adapter.on_disconnect(lambda: None)
    with pytest.raises(NotImplementedError):
        adapter.on_data_stale(lambda: None)
