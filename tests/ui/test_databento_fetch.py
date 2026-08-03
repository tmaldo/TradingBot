"""U7: offline Databento real-data fetch (fixtures + stubbed adapter, NO network).

Two layers per the brief:
  1. Pin the PURE mapper ``parse_databento_bars`` on a recorded records fixture
     (1e-9 fixed-point -> float64, ``ts_event`` -> bar-OPEN index, OHLCV columns).
  2. Drive ``fetch_databento_snapshot`` end-to-end with a STUBBED adapter
     (monkeypatch the network methods ``_list_raw`` / ``_fetch_raw`` to return
     recorded fixture records) so the real ``parse_*`` + ``build_continuous`` +
     ``SnapshotStore.save`` run offline -> a validation-grade snapshot with
     ``ContinuousMeta``. Plus: error surfaces no partial snapshot, and the API key
     is never logged nor persisted into the meta/manifest.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from futures_engine.data.adapters.databento_adapter import (
    DatabentoAdapter,
    parse_databento_bars,
)
from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.ui.app import create_app
from futures_engine.ui.data_panel import fetch_databento_snapshot

_FIXTURES = Path(__file__).parent / "fixtures"
_SECRET_KEY = "sk-SUPER-SECRET-databento-KEY-do-not-persist"


def _load_bars_fixture() -> dict[str, list[dict[str, object]]]:
    return json.loads((_FIXTURES / "mes_bars.json").read_text(encoding="utf-8"))


def _load_defs_fixture() -> list[dict[str, object]]:
    return json.loads((_FIXTURES / "mes_defs.json").read_text(encoding="utf-8"))


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "snapshots"


@pytest.fixture
def stub_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the adapter's network seams to serve recorded fixtures (offline)."""
    bars = _load_bars_fixture()
    defs = _load_defs_fixture()

    def fake_list_raw(self: DatabentoAdapter, symbol_root: str, start: object, end: object):
        return defs

    def fake_fetch_raw(
        self: DatabentoAdapter, contract: str, start: object, end: object, interval: object
    ):
        return bars[contract]

    monkeypatch.setattr(DatabentoAdapter, "_list_raw", fake_list_raw)
    monkeypatch.setattr(DatabentoAdapter, "_fetch_raw", fake_fetch_raw)


# --- Layer 1: pure mapper (1e-9 scaling, ts_event=open, OHLCV columns) -------


def test_parse_databento_bars_pins_scaling_and_open_index() -> None:
    records = _load_bars_fixture()["MESM4"]
    bars = parse_databento_bars(records)
    # columns are exactly the OHLCV contract, in order
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert str(bars.index.tz) == "UTC"
    # index is float64 for prices (fixed-point * 1e-9)
    assert all(str(bars[c].dtype) == "float64" for c in ("open", "high", "low", "close"))
    # first record: raw open 4995e9 -> 4995.0, close 5000e9 -> 5000.0
    assert bars["open"].iloc[0] == pytest.approx(4995.0)
    assert bars["close"].iloc[0] == pytest.approx(5000.0)
    # ts_event maps to the bar-OPEN index label (first record ts_event = 2024-06-03)
    import pandas as pd

    assert bars.index[0] == pd.Timestamp("2024-06-03T00:00:00Z")


# --- Layer 2: end-to-end offline snapshot via stubbed adapter ----------------


def test_fetch_databento_snapshot_offline_validation_grade(
    store_root: Path, stub_adapter: None
) -> None:
    snapshot_hash = fetch_databento_snapshot(
        store_root,
        symbol_root="MES",
        start=datetime(2024, 6, 1),
        end=datetime(2024, 6, 30),
        roll_rule="volume",
        adjustment="panama_diff",
        api_key=_SECRET_KEY,
    )
    bars, meta = SnapshotStore(store_root).load(snapshot_hash)
    require_validation_grade(meta)  # validation-grade + ContinuousMeta present
    assert meta.source == "databento"
    assert meta.validation_grade is True
    assert meta.continuous is not None
    assert meta.continuous.roll_rule == "volume"
    assert meta.continuous.adjustment == "panama_diff"
    assert meta.continuous.underlying_contracts == ["MESM4", "MESU4"]
    # continuous series spans both contracts, stitched at the roll
    assert len(bars) == 7
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


# --- Error path: no partial snapshot on failure -----------------------------


def test_fetch_error_writes_no_partial_snapshot(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defs = _load_defs_fixture()
    monkeypatch.setattr(DatabentoAdapter, "_list_raw", lambda self, *a, **k: defs)

    def boom(self: DatabentoAdapter, *a: object, **k: object):
        raise RuntimeError("simulated Databento HTTP 402 (bad key / quota)")

    monkeypatch.setattr(DatabentoAdapter, "_fetch_raw", boom)

    with pytest.raises(RuntimeError, match="simulated Databento"):
        fetch_databento_snapshot(
            store_root,
            symbol_root="MES",
            start=datetime(2024, 6, 1),
            end=datetime(2024, 6, 30),
            roll_rule="volume",
            adjustment="panama_diff",
            api_key=_SECRET_KEY,
        )
    # No snapshot artifacts were written (save only on a clean, complete fetch).
    written = list(store_root.glob("*.parquet")) + list(store_root.glob("*.meta.json"))
    assert written == []


# --- Key hygiene: never persisted into meta/manifest ------------------------


def test_api_key_never_persisted(store_root: Path, stub_adapter: None) -> None:
    snapshot_hash = fetch_databento_snapshot(
        store_root,
        symbol_root="MES",
        start=datetime(2024, 6, 1),
        end=datetime(2024, 6, 30),
        roll_rule="volume",
        adjustment="panama_diff",
        api_key=_SECRET_KEY,
    )
    meta_text = (store_root / f"{snapshot_hash}.meta.json").read_text(encoding="utf-8")
    assert _SECRET_KEY not in meta_text
    assert "SECRET" not in meta_text
    # Meta records provenance + hash, not the key.
    assert '"source":"databento"' in meta_text.replace(" ", "")
    assert snapshot_hash in meta_text


# --- Route: POST /data/fetch (gated on key presence) ------------------------


@pytest.fixture
def keyed_client(
    store_root: Path, stub_adapter: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABENTO_API_KEY", _SECRET_KEY)
    yield TestClient(create_app(snapshot_root=store_root))


def test_post_data_fetch_creates_snapshot(store_root: Path, keyed_client: TestClient) -> None:
    resp = keyed_client.post(
        "/data/fetch",
        data={
            "symbol_root": "MES",
            "start": "2024-06-01",
            "end": "2024-06-30",
            "roll_rule": "volume",
            "adjustment": "panama_diff",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/data"
    assert len(list(SnapshotStore(store_root).list())) == 1


def test_post_data_fetch_refused_without_key(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    client = TestClient(create_app(snapshot_root=store_root))
    resp = client.post(
        "/data/fetch",
        data={
            "symbol_root": "MES",
            "start": "2024-06-01",
            "end": "2024-06-30",
            "roll_rule": "volume",
            "adjustment": "panama_diff",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert list(store_root.glob("*.parquet")) == []


def test_post_data_fetch_error_rerenders_no_partial(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", _SECRET_KEY)
    defs = _load_defs_fixture()
    monkeypatch.setattr(DatabentoAdapter, "_list_raw", lambda self, *a, **k: defs)
    monkeypatch.setattr(
        DatabentoAdapter,
        "_fetch_raw",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom-net")),
    )
    client = TestClient(create_app(snapshot_root=store_root))
    resp = client.post(
        "/data/fetch",
        data={
            "symbol_root": "MES",
            "start": "2024-06-01",
            "end": "2024-06-30",
            "roll_rule": "volume",
            "adjustment": "panama_diff",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "error" in resp.text.lower() or "failed" in resp.text.lower()
    assert list(store_root.glob("*.parquet")) == []
