"""Tests for the cockpit Data screen: listing, synthetic generation, Databento gate (U3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from futures_engine.data.store import SnapshotStore, require_validation_grade
from futures_engine.ui.app import create_app
from futures_engine.ui.data_panel import (
    databento_enabled,
    generate_synthetic,
    list_snapshots,
)


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "snapshots"


@pytest.fixture
def data_client(store_root: Path) -> TestClient:
    return TestClient(create_app(snapshot_root=store_root))


# --- list_snapshots ---------------------------------------------------------


def test_list_snapshots_empty(store_root: Path) -> None:
    assert list_snapshots(store_root) == []


def test_list_snapshots_populated(store_root: Path) -> None:
    snapshot_hash = generate_synthetic(store_root, n_bars=120, seed=7)
    summaries = list_snapshots(store_root)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.hash == snapshot_hash
    assert summary.symbol_root == "MES"
    assert summary.interval == "1m"
    assert summary.validation_grade is True
    assert summary.has_continuous_meta is True


# --- generate_synthetic -----------------------------------------------------


def test_generate_synthetic_is_validation_grade_loadable(store_root: Path) -> None:
    snapshot_hash = generate_synthetic(store_root, n_bars=100, seed=3)
    _bars, meta = SnapshotStore(store_root).load(snapshot_hash)
    assert meta.validation_grade is True
    assert meta.continuous is not None
    require_validation_grade(meta)  # must not raise


# --- Databento gate ---------------------------------------------------------


def test_databento_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    assert databento_enabled() is False


def test_databento_enabled_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", "secret")
    assert databento_enabled() is True


# --- GET /data --------------------------------------------------------------


def test_get_data_empty_state(data_client: TestClient) -> None:
    resp = data_client.get("/data")
    assert resp.status_code == 200
    assert "No snapshots yet" in resp.text


def test_get_data_lists_snapshots(store_root: Path, data_client: TestClient) -> None:
    snapshot_hash = generate_synthetic(store_root, n_bars=120, seed=7)
    body = data_client.get("/data").text
    assert snapshot_hash[:12] in body
    assert "MES" in body


def test_get_data_databento_disabled_note(
    data_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    body = data_client.get("/data").text
    assert "DATABENTO_API_KEY" in body
    assert "disabled" in body.lower()


# --- POST /data/synthetic ---------------------------------------------------


def test_post_synthetic_creates_and_refreshes(store_root: Path, data_client: TestClient) -> None:
    resp = data_client.post(
        "/data/synthetic",
        data={"symbol_root": "MES", "n_bars": "120", "seed": "7"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/data"
    # The snapshot now exists and the refreshed list shows it.
    summaries = list_snapshots(store_root)
    assert len(summaries) == 1
    body = data_client.get("/data").text
    assert summaries[0].hash[:12] in body
