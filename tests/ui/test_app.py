"""Smoke + lazy-import-boundary tests for the research cockpit scaffold (U1)."""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_create_app_returns_fastapi() -> None:
    from futures_engine.ui.app import create_app

    assert isinstance(create_app(), FastAPI)


def test_index_ok_with_nav_links(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    for label in ("Configure", "Data", "Runs"):
        assert label in body


def test_index_loads_htmx_from_local_static_not_cdn(client: TestClient) -> None:
    body = client.get("/").text
    assert 'src="/static/htmx.min.js"' in body
    # UI-G3: HTMX is vendored -- no CDN reference anywhere in the rendered page.
    for cdn in ("unpkg.com", "cdn.jsdelivr.net", "cdnjs", "https://"):
        assert cdn not in body


def test_static_htmx_served(client: TestClient) -> None:
    resp = client.get("/static/htmx.min.js")
    assert resp.status_code == 200
    assert "htmx" in resp.text.lower()


def test_main_is_wired() -> None:
    from futures_engine.ui import app as app_module

    assert callable(app_module.main)


# --- Lazy-import boundary (UI-G2) -------------------------------------------


def test_importing_futures_engine_does_not_pull_in_ui() -> None:
    """`import futures_engine` must not transitively import the web stack."""
    for name in list(sys.modules):
        if name == "futures_engine" or name.startswith("futures_engine.ui"):
            del sys.modules[name]
    importlib.import_module("futures_engine")
    assert "futures_engine.ui.app" not in sys.modules


def test_ui_package_import_is_cheap() -> None:
    """Importing the ui package pulls in no web dependency by itself."""
    importlib.import_module("futures_engine.ui")  # must not raise


def test_app_module_raises_clear_error_without_fastapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the missing-extra path: a clear ImportError naming the extra."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("No module named 'fastapi'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    for name in list(sys.modules):
        if name.startswith("futures_engine.ui.app"):
            del sys.modules[name]
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"ui.*extra|extra.*ui"):
        importlib.import_module("futures_engine.ui.app")

    # Restore a clean, importable module for subsequent tests.
    for name in list(sys.modules):
        if name.startswith("futures_engine.ui.app"):
            del sys.modules[name]
    monkeypatch.undo()
    importlib.import_module("futures_engine.ui.app")
