"""Fixtures for the UI test suite.

The whole UI suite is skipped when the optional ``ui`` extra is not installed,
so the core (no-extras) suite stays green (UI-G2). Tests never start a live
server -- they drive the app in-process with FastAPI's ``TestClient`` (UI-G3).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="UI suite requires the 'ui' extra")

from fastapi.testclient import TestClient

from futures_engine.ui.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
