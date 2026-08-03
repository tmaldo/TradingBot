"""Fixtures for the UI test suite.

The whole UI suite is skipped when the optional ``ui`` extra is not installed,
so the core (no-extras) suite stays green (UI-G2). Tests never start a live
server -- they drive the app in-process with FastAPI's ``TestClient`` (UI-G3).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi", reason="UI suite requires the 'ui' extra")

from fastapi.testclient import TestClient

from futures_engine.ui.app import create_app


@pytest.fixture(autouse=True)
def _restore_futures_engine_modules() -> Iterator[None]:
    """Guarantee ``futures_engine`` module identity is stable across tests.

    The U1 lazy-import-boundary tests deliberately ``del`` and re-import
    ``futures_engine.ui.*`` from :data:`sys.modules`. Left un-restored, that swaps
    a later test's collection-time function references for stale module objects,
    which breaks pickling the process-pool callable by reference (UI-G5): pickle
    verifies ``sys.modules[func.__module__].<name> is func`` and the mismatch
    surfaces as a failed run. Snapshotting and restoring the original module
    objects after every test keeps the pool callable picklable regardless of order.
    """
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith("futures_engine")}
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n.startswith("futures_engine")]:
            if sys.modules.get(name) is not saved.get(name):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
