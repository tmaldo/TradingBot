"""FastAPI application factory and console entry point for the research cockpit.

This module is the ONLY place in :mod:`futures_engine.ui` that imports the web
stack. The web deps ship in the optional ``ui`` extra (UI-G2); if they are not
installed, importing this module raises a clear :class:`ImportError` naming the
extra to install. ``import futures_engine`` never triggers this path.

The server binds ``127.0.0.1`` only (UI-G4) and HTMX is served from the local
``/static`` mount (UI-G3) -- never a CDN.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - exercised via the lazy-import boundary test
    from fastapi import FastAPI, Form, Request
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError as exc:  # pragma: no cover - trivial re-raise
    raise ImportError(
        "The research cockpit requires the optional 'ui' extra. "
        'Install it with:  pip install -e ".[ui]"'
    ) from exc

from futures_engine.ui.data_panel import (
    databento_enabled,
    generate_synthetic,
    list_snapshots,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _PACKAGE_ROOT / "static"
_TEMPLATES_DIR = _PACKAGE_ROOT / "templates"

HOST = "127.0.0.1"
PORT = 8000

# Where snapshots live on disk. Overridable via env so tests (and operators) can
# point the cockpit at an isolated store without touching the repo default.
_DEFAULT_SNAPSHOT_ROOT = "data/snapshots"


def create_app(*, snapshot_root: str | Path | None = None) -> FastAPI:
    """Build and return the cockpit FastAPI app.

    Mounts ``/static`` (vendored HTMX + cockpit CSS), wires the Jinja template
    environment, and serves the nav shell plus the Data screen. The UI stays
    thin (UI-G1): routes only orchestrate the store + shared generator.

    ``snapshot_root`` selects the snapshot store; it defaults to
    ``$FE_SNAPSHOT_ROOT`` or ``data/snapshots``.
    """
    root = Path(
        snapshot_root
        if snapshot_root is not None
        else os.environ.get("FE_SNAPSHOT_ROOT", _DEFAULT_SNAPSHOT_ROOT)
    )
    app = FastAPI(title="futures-engine research cockpit")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    @app.get("/data", response_class=HTMLResponse)
    def data(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "data.html",
            {
                "snapshots": list_snapshots(root),
                "databento_enabled": databento_enabled(),
            },
        )

    @app.post("/data/synthetic")
    def create_synthetic(
        symbol_root: str = Form("MES"),
        n_bars: int = Form(2500),
        seed: int = Form(7),
    ) -> RedirectResponse:
        generate_synthetic(root, symbol_root=symbol_root, n_bars=n_bars, seed=seed)
        # POST/redirect/GET: refresh the list without re-submitting on reload.
        return RedirectResponse(url="/data", status_code=303)

    return app


def main() -> None:
    """Console entry point (``futures-cockpit``): run the local-only server."""
    import uvicorn

    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
