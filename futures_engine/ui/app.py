"""FastAPI application factory and console entry point for the research cockpit.

This module is the ONLY place in :mod:`futures_engine.ui` that imports the web
stack. The web deps ship in the optional ``ui`` extra (UI-G2); if they are not
installed, importing this module raises a clear :class:`ImportError` naming the
extra to install. ``import futures_engine`` never triggers this path.

The server binds ``127.0.0.1`` only (UI-G4) and HTMX is served from the local
``/static`` mount (UI-G3) -- never a CDN.
"""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - exercised via the lazy-import boundary test
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError as exc:  # pragma: no cover - trivial re-raise
    raise ImportError(
        "The research cockpit requires the optional 'ui' extra. "
        'Install it with:  pip install -e ".[ui]"'
    ) from exc

_PACKAGE_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _PACKAGE_ROOT / "static"
_TEMPLATES_DIR = _PACKAGE_ROOT / "templates"

HOST = "127.0.0.1"
PORT = 8000


def create_app() -> FastAPI:
    """Build and return the cockpit FastAPI app.

    Mounts ``/static`` (vendored HTMX + cockpit CSS), wires the Jinja template
    environment, and serves ``GET /`` (the index with the nav shell). This is a
    THIN scaffold: no pipeline calls yet (UI-G1).
    """
    app = FastAPI(title="futures-engine research cockpit")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    return app


def main() -> None:
    """Console entry point (``futures-cockpit``): run the local-only server."""
    import uvicorn

    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
